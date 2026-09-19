"""Question to cited answer (features 34-41, 48-52).

The order of operations is the safety property. Retrieval decides admissibility first; only
if evidence clears the relevance floor is a model called at all. A refusal is therefore not
something the model chooses to say - it happens before generation exists, which is the only
way to guarantee it.

If every provider fails, the system degrades rather than dies: the reranked evidence is
returned verbatim with its citations and an explicit notice that generation is unavailable.
The officer still gets the correct source pages, which was the actual need (feature 51).

Both entry points - ``answer()`` and the streaming ``stream()`` the interface uses - share
one path in front of retrieval (screening, the session cap, the cache) and one behind it
(session accounting, caching, the query log). Guardrails first lived in ``answer()`` alone,
and each time the streaming path quietly went without them.
"""

from __future__ import annotations

import time
from collections.abc import Iterator
from dataclasses import dataclass, field
from typing import Any

from citara.config import Settings, get_settings
from citara.generation.citations import build_citations, validate
from citara.generation.models import Citation, GeneratedAnswer
from citara.generation.prompts import SYSTEM_PROMPT, build_user_prompt
from citara.generation.providers import Provider, available_providers
from citara.guardrails.injection import screen_input
from citara.guardrails.query_log import record_query
from citara.guardrails.scope import SCOPE_MESSAGE, is_out_of_scope
from citara.log import get_logger, stage
from citara.resilience.budget import RequestBudget, SessionLimiter
from citara.resilience.cache import AnswerCache, cache_key
from citara.retrieval.models import RetrievalOutcome, RetrievedChunk
from citara.retrieval.query import looks_like_follow_up
from citara.retrieval.retriever import HybridRetriever

log = get_logger("generation")

_SESSION_CAP_MESSAGE = (
    "This session has reached its question limit. The limit exists so one visitor cannot "
    "exhaust the shared free-tier quota this prototype runs on. Reload the page to continue."
)

_BLOCKED_MESSAGE = (
    "That request looks like an attempt to change how this assistant works. CITARA answers "
    "questions from NDMA's published documents and cannot take instructions from a message."
)

_DEGRADED_NOTICE = (
    "Answer generation is unavailable right now, so the assistant cannot summarise. "
    "The most relevant passages from the indexed NDMA corpus are shown below, with their "
    "sources, so they can be read directly."
)


@dataclass
class _Turn:
    """One question on its way through the pipeline."""

    question: str
    key: str
    started: float = field(default_factory=time.perf_counter)


class Answerer:
    """Retrieval, grounded generation, citation validation and failover."""

    def __init__(
        self,
        settings: Settings | None = None,
        retriever: HybridRetriever | None = None,
        providers: list[Provider] | None = None,
    ) -> None:
        self.settings = settings or get_settings()
        self.retriever = retriever or HybridRetriever(self.settings)
        self.providers = providers if providers is not None else available_providers(self.settings)
        paths = self.settings.paths
        resilience = self.settings.resilience
        self.cache = AnswerCache(
            paths.resolved(paths.cache_dir), resilience.cache_ttl_s, resilience.cache_max_entries
        )
        # Read here, counted where requests are actually sent: in the providers and the query
        # rewriter, per attempt (feature 48).
        self.budget = RequestBudget.from_settings(self.settings)
        self.sessions = SessionLimiter(resilience.session_query_cap)

    def warm_up(self) -> None:
        """Load models and clients before the first question arrives (feature 64).

        Measured cold, the first answer took 48.9 s and almost none of it was answering. This
        spends no quota: provider clients are built, not called.
        """
        with stage(log, "answerer.warm_up", providers=len(self.providers)):
            self.retriever.warm_up()
            for provider in self.providers:
                provider.warm_up()

    # -- entry points -------------------------------------------------------------------

    def answer(
        self,
        question: str,
        history: list[str] | None = None,
        doc_ids: list[str] | None = None,
        year: int | None = None,
        session_id: str = "",
    ) -> GeneratedAnswer:
        """Answer *question*, or refuse."""
        turn, early = self._begin(question, history, doc_ids, year, session_id)
        if early is not None:
            return early

        outcome = self.retriever.retrieve(question, history=history, doc_ids=doc_ids, year=year)
        answer = self.answer_from(question, outcome)
        self._finish(turn, answer, outcome)
        return answer

    def stream(
        self,
        question: str,
        history: list[str] | None = None,
        doc_ids: list[str] | None = None,
        year: int | None = None,
        session_id: str = "",
    ) -> Iterator[tuple[str, GeneratedAnswer | None]]:
        """Yield answer text as it arrives, then the finished answer.

        Each item is ``(text_piece, None)`` while generating and ``("", answer)`` at the end,
        so a caller can render tokens as they arrive and still receive validated citations.
        Three seconds of blank screen reads as a crash in a live demo, which is the whole
        reason this path exists (feature 41).

        Anything not generated live - a refusal, a cached answer, a degraded one - is
        delivered complete in a single item.
        """
        turn, early = self._begin(question, history, doc_ids, year, session_id)
        if early is not None:
            yield early.text, early
            return

        outcome = self.retriever.retrieve(question, history=history, doc_ids=doc_ids, year=year)
        # Nothing to stream when nothing will be generated: answer_from() refuses, or turns a
        # spent budget into cited evidence, exactly as it does for answer().
        if not outcome.has_evidence or self.budget.state().exhausted:
            answer = self.answer_from(question, outcome)
            self._finish(turn, answer, outcome)
            yield answer.text, answer
            return

        user_prompt, injection_flags = build_user_prompt(question, outcome.results)
        answer = GeneratedAnswer(
            text="",
            citations=build_citations(outcome.results),
            evidence=outcome.results,
            retrieval_ms=outcome.retrieval_ms + outcome.rerank_ms,
            injection_flags=injection_flags,
        )

        with stage(log, "generate", providers=len(self.providers), streaming=True) as details:
            started = time.perf_counter()
            for index, provider in enumerate(self.providers):
                pieces: list[str] = []
                try:
                    for piece in provider.stream(SYSTEM_PROMPT, user_prompt):
                        pieces.append(piece)
                        yield piece, None
                except Exception as error:
                    log.warning(
                        "streaming provider failed",
                        extra={"provider": provider.name, "error": str(error)[:200]},
                    )
                    answer.error = f"{provider.name}: {type(error).__name__}"
                    # Anything already shown is discarded rather than spliced onto another
                    # provider's output, which would read as one answer contradicting itself.
                    continue

                answer.text = "".join(pieces).strip()
                answer.provider = provider.name
                answer.model = provider.model
                answer.failover_used = index > 0
                answer.mode = "generated"
                break
            else:
                answer = self._degrade(answer)
                yield answer.text, None

            answer.generation_ms = round((time.perf_counter() - started) * 1000, 1)
            self._check_citations(answer)
            details.update({"mode": answer.mode, "provider": answer.provider})

        self._finish(turn, answer, outcome)
        yield "", answer

    # -- shared path around retrieval ---------------------------------------------------

    def _begin(
        self,
        question: str,
        history: list[str] | None,
        doc_ids: list[str] | None,
        year: int | None,
        session_id: str,
    ) -> tuple[_Turn, GeneratedAnswer | None]:
        """Everything before retrieval, returning an answer when the turn ends here.

        Screening and scope control come first: an instruction-override attempt or a question
        about French geography should not reach the corpus, the model, or the quota. The cache
        is consulted after the session cap and a hit counts against neither the cap nor the
        budget, because it spends no provider quota - which is what both limits protect.
        """
        turn = _Turn(question=question, key=self._cache_key(question, history, doc_ids, year))

        early = self.preflight(question)
        if early is None and session_id and not self.sessions.allows(session_id):
            early = GeneratedAnswer(text=_SESSION_CAP_MESSAGE, mode="refused")
        if early is None and self.settings.resilience.enable_cache:
            early = self._cached(turn.key)

        if early is not None:
            self._record(question, early, turn.started)
        elif session_id:
            # Counted as the turn heads for retrieval, not when it completes: a stream the
            # visitor abandons part-way has still spent quota, and must still count.
            self.sessions.record(session_id)
        return turn, early

    def _finish(self, turn: _Turn, answer: GeneratedAnswer, outcome: RetrievalOutcome) -> None:
        """Account for a turn that reached retrieval."""
        # An answer citing evidence that does not exist is a defect; caching it would replay
        # the defect to everyone who asks for the next day.
        if (
            answer.mode == "generated"
            and not answer.invalid_markers
            and self.settings.resilience.enable_cache
        ):
            self.cache.set(turn.key, self._to_cache(answer))
        self._record(turn.question, answer, turn.started, outcome)

    def _cache_key(
        self,
        question: str,
        history: list[str] | None,
        doc_ids: list[str] | None,
        year: int | None,
    ) -> str:
        """Key on exactly what shapes the answer, and nothing that does not.

        History changes retrieval only for a follow-up, and then only through the turns the
        rewriter reads. Keying on more would make a showcase question miss the cache merely
        because it was asked mid-conversation; keying on less would let two different
        conversations share one follow-up's answer.
        """
        turns = self.settings.retrieval.history_turns
        window = (history or [])[-turns:] if turns else []
        context = "\n".join(window) if window and looks_like_follow_up(question) else ""
        return cache_key(
            question,
            self.settings.fingerprint(),
            docs=",".join(sorted(doc_ids or [])),
            year=year or "",
            history=context,
        )

    @staticmethod
    def _to_cache(answer: GeneratedAnswer) -> dict[str, Any]:
        """The answer, its citations, and a reference to each piece of evidence behind it."""
        return {
            "text": answer.text,
            "provider": answer.provider,
            "model": answer.model,
            "citations": [
                {
                    "marker": c.marker,
                    "citation": c.citation,
                    "chunk_id": c.chunk_id,
                    "doc_id": c.doc_id,
                    "page_start": c.page_start,
                    "page_end": c.page_end,
                    "used": c.used,
                }
                for c in answer.citations
            ],
            # References, not text: the passages are rebuilt from the live index, so a cached
            # answer shows the same sources as a fresh one without storing the corpus twice.
            "evidence": [
                {
                    "chunk_id": r.chunk_id,
                    "dense_rank": r.dense_rank,
                    "dense_score": r.dense_score,
                    "sparse_rank": r.sparse_rank,
                    "sparse_score": r.sparse_score,
                    "fusion_score": r.fusion_score,
                    "rerank_score": r.rerank_score,
                }
                for r in answer.evidence
            ],
        }

    def _cached(self, key: str) -> GeneratedAnswer | None:
        """Rebuild a cached answer, or None when there is none that can still be trusted.

        An entry whose evidence is no longer in the index is treated as a miss: the index was
        rebuilt, and an answer whose sources cannot be shown is not one to serve.
        """
        payload = self.cache.get(key)
        if not payload:
            return None

        chunks = self.retriever.chunks_by_id
        references: list[dict[str, Any]] = list(payload.get("evidence") or [])
        if not references or any(ref.get("chunk_id") not in chunks for ref in references):
            log.info("cached answer no longer matches the index; regenerating")
            return None

        evidence = [
            RetrievedChunk(
                chunk=chunks[ref["chunk_id"]],
                dense_rank=ref.get("dense_rank"),
                dense_score=ref.get("dense_score"),
                sparse_rank=ref.get("sparse_rank"),
                sparse_score=ref.get("sparse_score"),
                fusion_score=float(ref.get("fusion_score") or 0.0),
                rerank_score=ref.get("rerank_score"),
            )
            for ref in references
        ]
        citations = [
            Citation(
                marker=int(item["marker"]),
                citation=str(item["citation"]),
                chunk_id=str(item["chunk_id"]),
                doc_id=str(item["doc_id"]),
                page_start=int(item["page_start"]),
                page_end=int(item["page_end"]),
                used=bool(item["used"]),
            )
            for item in payload.get("citations", [])
        ]
        return GeneratedAnswer(
            text=str(payload.get("text", "")),
            mode="generated",
            citations=citations,
            evidence=evidence,
            provider=str(payload.get("provider", "")),
            model=str(payload.get("model", "")),
            cached=True,
        )

    def preflight(self, question: str) -> GeneratedAnswer | None:
        """Guardrail checks that run before anything else, or None to continue.

        Every entry point must pass through this. Screening lived only in answer() at first,
        so the streaming path - the one the interface uses - reached retrieval unscreened and
        was protected only by whatever the relevance floor happened to reject.
        """
        screening = screen_input(question)
        if screening.blocked:
            return GeneratedAnswer(
                text=_BLOCKED_MESSAGE,
                mode="blocked",
                screening=",".join(screening.reasons),
            )
        if is_out_of_scope(question):
            return GeneratedAnswer(text=SCOPE_MESSAGE, mode="out_of_scope")
        return None

    def _record(
        self,
        question: str,
        answer: GeneratedAnswer,
        started: float,
        outcome: RetrievalOutcome | None = None,
    ) -> None:
        """Append the query to the non-identifying log (feature 46)."""
        if not self.settings.guardrails.log_queries:
            return
        record_query(
            self.settings.paths.resolved(self.settings.paths.query_log_path),
            question,
            rewritten=outcome.rewritten_query if outcome else "",
            mode=answer.mode,
            refused=answer.refused,
            refusal_reason=outcome.refusal_reason if outcome else "",
            best_score=outcome.best_score if outcome else None,
            evidence=len(answer.evidence),
            documents=[result.chunk.doc_id for result in answer.evidence],
            latency_ms=(time.perf_counter() - started) * 1000,
            screening=answer.screening,
        )

    # -- generation ---------------------------------------------------------------------

    def answer_from(self, question: str, outcome: RetrievalOutcome) -> GeneratedAnswer:
        """Generate from an existing retrieval outcome.

        Screening repeats here even though the entry points already ran it: this method is
        public, and the check costs a few regular expressions against the cost of sending an
        injection attempt to a model.
        """
        blocked = self.preflight(question)
        if blocked is not None:
            return blocked

        if not outcome.has_evidence:
            log.info("refusing", extra={"reason": outcome.refusal_reason, "question": question})
            return GeneratedAnswer(
                text=self.settings.generation.refusal_message,
                mode="refused",
                retrieval_ms=outcome.retrieval_ms + outcome.rerank_ms,
                evidence=[],
            )

        citations = build_citations(outcome.results)
        user_prompt, injection_flags = build_user_prompt(question, outcome.results)
        answer = GeneratedAnswer(
            text="",
            citations=citations,
            evidence=outcome.results,
            retrieval_ms=outcome.retrieval_ms + outcome.rerank_ms,
        )

        answer.injection_flags = injection_flags
        with stage(log, "generate", providers=len(self.providers)) as details:
            started = time.perf_counter()

            # Running out of quota is not a failure to report, it is a reason to stop
            # summarising. The reranked evidence and its citations are still correct, and
            # they were the actual need; the same path serves a total provider outage.
            budget = self.budget.state()
            if budget.exhausted:
                answer = self._degrade(answer)
                answer.error = "daily request budget exhausted"
                answer.generation_ms = round((time.perf_counter() - started) * 1000, 1)
                details.update({"mode": answer.mode, "budget_exhausted": True})
                return answer

            for index, provider in enumerate(self.providers):
                try:
                    text = provider.generate(SYSTEM_PROMPT, user_prompt)
                except Exception as error:  # any provider failure moves to the next one
                    log.warning(
                        "provider failed",
                        extra={"provider": provider.name, "error": str(error)[:200]},
                    )
                    answer.error = f"{provider.name}: {type(error).__name__}"
                    continue

                answer.text = text
                answer.provider = provider.name
                answer.model = provider.model
                answer.failover_used = index > 0
                answer.mode = "generated"
                break
            else:
                answer = self._degrade(answer)

            answer.generation_ms = round((time.perf_counter() - started) * 1000, 1)
            self._check_citations(answer)

            details.update(
                {
                    "mode": answer.mode,
                    "provider": answer.provider,
                    "failover": answer.failover_used,
                    "cited": len(answer.cited),
                    "invalid_markers": len(answer.invalid_markers),
                    "uncited_sentences": answer.uncited_sentences,
                    "evidence_injection": answer.injection_flags,
                }
            )

        return answer

    @staticmethod
    def _check_citations(answer: GeneratedAnswer) -> None:
        """Validate a generated answer's citations against the evidence it was given."""
        if answer.mode != "generated":
            return
        answer.invalid_markers, answer.uncited_sentences = validate(answer.text, answer.citations)
        if answer.invalid_markers:
            log.warning(
                "answer cited evidence that does not exist",
                extra={"markers": answer.invalid_markers},
            )
        if answer.uncited_sentences:
            log.warning(
                "answer contains uncited claims",
                extra={"sentences": answer.uncited_sentences},
            )

    def _degrade(self, answer: GeneratedAnswer) -> GeneratedAnswer:
        """Return cited evidence verbatim when no provider can be reached."""
        log.error("all providers failed; serving cited sources without generation")
        blocks = [
            f"**{citation.citation}**\n\n{result.chunk.content.strip()}"
            for citation, result in zip(answer.citations, answer.evidence, strict=True)
        ]
        answer.text = f"{_DEGRADED_NOTICE}\n\n" + "\n\n---\n\n".join(blocks)
        answer.mode = "degraded"
        answer.provider = "none"
        for citation in answer.citations:
            citation.used = True
        return answer

    def close(self) -> None:
        self.retriever.close()
