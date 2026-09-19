"""Question to cited answer (features 34-41, 51).

The order of operations is the safety property. Retrieval decides admissibility first; only
if evidence clears the relevance floor is a model called at all. A refusal is therefore not
something the model chooses to say - it happens before generation exists, which is the only
way to guarantee it.

If every provider fails, the system degrades rather than dies: the reranked evidence is
returned verbatim with its citations and an explicit notice that generation is unavailable.
The officer still gets the correct source pages, which was the actual need (feature 51).
"""

from __future__ import annotations

import time
from collections.abc import Iterator
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
from citara.retrieval.models import RetrievalOutcome
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
        self.budget = RequestBudget(
            paths.resolved(paths.usage_path), resilience.daily_request_budget
        )
        self.sessions = SessionLimiter(resilience.session_query_cap)

    def answer(
        self,
        question: str,
        history: list[str] | None = None,
        doc_ids: list[str] | None = None,
        year: int | None = None,
        session_id: str = "",
    ) -> GeneratedAnswer:
        """Answer *question*, or refuse.

        Screening and scope control run before retrieval: an instruction-override attempt or
        a question about French geography should not reach the corpus, the model, or the
        provider quota.
        """
        started = time.perf_counter()
        blocked = self.preflight(question)
        if blocked is not None:
            self._record(question, blocked, started)
            return blocked

        if session_id and not self.sessions.allows(session_id):
            answer = GeneratedAnswer(text=_SESSION_CAP_MESSAGE, mode="refused")
            self._record(question, answer, started)
            return answer

        key = cache_key(
            question,
            self.settings.fingerprint(),
            docs=",".join(sorted(doc_ids or [])),
            year=year or "",
            history=history[-1] if history else "",
        )
        # A cache hit is checked after the session cap but recorded against neither budget
        # nor session: it spends no provider quota, which is what both limits protect.
        if self.settings.resilience.enable_cache:
            cached = self.cache.get(key)
            if cached:
                answer = self._from_cache(cached)
                self._record(question, answer, started)
                return answer

        outcome = self.retriever.retrieve(question, history=history, doc_ids=doc_ids, year=year)
        answer = self.answer_from(question, outcome)

        if session_id:
            self.sessions.record(session_id)
        if answer.mode == "generated":
            self.budget.record()
            if self.settings.resilience.enable_cache:
                self.cache.set(key, self._to_cache(answer))

        self._record(question, answer, started, outcome)
        return answer

    @staticmethod
    def _to_cache(answer: GeneratedAnswer) -> dict[str, Any]:
        """The parts of an answer worth storing: the text and how to verify it."""
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
        }

    @staticmethod
    def _from_cache(payload: dict[str, Any]) -> GeneratedAnswer:
        """Rebuild an answer from the cache, marked so the interface can say it is cached."""
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

    def answer_from(self, question: str, outcome: RetrievalOutcome) -> GeneratedAnswer:
        """Generate from an existing retrieval outcome.

        Screening repeats here even though the entry points already ran it: this method is
        public, and the check costs a few regular expressions against the cost of sending an
        injection attempt to a model.
        """
        blocked = self.preflight(question)
        if blocked is not None:
            return blocked

        if outcome.refused or not outcome.results:
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
            if answer.mode == "generated":
                answer.invalid_markers, answer.uncited_sentences = validate(
                    answer.text, answer.citations
                )
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

    def stream(
        self, question: str, history: list[str] | None = None
    ) -> Iterator[tuple[str, GeneratedAnswer | None]]:
        """Yield answer text as it arrives, then the finished answer.

        Each item is ``(text_piece, None)`` while generating and ``("", answer)`` at the end,
        so a caller can render tokens as they arrive and still receive validated citations.
        Three seconds of blank screen reads as a crash in a live demo, which is the whole
        reason this path exists (feature 41).

        Refusals and degraded answers are not streamed: there is nothing being generated, so
        the complete text is delivered in one piece.
        """
        started = time.perf_counter()
        blocked = self.preflight(question)
        if blocked is not None:
            self._record(question, blocked, started)
            yield blocked.text, blocked
            return

        outcome = self.retriever.retrieve(question, history=history)
        if outcome.refused or not outcome.results:
            answer = self.answer_from(question, outcome)
            self._record(question, answer, started, outcome)
            yield answer.text, answer
            return

        citations = build_citations(outcome.results)
        user_prompt, injection_flags = build_user_prompt(question, outcome.results)
        answer = GeneratedAnswer(
            text="",
            citations=citations,
            evidence=outcome.results,
            retrieval_ms=outcome.retrieval_ms + outcome.rerank_ms,
            injection_flags=injection_flags,
        )

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
        if answer.mode == "generated":
            answer.invalid_markers, answer.uncited_sentences = validate(
                answer.text, answer.citations
            )
        self._record(question, answer, started, outcome)
        yield "", answer

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
