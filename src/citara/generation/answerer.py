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

from citara.config import Settings, get_settings
from citara.generation.citations import build_citations, validate
from citara.generation.models import GeneratedAnswer
from citara.generation.prompts import SYSTEM_PROMPT, build_user_prompt
from citara.generation.providers import Provider, available_providers
from citara.log import get_logger, stage
from citara.retrieval.models import RetrievalOutcome
from citara.retrieval.retriever import HybridRetriever

log = get_logger("generation")

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

    def answer(
        self,
        question: str,
        history: list[str] | None = None,
        doc_ids: list[str] | None = None,
        year: int | None = None,
    ) -> GeneratedAnswer:
        """Answer *question*, or refuse."""
        outcome = self.retriever.retrieve(question, history=history, doc_ids=doc_ids, year=year)
        return self.answer_from(question, outcome)

    def answer_from(self, question: str, outcome: RetrievalOutcome) -> GeneratedAnswer:
        """Generate from an existing retrieval outcome."""
        if outcome.refused or not outcome.results:
            log.info("refusing", extra={"reason": outcome.refusal_reason, "question": question})
            return GeneratedAnswer(
                text=self.settings.generation.refusal_message,
                mode="refused",
                retrieval_ms=outcome.retrieval_ms + outcome.rerank_ms,
                evidence=[],
            )

        citations = build_citations(outcome.results)
        user_prompt = build_user_prompt(question, outcome.results)
        answer = GeneratedAnswer(
            text="",
            citations=citations,
            evidence=outcome.results,
            retrieval_ms=outcome.retrieval_ms + outcome.rerank_ms,
        )

        with stage(log, "generate", providers=len(self.providers)) as details:
            started = time.perf_counter()
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
                }
            )

        return answer

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
