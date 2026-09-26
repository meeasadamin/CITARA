"""End-to-end generation quality and latency (features 68, 69, 71).

One pass over the gold set produces all three numbers, because they are properties of the
same answers and measuring them separately would mean three different runs of a
non-deterministic system, reported as though they described one.

Three decisions shape what the numbers mean.

**The cache is switched off.** The assistant caches answers so a repeated question costs
nothing, which is right in the interface and fatal here: a cached answer takes 0.4 s and
spends no model call, so a run with the cache on measures the cache.

**Refusals are counted, not scored.** A refusal makes no claims, so faithfulness is
undefined for it. It is still reported: a refusal to a question the corpus *can* answer is
one of the two failures this system has, and it is visible in the table as an answerable
question that was not answered.

**Latency is split the way the specification asks.** Retrieval is measured over every
question, because every question retrieves. Generation is measured only over the answers
that were generated, because a question refused at the gate never reaches a model and
folding its zero into the percentile would report a speed the system does not have.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any, Protocol

from citara.config import Settings
from citara.evaluation.judge import Judge
from citara.evaluation.metrics import mean, percentile
from citara.evaluation.models import GoldQuestion, GoldSet
from citara.generation.models import GeneratedAnswer
from citara.log import get_logger

log = get_logger("evaluation.generation")

# Every request in this system is capped: 30 s for an answer, 90 s for a judge. A measurement
# above this ceiling therefore did not measure the system. It measured a laptop that suspended
# mid-run, and the first run of this suite produced two of them - one "retrieval" of 4.25 hours
# that matched a gap of exactly that length in the log. Such a question is named in the artifact
# and left out of the percentiles rather than averaged, because a benchmark that quietly
# swallows a sleeping machine is worse than one that says a question was not measured.
IMPLAUSIBLE_MS = 120_000.0


class AnswerSource(Protocol):
    """What the run loop needs from the assistant, so a test can stand in for it."""

    def answer(
        self,
        question: str,
        history: list[str] | None = ...,
        doc_ids: list[str] | None = ...,
        year: int | None = ...,
        session_id: str = ...,
    ) -> GeneratedAnswer: ...


@dataclass
class QuestionResult:
    """What one gold question produced, and what the judge made of it."""

    id: str
    question: str
    category: str
    answerable: bool
    mode: str = ""
    provider: str = ""
    model: str = ""
    faithfulness: float | None = None
    relevance: float | None = None
    claims_checked: int = 0
    unsupported: tuple[str, ...] = ()
    retrieval_ms: float = 0.0
    generation_ms: float = 0.0
    cited_sources: int = 0
    fully_cited: bool = False
    error: str = ""
    # Kept so a surprising score can be read back against the answer that earned it, and so
    # the same answers can be put to a second judge without generating them again.
    answer_text: str = ""
    evidence_citations: tuple[str, ...] = ()

    @property
    def answered(self) -> bool:
        return self.mode == "generated"

    @property
    def total_ms(self) -> float:
        return round(self.retrieval_ms + self.generation_ms, 1)

    @property
    def timed(self) -> bool:
        """False when the clock, not the system, produced this number."""
        return max(self.retrieval_ms, self.generation_ms) < IMPLAUSIBLE_MS

    def as_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "question": self.question,
            "category": self.category,
            "answerable": self.answerable,
            "mode": self.mode,
            "provider": self.provider,
            "model": self.model,
            "faithfulness": self.faithfulness,
            "relevance": self.relevance,
            "claims_checked": self.claims_checked,
            "unsupported": list(self.unsupported),
            "retrieval_ms": self.retrieval_ms,
            "generation_ms": self.generation_ms,
            "cited_sources": self.cited_sources,
            "fully_cited": self.fully_cited,
            "error": self.error,
            "answer_text": self.answer_text,
            "evidence_citations": list(self.evidence_citations),
        }


@dataclass
class GenerationReport:
    """Every result from one run, plus the setup that produced it."""

    judge: str = ""
    judge_model: str = ""
    generator_model: str = ""
    fingerprint: str = ""
    generated_at: str = ""
    results: list[QuestionResult] = field(default_factory=list)

    # -- the questions, grouped by what happened to them ---------------------------------

    @property
    def scored(self) -> list[QuestionResult]:
        """Every answer that was generated, whether or not both judgements landed.

        A judge that failed on relevance has not invalidated the faithfulness verdict it
        already returned, so the two are averaged over whatever each of them has rather than
        over the intersection. ``mean`` skips what is missing; the artifact says how many
        each row covers.
        """
        return [r for r in self.results if r.answered]

    @property
    def answerable(self) -> list[QuestionResult]:
        return [r for r in self.results if r.answerable]

    @property
    def refusal_cases(self) -> list[QuestionResult]:
        return [r for r in self.results if not r.answerable]

    @property
    def wrongly_refused(self) -> list[QuestionResult]:
        """Answerable questions the system declined: the corpus had it, the pipeline missed."""
        return [r for r in self.answerable if not r.answered and not r.error]

    @property
    def wrongly_answered(self) -> list[QuestionResult]:
        """Questions the corpus cannot answer, answered anyway: the failure that matters most."""
        return [r for r in self.refusal_cases if r.answered]

    # -- the numbers ---------------------------------------------------------------------

    @property
    def faithfulness(self) -> float | None:
        return mean([r.faithfulness for r in self.scored])

    @property
    def relevance(self) -> float | None:
        return mean([r.relevance for r in self.scored])

    @property
    def fully_cited_rate(self) -> float | None:
        scored = self.scored
        if not scored:
            return None
        return sum(1 for r in scored if r.fully_cited) / len(scored)

    @property
    def untimed(self) -> list[QuestionResult]:
        """Questions whose clock reading cannot be true, so no percentile may use them."""
        return [r for r in self.results if r.mode and not r.timed]

    def latency(self, fraction: float) -> dict[str, float]:
        """Retrieval, generation and end-to-end at one percentile (feature 71)."""
        every = [r for r in self.results if not r.error and r.timed]
        generated = [r for r in every if r.answered]
        return {
            "retrieval_ms": percentile([r.retrieval_ms for r in every], fraction),
            "generation_ms": percentile([r.generation_ms for r in generated], fraction),
            "end_to_end_ms": percentile([r.total_ms for r in generated], fraction),
        }

    def as_dict(self) -> dict[str, Any]:
        return {
            "generated_at": self.generated_at,
            "judge": self.judge,
            "judge_model": self.judge_model,
            "generator_model": self.generator_model,
            "fingerprint": self.fingerprint,
            "faithfulness": self.faithfulness,
            "relevance": self.relevance,
            "fully_cited_rate": self.fully_cited_rate,
            "answered": len(self.scored),
            "wrongly_refused": [r.id for r in self.wrongly_refused],
            "wrongly_answered": [r.id for r in self.wrongly_answered],
            "latency_p50": self.latency(0.5),
            "latency_p95": self.latency(0.95),
            "untimed": [r.id for r in self.untimed],
            "results": [r.as_dict() for r in self.results],
        }


def evaluation_settings(settings: Settings) -> Settings:
    """Settings for a measured run: no cache, and no session cap in the way.

    The cap protects a public URL from one visitor asking forty questions. A scoring run is
    exactly that, legitimately, so it is lifted rather than worked around with fake sessions.
    """
    return settings.model_copy(
        update={
            "resilience": settings.resilience.model_copy(
                update={
                    "enable_cache": False,
                    "session_query_cap": max(settings.resilience.session_query_cap, 1000),
                }
            )
        }
    )


def _evidence(answer: GeneratedAnswer) -> list[tuple[str, str]]:
    """The passages the model was given, numbered as its markers number them."""
    return [
        (
            f"{result.chunk.title}, pp. {result.chunk.page_start}-{result.chunk.page_end}",
            result.chunk.content,
        )
        for result in answer.evidence
    ]


def score_answer(
    judge: Judge, question: GoldQuestion, answer: GeneratedAnswer, result: QuestionResult
) -> None:
    """Fill *result* with the judge's two verdicts, leaving it usable if either fails."""
    # A judge that cannot be reached, or replies with something unreadable, is one lost score
    # rather than a lost run: the questions before it have already been paid for. JudgeUnavailable
    # covers an unreadable reply; the provider raises its own errors for a timeout or a quota.
    try:
        faithfulness = judge.faithfulness(answer.text, _evidence(answer))
        result.faithfulness = faithfulness.score
        result.claims_checked = len(faithfulness.claims)
        result.unsupported = faithfulness.unsupported
    except Exception as error:
        result.error = f"faithfulness: {error}"
        log.warning(
            "faithfulness scoring failed",
            extra={"question": question.id, "error": str(error)[:200]},
        )
    try:
        result.relevance = judge.relevance(question.question, answer.text).score
    except Exception as error:
        result.error = f"{result.error}; relevance: {error}".lstrip("; ")
        log.warning(
            "relevance scoring failed", extra={"question": question.id, "error": str(error)[:200]}
        )


def run_generation(
    answerer: AnswerSource,
    judge: Judge,
    gold: GoldSet,
    settings: Settings,
    limit: int | None = None,
    on_progress: Any = None,
) -> GenerationReport:
    """Answer every gold question once and score what comes back."""
    questions = gold.questions[:limit] if limit else gold.questions
    report = GenerationReport(
        judge=judge.name,
        judge_model=judge.model,
        generator_model=settings.generation.primary_model,
        fingerprint=settings.fingerprint(),
        generated_at=datetime.now(UTC).isoformat(),
    )

    for index, question in enumerate(questions, start=1):
        if on_progress is not None:
            on_progress(index, len(questions), question)
        result = QuestionResult(
            id=question.id,
            question=question.question,
            category=question.category,
            answerable=question.answerable,
        )
        try:
            answer = answerer.answer(
                question.question,
                history=question.context or None,
                session_id="evaluation",
            )
        # One bad question must not end a run that has already spent quota on the rest.
        except Exception as error:
            result.error = f"answer: {error}"
            log.warning("question failed", extra={"question": question.id, "error": str(error)})
            report.results.append(result)
            continue

        result.mode = answer.mode
        result.provider = answer.provider
        result.model = answer.model
        result.retrieval_ms = answer.retrieval_ms
        result.generation_ms = answer.generation_ms
        result.cited_sources = len(answer.cited)
        result.fully_cited = answer.is_fully_cited
        result.answer_text = answer.text
        result.evidence_citations = tuple(citation for citation, _ in _evidence(answer))

        if answer.mode == "generated":
            score_answer(judge, question, answer, result)
        report.results.append(result)
        log.info(
            "question scored",
            extra={
                "question": question.id,
                "mode": result.mode,
                "faithfulness": result.faithfulness,
                "relevance": result.relevance,
            },
        )

    return report


# -- the artifact ---------------------------------------------------------------------------


def _number(value: float | None, places: int = 3) -> str:
    return "n/a" if value is None else f"{value:.{places}f}"


def _summary_table(report: GenerationReport) -> list[str]:
    p50, p95 = report.latency(0.5), report.latency(0.95)
    timed = len([r for r in report.results if r.mode and r.timed])
    scored = report.scored
    faithful_n = sum(1 for r in scored if r.faithfulness is not None)
    relevant_n = sum(1 for r in scored if r.relevance is not None)
    return [
        "| Measure | Value | Over |",
        "|---|---|---|",
        f"| Faithfulness | {_number(report.faithfulness)} | {faithful_n} answers |",
        f"| Answer relevance | {_number(report.relevance)} | {relevant_n} answers |",
        f"| Fully cited | {_number(report.fully_cited_rate)} | {len(scored)} answers |",
        f"| Retrieval p50 / p95 | {p50['retrieval_ms']:.0f} ms / {p95['retrieval_ms']:.0f} ms | "
        f"{timed} timed questions |",
        f"| Generation p50 / p95 | {p50['generation_ms']:.0f} ms / "
        f"{p95['generation_ms']:.0f} ms | generated answers |",
        f"| End-to-end p50 / p95 | {p50['end_to_end_ms']:.0f} ms / "
        f"{p95['end_to_end_ms']:.0f} ms | generated answers |",
    ]


def _category_table(report: GenerationReport) -> list[str]:
    categories: dict[str, list[QuestionResult]] = {}
    for result in report.scored:
        categories.setdefault(result.category, []).append(result)
    if not categories:
        return []
    lines = [
        "## By question category",
        "",
        "| Category | Answers | Faithfulness | Relevance |",
        "|---|---|---|---|",
    ]
    for category, results in sorted(categories.items()):
        lines.append(
            f"| {category} | {len(results)} | "
            f"{_number(mean([r.faithfulness for r in results]), 2)} | "
            f"{_number(mean([r.relevance for r in results]), 2)} |"
        )
    return lines


def _interpretation(report: GenerationReport) -> list[str]:
    """State what the numbers show, computed from the numbers themselves."""
    notes: list[str] = []
    faithfulness = report.faithfulness
    scored = report.scored

    if faithfulness is not None and scored:
        judged = [r for r in scored if r.faithfulness is not None]
        clean = [r for r in judged if r.faithfulness == 1.0]
        notes.append(
            f"**{len(clean)} of {len(judged)} answers were entirely supported by their "
            f"evidence**, for a mean faithfulness of {faithfulness:.3f} over "
            f"{sum(r.claims_checked for r in scored)} claims checked one at a time. The judge "
            "was shown the retrieved passages and nothing else, and told that a claim which is "
            "true but absent from them is unsupported, so this measures grounding rather than "
            "correctness."
        )

    if report.wrongly_answered:
        notes.append(
            f"**{len(report.wrongly_answered)} question(s) the corpus cannot answer were "
            f"answered anyway** ({', '.join(r.id for r in report.wrongly_answered)}). This is "
            "the failure the relevance floor exists to prevent and the one that matters most."
        )
    elif report.refusal_cases:
        notes.append(
            f"**Every one of the {len(report.refusal_cases)} questions the corpus cannot "
            "answer was refused.** The gate held for the whole set."
        )

    if report.wrongly_refused:
        notes.append(
            f"**{len(report.wrongly_refused)} answerable question(s) were refused** "
            f"({', '.join(r.id for r in report.wrongly_refused)}): the evidence is in the "
            "corpus and retrieval did not put it above the floor. This is the cost of the "
            "gate, paid in silence rather than in wrong answers."
        )

    p50, p95 = report.latency(0.5), report.latency(0.95)
    if p50["end_to_end_ms"]:
        notes.append(
            f"**A question takes {p50['end_to_end_ms'] / 1000:.1f} s at the median and "
            f"{p95['end_to_end_ms'] / 1000:.1f} s at p95**, of which "
            f"{p50['retrieval_ms'] / 1000:.1f} s is retrieval - almost entirely cross-encoder "
            "inference on CPU. Measured with the answer cache switched off; in the running "
            "interface a repeated question is served from cache in well under a second."
        )

    return notes


def render_markdown(report: GenerationReport, gold: GoldSet) -> str:
    """The generated artifact. Do not hand-edit: regenerate it."""
    errors = [r for r in report.results if r.error]
    lines = [
        "# Generation quality and latency",
        "",
        "Generated by `uv run python -m citara.evaluation generation` from the run record in",
        "[`runs/generation.json`](runs/generation.json). Do not hand-edit it.",
        "",
        f"- Gold set: {len(gold.answerable())} answerable questions, "
        f"{len(gold.refusal_cases())} the corpus cannot answer.",
        f"- Answers generated by `{report.generator_model}`.",
        f"- Scored by `{report.judge_model}` on {report.judge}.",
        f"- Configuration fingerprint: `{report.fingerprint}`.",
        f"- Run: {report.generated_at}.",
        "",
        *_summary_table(report),
        "",
    ]
    lines += _category_table(report)
    lines += ["", "## What the numbers show", ""]
    lines += [f"- {note}" for note in _interpretation(report)]

    unsupported = [(r.id, claim) for r in report.scored for claim in r.unsupported]
    if unsupported:
        lines += [
            "",
            "## Claims the evidence did not support",
            "",
            "Every unsupported claim the judge found, so a surprising score can be read back",
            "and checked against the passage rather than taken on trust.",
            "",
        ]
        lines += [f"- **{question_id}** - {claim}" for question_id, claim in unsupported]

    self_scored = report.generator_model.startswith(report.judge)
    lines += ["", "## Honest limits", ""]
    if self_scored:
        lines += [
            "On this run the judge was the same model family that wrote the answers. A model",
            "scoring its own output is the weakest form of this measurement: it shares the blind",
            "spots it is being asked to find, and these numbers should be read as the generous",
            "end of the range. Re-run with `--judge groq` for an independent reading.",
            "",
        ]
    else:
        listed = (
            "every unsupported claim is listed below rather than only counted"
            if any(r.unsupported for r in report.scored)
            else "the run record keeps every answer it judged, and any verdict can be read back"
        )
        lines += [
            f"The answers were written by `{report.generator_model}` and judged by",
            f"`{report.judge_model}` - a different company's model, so the judge is not scoring",
            "its own output. Two judges still do not always agree, and a disagreement is a fact",
            f"about the judges as much as about the answer, so {listed}.",
            "",
        ]
    lines += [
        "Faithfulness says nothing about whether an answer is correct - only whether it stayed",
        "inside its evidence. An answer faithfully drawn from a passage that does not actually",
        "answer the question scores 1.0 here and is still wrong; that failure shows up in answer",
        "relevance and in the retrieval table, not in this column.",
    ]

    if report.untimed:
        lines += [
            "",
            "## Questions the clock did not measure",
            "",
            "These answered normally and their scores above are sound; only their timings are "
            "not, because the machine suspended mid-run and the elapsed-time counter carried "
            "the sleep. They are excluded from every percentile.",
            "",
            *[
                f"- **{r.id}** - recorded {r.retrieval_ms / 1000:.0f} s retrieval, "
                f"{r.generation_ms / 1000:.0f} s generation"
                for r in report.untimed
            ],
        ]

    if errors:
        lines += [
            "",
            "## Judgements that did not land",
            "",
            "The answer itself is above and counts towards every measure it still has a score",
            "for; only the judgement named here is missing, and that measure is averaged over",
            "one question fewer rather than over a guess.",
            "",
            *[f"- **{r.id}** - {r.error}" for r in errors],
        ]

    return "\n".join(lines) + "\n"
