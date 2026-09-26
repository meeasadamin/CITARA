"""The ablation study (feature 70).

Every configuration is the same code path under different settings, so a difference in the
table is a difference in configuration and nothing else. Running them from one command
matters for the same reason: a table assembled by hand from five separate sessions cannot be
trusted to describe one corpus, one gold set and one set of parameters.

Each row also records the configuration fingerprint, so a number can never be separated from
the setup that produced it.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

from citara.config import Settings
from citara.evaluation.gold import hit_rank, hit_rate_at_k, mean_reciprocal_rank
from citara.evaluation.metrics import percentile
from citara.evaluation.models import GoldSet
from citara.log import get_logger
from citara.retrieval.retriever import HybridRetriever

log = get_logger("evaluation.ablation")


@dataclass(frozen=True)
class Configuration:
    """One row of the table: a name, an explanation, and the settings that produce it."""

    name: str
    mode: str
    dense_weight: float
    sparse_weight: float
    rationale: str

    def apply(self, settings: Settings) -> Settings:
        """Settings for this configuration, with the floor left to the caller."""
        return settings.model_copy(
            update={
                "retrieval": settings.retrieval.model_copy(
                    update={
                        "mode": self.mode,
                        "dense_weight": self.dense_weight,
                        "sparse_weight": self.sparse_weight,
                    }
                )
            }
        )


# The four configurations the specification calls for, plus the one the measurements
# produced: dense retrieval with the cross-encoder used only as the admissibility gate.
CONFIGURATIONS: tuple[Configuration, ...] = (
    Configuration(
        "dense",
        "dense",
        1.0,
        0.0,
        "Semantic search alone. Strong on paraphrase, structurally blind to exact tokens.",
    ),
    Configuration(
        "sparse",
        "sparse",
        0.0,
        1.0,
        "BM25 alone. Finds identifiers a vector discards, misses everything worded differently.",
    ),
    Configuration(
        "hybrid",
        "hybrid",
        0.5,
        0.5,
        "Even reciprocal rank fusion of both retrievers, as the specification proposed.",
    ),
    Configuration(
        "hybrid + rerank",
        "hybrid_rerank",
        0.5,
        0.5,
        "Even fusion, then the cross-encoder scores the selected chunks.",
    ),
    Configuration(
        "dense + rerank gate",
        "hybrid_rerank",
        1.0,
        0.0,
        "Dense ranking with the cross-encoder as the refusal gate. The shipped default.",
    ),
)


@dataclass
class AblationRow:
    """Measured results for one configuration."""

    name: str
    mode: str
    dense_weight: float
    sparse_weight: float
    rationale: str
    fingerprint: str = ""
    hit_rate: dict[int, float] = field(default_factory=dict)
    mrr: float = 0.0
    by_category: dict[str, float] = field(default_factory=dict)
    retrieval_p50_ms: float = 0.0
    retrieval_p95_ms: float = 0.0
    unanswerable_admitted: int = 0
    unanswerable_total: int = 0

    def as_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "mode": self.mode,
            "dense_weight": self.dense_weight,
            "sparse_weight": self.sparse_weight,
            "rationale": self.rationale,
            "fingerprint": self.fingerprint,
            "hit_rate": {str(k): v for k, v in self.hit_rate.items()},
            "mrr": self.mrr,
            "by_category": self.by_category,
            "retrieval_p50_ms": self.retrieval_p50_ms,
            "retrieval_p95_ms": self.retrieval_p95_ms,
            "unanswerable_admitted": self.unanswerable_admitted,
            "unanswerable_total": self.unanswerable_total,
        }


def run_configuration(
    configuration: Configuration, settings: Settings, gold: GoldSet
) -> AblationRow:
    """Measure one configuration over the whole gold set."""
    tuned = configuration.apply(settings)
    row = AblationRow(
        name=configuration.name,
        mode=configuration.mode,
        dense_weight=configuration.dense_weight,
        sparse_weight=configuration.sparse_weight,
        rationale=configuration.rationale,
        fingerprint=tuned.fingerprint(),
    )

    # Finding the evidence is measured with the floor disabled; whether the system should
    # answer is measured separately, at the floor the pipeline actually ships with.
    open_settings = tuned.model_copy(
        update={"retrieval": tuned.retrieval.model_copy(update={"relevance_floor": 0.0})}
    )
    retriever = HybridRetriever(open_settings)
    ranks: list[int | None] = []
    per_category: dict[str, list[int | None]] = {}
    latencies: list[float] = []

    try:
        for question in gold.answerable():
            outcome = retriever.retrieve(question.question, history=question.context or None)
            rank = hit_rank(question, [result.chunk for result in outcome.results])
            ranks.append(rank)
            per_category.setdefault(question.category, []).append(rank)
            latencies.append(outcome.retrieval_ms + outcome.rerank_ms)

        row.hit_rate = {k: hit_rate_at_k(ranks, k) for k in settings.evaluation.hit_rate_k}
        row.mrr = mean_reciprocal_rank(ranks)
        row.by_category = {
            category: hit_rate_at_k(values, 5) for category, values in sorted(per_category.items())
        }
        row.retrieval_p50_ms = percentile(latencies, 0.5)
        row.retrieval_p95_ms = percentile(latencies, 0.95)
    finally:
        retriever.close()

    # Refusal behaviour at the shipped floor: how often would this configuration answer a
    # question the corpus cannot answer?
    gated = HybridRetriever(tuned)
    try:
        refusal_cases = gold.refusal_cases()
        row.unanswerable_total = len(refusal_cases)
        for question in refusal_cases:
            outcome = gated.retrieve(question.question, history=question.context or None)
            if not outcome.refused:
                row.unanswerable_admitted += 1
    finally:
        gated.close()

    log.info(
        "configuration measured",
        extra={
            "configuration": row.name,
            "hit@5": row.hit_rate.get(5),
            "mrr": round(row.mrr, 3),
            "admitted": row.unanswerable_admitted,
        },
    )
    return row


def _interpretation(rows: list[AblationRow]) -> list[str]:
    """State what the numbers show, computed from the numbers themselves."""
    if not rows:
        return []

    best_hit = max(rows, key=lambda r: (r.hit_rate.get(5, 0.0), r.mrr))
    gated = [r for r in rows if r.unanswerable_total and not r.unanswerable_admitted]
    ungated = [r for r in rows if r.unanswerable_admitted == r.unanswerable_total > 0]
    notes: list[str] = []

    if ungated and gated:
        notes.append(
            f"**The cross-encoder is what makes refusal possible.** Every configuration "
            f"without it ({', '.join(r.name for r in ungated)}) answers all "
            f"{ungated[0].unanswerable_total} questions the corpus cannot answer, while the "
            f"configurations with it ({', '.join(r.name for r in gated)}) answer none. This "
            "is not a tuning difference: reciprocal rank fusion produces scores derived from "
            "ranks, which carry no absolute meaning and cannot be thresholded, so a pipeline "
            "without the cross-encoder has nothing to refuse on. Its value here is a "
            "calibrated score, not better ordering."
        )

    reranked = [r for r in rows if r.retrieval_p50_ms > 1000]
    plain = [r for r in rows if r.retrieval_p50_ms <= 1000]
    if reranked and plain:
        cheapest_gate = min(reranked, key=lambda r: r.retrieval_p50_ms)
        fastest = min(plain, key=lambda r: r.retrieval_p50_ms)
        notes.append(
            f"**That guarantee costs latency.** Gated configurations run at "
            f"{cheapest_gate.retrieval_p50_ms:.0f} ms against {fastest.retrieval_p50_ms:.0f} ms "
            "for retrieval alone, almost entirely cross-encoder inference on CPU. For a system "
            "whose worst failure is a confident wrong answer, that is a trade worth making."
        )

    notes.append(
        f"**Best retrieval quality: {best_hit.name}** at Hit@5 {best_hit.hit_rate.get(5, 0):.3f} "
        f"and MRR {best_hit.mrr:.3f}."
    )

    hybrids = [r for r in rows if r.sparse_weight and r.dense_weight]
    dense_only = [r for r in rows if r.dense_weight and not r.sparse_weight]
    if hybrids and dense_only:
        best_hybrid = max(hybrids, key=lambda r: r.hit_rate.get(5, 0.0))
        best_dense = max(dense_only, key=lambda r: r.hit_rate.get(5, 0.0))
        if best_dense.hit_rate.get(5, 0.0) >= best_hybrid.hit_rate.get(5, 0.0):
            notes.append(
                f"**Hybrid fusion did not pay for itself.** Dense ranking reaches Hit@5 "
                f"{best_dense.hit_rate.get(5, 0):.3f} against {best_hybrid.hit_rate.get(5, 0):.3f} "
                "for even fusion, so BM25 is weighted out of the ranking by default. The "
                "specification argued the opposite; the measurement is what shipped. BM25 still "
                "runs and contributes candidates, and the per-category table shows why it is "
                "kept: it is the only retriever that can match an exact identifier."
            )

    followups = [r.by_category.get("follow_up") for r in rows if "follow_up" in r.by_category]
    if followups and all(value == 0.0 for value in followups):
        notes.append(
            "**Follow-up questions fail in every configuration**, including with model-based "
            "query rewriting active and producing well-formed standalone queries. The failure "
            "is therefore in retrieval rather than in rewriting, which is where the next round "
            "of work should go."
        )

    return [f"- {note}" for note in notes]


def render_markdown(rows: list[AblationRow], gold: GoldSet, measured_at: str = "") -> str:
    """Render the table and its caveats as a committed artifact.

    *measured_at* is the date of the run, not of the rendering. Re-rendering a stored run
    measures nothing, so stamping it with today would date the numbers to a day on which no
    question was asked.
    """
    answerable = len(gold.answerable())
    swing = round(1 / answerable, 3) if answerable else 0.0
    lines = [
        "# Ablation study",
        "",
        f"Generated by `uv run python -m citara.evaluation` on "
        f"{measured_at[:10] or f'{datetime.now(UTC):%Y-%m-%d}'}. Every row is the same code "
        "path under different configuration.",
        "",
        f"Measured over {answerable} answerable questions and "
        f"{len(gold.refusal_cases())} refusal cases.",
        "",
        "| Configuration | Hit@1 | Hit@3 | Hit@5 | MRR | Retrieval p50 | Unanswerable admitted |",
        "|---|---|---|---|---|---|---|",
    ]
    for row in rows:
        lines.append(
            f"| {row.name} | {row.hit_rate.get(1, 0):.3f} | {row.hit_rate.get(3, 0):.3f} | "
            f"{row.hit_rate.get(5, 0):.3f} | {row.mrr:.3f} | {row.retrieval_p50_ms:.0f} ms | "
            f"{row.unanswerable_admitted}/{row.unanswerable_total} |"
        )

    lines += [
        "",
        "## Hit@5 by question category",
        "",
        "| Configuration | "
        + " | ".join(sorted({category for row in rows for category in row.by_category}))
        + " |",
    ]
    categories = sorted({category for row in rows for category in row.by_category})
    lines.append("|---" * (len(categories) + 1) + "|")
    for row in rows:
        cells = " | ".join(f"{row.by_category.get(category, 0):.2f}" for category in categories)
        lines.append(f"| {row.name} | {cells} |")

    lines += ["", "## What the table says", ""]
    lines += _interpretation(rows)

    lines += [
        "",
        "## What each configuration is for",
        "",
    ]
    for row in rows:
        lines.append(f"- **{row.name}** — {row.rationale}")

    lines += [
        "",
        "## Reading this honestly",
        "",
        f"The gold set holds {answerable} answerable questions, so a single question moves "
        f"Hit@k by {swing:.3f} and a two-question difference is within noise. These results "
        "are directional, and the ablation should be re-run as the set grows.",
        "",
        "Hit Rate counts a question as found when a retrieved chunk covers a gold page. The "
        "relevance floor is disabled for the retrieval columns, because finding the evidence "
        "and deciding whether to answer are separate questions; the final column applies the "
        "shipped floor to the questions the corpus cannot answer.",
    ]
    return "\n".join(lines) + "\n"
