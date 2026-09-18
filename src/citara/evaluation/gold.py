"""Loading the gold set and scoring retrieval against it (features 66, 67).

Hit Rate and MRR are defined here, against page ranges rather than chunk ids, so the metrics
survive a change to chunking parameters.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

from citara.chunking.models import Chunk
from citara.config import Settings, get_settings
from citara.evaluation.models import GoldQuestion, GoldSet


def load_gold_set(path: Path | None = None, settings: Settings | None = None) -> GoldSet:
    """Read the curated question set."""
    settings = settings or get_settings()
    path = path or settings.paths.resolved(settings.evaluation.gold_set_path)
    return GoldSet.model_validate_json(path.read_text(encoding="utf-8"))


def is_hit(question: GoldQuestion, doc_id: str, page_start: int, page_end: int) -> bool:
    """True when a retrieved location overlaps any gold source for *question*."""
    return any(source.covers(doc_id, page_start, page_end) for source in question.sources)


def hit_rank(question: GoldQuestion, retrieved: Sequence[Chunk]) -> int | None:
    """1-based rank of the first retrieved chunk covering a gold source, or None."""
    for rank, chunk in enumerate(retrieved, start=1):
        if is_hit(question, chunk.doc_id, chunk.page_start, chunk.page_end):
            return rank
    return None


def hit_rate_at_k(ranks: Sequence[int | None], k: int) -> float:
    """Share of questions whose gold page appeared within the top k."""
    if not ranks:
        return 0.0
    return sum(1 for rank in ranks if rank is not None and rank <= k) / len(ranks)


def mean_reciprocal_rank(ranks: Sequence[int | None]) -> float:
    """Mean of 1/rank, treating a miss as zero.

    Rewards putting the right evidence at the top rather than merely somewhere in the list,
    which is what matters when only the top few chunks reach the model.
    """
    if not ranks:
        return 0.0
    return sum(1.0 / rank if rank else 0.0 for rank in ranks) / len(ranks)
