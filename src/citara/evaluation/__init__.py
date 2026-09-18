"""Offline evaluation: gold set, retrieval metrics, ablation (features 66-71)."""

from citara.evaluation.gold import (
    hit_rank,
    hit_rate_at_k,
    is_hit,
    load_gold_set,
    mean_reciprocal_rank,
)
from citara.evaluation.models import GoldQuestion, GoldSet, GoldSource

__all__ = [
    "GoldQuestion",
    "GoldSet",
    "GoldSource",
    "hit_rank",
    "hit_rate_at_k",
    "is_hit",
    "load_gold_set",
    "mean_reciprocal_rank",
]
