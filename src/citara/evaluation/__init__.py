"""Offline evaluation: gold set, retrieval metrics, ablation, generation quality (66-71)."""

from citara.evaluation.ablation import (
    CONFIGURATIONS,
    AblationRow,
    Configuration,
    render_markdown,
    run_configuration,
)
from citara.evaluation.generation import (
    GenerationReport,
    QuestionResult,
    evaluation_settings,
    run_generation,
)
from citara.evaluation.gold import (
    hit_rank,
    hit_rate_at_k,
    is_hit,
    load_gold_set,
    mean_reciprocal_rank,
)
from citara.evaluation.judge import (
    ClaimVerdict,
    FaithfulnessScore,
    Judge,
    JudgeUnavailable,
    RelevanceScore,
    build_judge,
    parse_json_object,
)
from citara.evaluation.metrics import mean, percentile
from citara.evaluation.models import GoldQuestion, GoldSet, GoldSource

__all__ = [
    "CONFIGURATIONS",
    "AblationRow",
    "ClaimVerdict",
    "Configuration",
    "FaithfulnessScore",
    "GenerationReport",
    "GoldQuestion",
    "GoldSet",
    "GoldSource",
    "Judge",
    "JudgeUnavailable",
    "QuestionResult",
    "RelevanceScore",
    "build_judge",
    "evaluation_settings",
    "hit_rank",
    "hit_rate_at_k",
    "is_hit",
    "load_gold_set",
    "mean",
    "mean_reciprocal_rank",
    "parse_json_object",
    "percentile",
    "render_markdown",
    "run_configuration",
    "run_generation",
]
