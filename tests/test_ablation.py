"""Tests for the ablation harness (feature 70)."""

from __future__ import annotations

from citara.config import Settings
from citara.evaluation.ablation import (
    CONFIGURATIONS,
    AblationRow,
    render_markdown,
)
from citara.evaluation.models import GoldQuestion, GoldSet, GoldSource


def gold_set() -> GoldSet:
    answerable = [
        GoldQuestion(
            id=f"a{i}",
            question=f"question {i}",
            category="conceptual",
            expected_answer="answer",
            sources=[GoldSource(doc_id="ndrp", page_start=1, page_end=1)],
        )
        for i in range(24)
    ]
    refusals = [
        GoldQuestion(
            id=f"u{i}", question=f"unanswerable {i}", category="unanswerable", answerable=False
        )
        for i in range(6)
    ]
    return GoldSet(questions=[*answerable, *refusals])


def row(name: str, hit5: float, mrr: float, admitted: int, p50: float = 50.0) -> AblationRow:
    return AblationRow(
        name=name,
        mode="dense",
        dense_weight=1.0,
        sparse_weight=0.0,
        rationale="test",
        hit_rate={1: 0.1, 3: 0.2, 5: hit5},
        mrr=mrr,
        by_category={"conceptual": hit5, "follow_up": 0.0},
        retrieval_p50_ms=p50,
        unanswerable_admitted=admitted,
        unanswerable_total=6,
    )


def test_every_configuration_is_expressible_as_settings() -> None:
    """Each row must be the same code path under different configuration, not a variant."""
    settings = Settings(_env_file=None)  # type: ignore[call-arg]
    for configuration in CONFIGURATIONS:
        tuned = configuration.apply(settings)
        assert tuned.retrieval.mode == configuration.mode
        assert tuned.retrieval.dense_weight == configuration.dense_weight
        assert tuned.retrieval.sparse_weight == configuration.sparse_weight


def test_configurations_cover_the_four_the_spec_names() -> None:
    names = {c.name for c in CONFIGURATIONS}
    assert {"dense", "sparse", "hybrid", "hybrid + rerank"} <= names


def test_each_configuration_has_a_distinct_fingerprint() -> None:
    """A number that cannot be traced to its configuration is not evidence."""
    settings = Settings(_env_file=None)  # type: ignore[call-arg]
    fingerprints = {c.apply(settings).fingerprint() for c in CONFIGURATIONS}
    # hybrid and hybrid+rerank differ only by mode, which is part of the fingerprint.
    assert len(fingerprints) == len(CONFIGURATIONS)


def test_table_reports_refusal_behaviour_per_configuration() -> None:
    markdown = render_markdown([row("dense", 0.458, 0.238, admitted=6)], gold_set())
    assert "Unanswerable admitted" in markdown
    assert "6/6" in markdown


def test_interpretation_names_the_gate_when_only_some_rows_refuse() -> None:
    """The headline finding must be derived from the numbers, not written by hand."""
    rows = [row("dense", 0.458, 0.238, admitted=6, p50=60), row("gated", 0.458, 0.255, 0, 3300)]
    markdown = render_markdown(rows, gold_set())
    assert "makes refusal possible" in markdown
    assert "dense" in markdown and "gated" in markdown


def test_interpretation_reports_the_latency_cost_of_the_gate() -> None:
    rows = [row("fast", 0.4, 0.2, admitted=6, p50=12), row("gated", 0.4, 0.25, 0, 3300)]
    assert "costs latency" in render_markdown(rows, gold_set())


def test_interpretation_flags_a_category_that_fails_everywhere() -> None:
    rows = [row("dense", 0.4, 0.2, admitted=6), row("gated", 0.4, 0.25, admitted=0)]
    markdown = render_markdown(rows, gold_set())
    assert "Follow-up questions fail in every configuration" in markdown


def test_sample_size_caveat_is_computed_not_asserted() -> None:
    """24 questions means one question moves Hit@k by 0.042; the file must say so."""
    markdown = render_markdown([row("dense", 0.4, 0.2, admitted=0)], gold_set())
    assert "0.042" in markdown
    assert "24 answerable questions" in markdown


def test_render_handles_an_empty_run() -> None:
    assert "Ablation study" in render_markdown([], gold_set())


def test_re_rendering_keeps_the_date_the_numbers_were_measured_on() -> None:
    """Re-rendering measures nothing, so it must not date the numbers to today."""
    markdown = render_markdown([row("dense", 0.5, 0.3, 6)], gold_set(), "2026-09-18T10:00:00+00:00")
    assert "on 2026-09-18" in markdown
