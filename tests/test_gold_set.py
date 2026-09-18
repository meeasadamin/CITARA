"""Tests for the gold question set and retrieval metrics (features 66, 67)."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from citara.chunking.models import Chunk
from citara.evaluation.gold import (
    hit_rank,
    hit_rate_at_k,
    is_hit,
    load_gold_set,
    mean_reciprocal_rank,
)
from citara.evaluation.models import GoldQuestion, GoldSet, GoldSource


def chunk_at(doc_id: str, page_start: int, page_end: int | None = None) -> Chunk:
    return Chunk(
        chunk_id=f"{doc_id}-{page_start}",
        doc_id=doc_id,
        title=doc_id,
        filename=f"{doc_id}.pdf",
        page_start=page_start,
        page_end=page_end or page_start,
        total_pages=200,
        ordinal=0,
        content="text",
    )


def question_at(doc_id: str, page_start: int, page_end: int) -> GoldQuestion:
    return GoldQuestion(
        id="q1",
        question="test",
        category="conceptual",
        expected_answer="something",
        sources=[GoldSource(doc_id=doc_id, page_start=page_start, page_end=page_end)],
    )


# --- schema rules ------------------------------------------------------------------


def test_answerable_question_requires_a_source() -> None:
    with pytest.raises(ValidationError, match="needs at least one source"):
        GoldQuestion(id="x", question="q", category="conceptual", expected_answer="a")


def test_unanswerable_question_must_not_cite_sources() -> None:
    """A refusal case with a citation is a contradiction: the answer would be reachable."""
    with pytest.raises(ValidationError, match="must not cite sources"):
        GoldQuestion(
            id="x",
            question="q",
            category="unanswerable",
            answerable=False,
            sources=[GoldSource(doc_id="d", page_start=1, page_end=1)],
        )


def test_unanswerable_question_expects_a_refusal_not_an_answer() -> None:
    with pytest.raises(ValidationError, match="expects a refusal"):
        GoldQuestion(
            id="x",
            question="q",
            category="unanswerable",
            answerable=False,
            expected_answer="something",
        )


def test_follow_up_requires_its_preceding_turn() -> None:
    """'What about Sindh?' is meaningless without the turn before it."""
    with pytest.raises(ValidationError, match="needs its preceding turn"):
        GoldQuestion(
            id="x",
            question="What about Sindh?",
            category="follow_up",
            expected_answer="a",
            sources=[GoldSource(doc_id="d", page_start=1, page_end=1)],
        )


def test_page_range_must_be_ordered() -> None:
    with pytest.raises(ValidationError, match="must not precede"):
        GoldSource(doc_id="d", page_start=10, page_end=2)


def test_duplicate_ids_rejected() -> None:
    item = GoldQuestion(
        id="same",
        question="q",
        category="conceptual",
        expected_answer="a",
        sources=[GoldSource(doc_id="d", page_start=1, page_end=1)],
    )
    with pytest.raises(ValidationError, match="unique"):
        GoldSet(questions=[item, item])


# --- hit detection -----------------------------------------------------------------


def test_hit_requires_the_same_document() -> None:
    question = question_at("ndrp", 47, 47)
    assert is_hit(question, "ndrp", 47, 47) is True
    assert is_hit(question, "pdna", 47, 47) is False


def test_overlapping_page_ranges_count_as_hits() -> None:
    """A chunk citing pp. 46-48 does contain page 47, so it is a hit."""
    question = question_at("ndrp", 47, 47)
    assert is_hit(question, "ndrp", 46, 48) is True
    assert is_hit(question, "ndrp", 48, 50) is False
    assert is_hit(question, "ndrp", 44, 46) is False


def test_hit_rank_finds_the_first_match() -> None:
    question = question_at("ndrp", 47, 47)
    retrieved = [chunk_at("pdna", 1), chunk_at("ndrp", 12), chunk_at("ndrp", 47)]
    assert hit_rank(question, retrieved) == 3
    assert hit_rank(question, [chunk_at("pdna", 1)]) is None


# --- metrics -----------------------------------------------------------------------


def test_hit_rate_at_k() -> None:
    ranks = [1, 3, None, 5]
    assert hit_rate_at_k(ranks, 1) == 0.25
    assert hit_rate_at_k(ranks, 3) == 0.5
    assert hit_rate_at_k(ranks, 5) == 0.75
    assert hit_rate_at_k([], 5) == 0.0


def test_mrr_rewards_rank_one() -> None:
    assert mean_reciprocal_rank([1, 1]) == 1.0
    assert mean_reciprocal_rank([2]) == 0.5
    assert mean_reciprocal_rank([None]) == 0.0
    assert mean_reciprocal_rank([1, None]) == 0.5
    assert mean_reciprocal_rank([]) == 0.0


# --- the real set ------------------------------------------------------------------


def test_real_gold_set_loads_and_is_balanced() -> None:
    """The category mix is the experiment design, so it is asserted, not assumed."""
    gold = load_gold_set()
    assert len(gold.questions) >= 25  # feature 66 calls for 25-30

    counts = gold.category_counts()
    # Without identifier and table questions the ablation cannot show what hybrid retrieval
    # and table extraction are for.
    assert counts.get("identifier", 0) >= 4
    assert counts.get("table", 0) >= 4
    assert counts.get("conceptual", 0) >= 4
    assert counts.get("follow_up", 0) >= 2
    assert len(gold.refusal_cases()) >= 4


def test_real_gold_set_sources_are_specific() -> None:
    """A gold range spanning a whole document makes Hit Rate meaningless."""
    for question in load_gold_set().answerable():
        for source in question.sources:
            span = source.page_end - source.page_start
            assert span <= 10, f"{question.id} cites {span + 1} pages"


def test_real_gold_set_quotes_present_for_verification() -> None:
    """Every answerable item carries the text a human checks against the PDF."""
    for question in load_gold_set().answerable():
        assert question.expected_answer, f"{question.id} has no expected answer"
        assert any(s.quote for s in question.sources), f"{question.id} has no supporting quote"
