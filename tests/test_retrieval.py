"""Tests for hybrid retrieval, fusion, query preparation and the refusal gate (24-33)."""

from __future__ import annotations

import numpy as np
import pytest

from citara.chunking.models import Chunk
from citara.config import RetrievalSettings, Settings
from citara.indexing.sparse_index import SparseHit, SparseIndex
from citara.indexing.vector_store import DenseHit
from citara.retrieval.fusion import reciprocal_rank_fusion
from citara.retrieval.models import RetrievalOutcome, RetrievedChunk
from citara.retrieval.query import (
    HeuristicRewriter,
    expand_acronyms,
    looks_like_follow_up,
    prepare,
)
from citara.retrieval.retriever import HybridRetriever


def make_chunk(chunk_id: str, content: str = "text", doc: str = "ndrp", year: int = 2019) -> Chunk:
    return Chunk(
        chunk_id=chunk_id,
        doc_id=doc,
        title=doc.upper(),
        filename=f"{doc}.pdf",
        page_start=1,
        page_end=1,
        total_pages=10,
        year=year,
        ordinal=0,
        content=content,
    )


# --- fusion ------------------------------------------------------------------------


def test_fusion_rewards_agreement_between_retrievers() -> None:
    """A chunk both retrievers rank highly should beat one only a single retriever liked."""
    chunks = {c.chunk_id: c for c in [make_chunk("a"), make_chunk("b"), make_chunk("c")]}
    dense = [DenseHit("a", 0.9, "text", {}), DenseHit("b", 0.8, "text", {})]
    sparse = [SparseHit("c", 12.0), SparseHit("a", 9.0)]

    fused = reciprocal_rank_fusion(dense, sparse, chunks, k=60)
    assert fused[0].chunk_id == "a"
    assert fused[0].found_by == "both"


def test_fusion_records_where_each_result_came_from() -> None:
    chunks = {c.chunk_id: c for c in [make_chunk("a"), make_chunk("b")]}
    fused = reciprocal_rank_fusion([DenseHit("a", 0.9, "t", {})], [SparseHit("b", 5.0)], chunks)
    by_id = {r.chunk_id: r for r in fused}
    assert by_id["a"].found_by == "dense"
    assert by_id["b"].found_by == "sparse"


def test_fusion_weights_can_disable_a_retriever() -> None:
    """The ablation's dense-only and sparse-only rows run through this same code path."""
    chunks = {c.chunk_id: c for c in [make_chunk("a"), make_chunk("b")]}
    fused = reciprocal_rank_fusion(
        [DenseHit("a", 0.9, "t", {})],
        [SparseHit("b", 99.0)],
        chunks,
        sparse_weight=0.0,
    )
    assert fused[0].chunk_id == "a"


def test_fusion_ignores_unknown_chunk_ids() -> None:
    """A stale id in one index must not crash retrieval."""
    fused = reciprocal_rank_fusion([DenseHit("ghost", 0.9, "t", {})], [], {})
    assert fused == []


def test_fusion_is_deterministic_on_ties() -> None:
    chunks = {c.chunk_id: c for c in [make_chunk("a"), make_chunk("b")]}
    dense = [DenseHit("a", 0.5, "t", {}), DenseHit("b", 0.5, "t", {})]
    first = [r.chunk_id for r in reciprocal_rank_fusion(dense, [], chunks)]
    second = [r.chunk_id for r in reciprocal_rank_fusion(dense, [], chunks)]
    assert first == second


# --- query preparation -------------------------------------------------------------


def test_acronyms_expand_both_directions() -> None:
    assert "Glacial Lake Outburst Flood" in expand_acronyms("What is a GLOF?")
    assert "NEOC" in expand_acronyms("role of the National Emergency Operations Centre")


def test_expansion_leaves_unrelated_queries_alone() -> None:
    assert expand_acronyms("how are relief camps organised") == "how are relief camps organised"


def test_follow_up_detection() -> None:
    assert looks_like_follow_up("What about Sindh?") is True
    assert looks_like_follow_up("And the monsoon months?") is True
    assert (
        looks_like_follow_up("What is the national evacuation policy for riverine floods?") is False
    )


def test_history_rewriting_makes_a_follow_up_standalone() -> None:
    """'What about Sindh?' retrieves nothing without the turn before it (feature 30)."""
    settings = RetrievalSettings()
    dense, _ = prepare(
        "What about Sindh?",
        ["What were the total recovery needs after the 2022 floods?"],
        settings,
    )
    assert "sindh" in dense.lower()
    assert "recovery" in dense.lower() or "needs" in dense.lower()


def test_standalone_questions_are_left_untouched() -> None:
    settings = RetrievalSettings()
    question = "What is the monsoon contingency plan for Balochistan?"
    dense, _ = prepare(question, ["Earlier unrelated question about earthquakes?"], settings)
    assert dense == question


def test_history_window_is_capped() -> None:
    """An unbounded window lets a flood conversation contaminate an earthquake query (31)."""
    settings = RetrievalSettings(history_turns=1)
    dense, _ = prepare(
        "What about Sindh?", ["First about earthquakes?", "Then about floods?"], settings
    )
    assert "flood" in dense.lower()
    assert "earthquake" not in dense.lower()


def test_rewriting_disabled_with_zero_turns() -> None:
    settings = RetrievalSettings(history_turns=0)
    dense, _ = prepare("What about Sindh?", ["Anything at all?"], settings)
    assert dense == "What about Sindh?"


def test_heuristic_rewriter_survives_empty_history() -> None:
    assert HeuristicRewriter().rewrite("What about Sindh?", []) == "What about Sindh?"


# --- the funnel and the refusal gate ------------------------------------------------


class FakeEmbedder:
    dimension = 8

    def embed_documents(self, texts: list[str]) -> np.ndarray:
        return np.zeros((len(texts), self.dimension), dtype=np.float32)

    def embed_query(self, text: str) -> np.ndarray:
        return np.zeros(self.dimension, dtype=np.float32)


class FakeStore:
    def __init__(self, hits: list[DenseHit]) -> None:
        self.hits = hits
        self.last_where: dict | None = None

    def search(self, query_vector, k=12, where=None):  # type: ignore[no-untyped-def]
        self.last_where = where
        return self.hits[:k]

    def close(self) -> None: ...


class FakeReranker:
    """Scores by a lookup table, so thresholds can be asserted exactly."""

    def __init__(self, scores: dict[str, float]) -> None:
        self.scores = scores

    def rerank(self, query: str, candidates: list[RetrievedChunk]) -> list[RetrievedChunk]:
        for candidate in candidates:
            candidate.rerank_score = self.scores.get(candidate.chunk_id, 0.0)
        return sorted(candidates, key=lambda c: -(c.rerank_score or 0.0))


def build_retriever(
    chunks: list[Chunk],
    dense: list[DenseHit],
    scores: dict[str, float] | None = None,
    **retrieval: object,
) -> HybridRetriever:
    settings = Settings(
        _env_file=None,  # type: ignore[call-arg]
        retrieval=RetrievalSettings(**retrieval),  # type: ignore[arg-type]
    )
    return HybridRetriever(
        settings=settings,
        embedder=FakeEmbedder(),  # type: ignore[arg-type]
        store=FakeStore(dense),  # type: ignore[arg-type]
        sparse=SparseIndex.build(chunks),
        chunks=chunks,
        reranker=FakeReranker(scores or {}),  # type: ignore[arg-type]
    )


def test_refusal_when_nothing_clears_the_floor() -> None:
    """The floor is arithmetic, not a prompt instruction a model may ignore (feature 29)."""
    chunks = [make_chunk("a", "Evacuation guidance for riverine floods.")]
    retriever = build_retriever(
        chunks, [DenseHit("a", 0.9, "t", {})], {"a": 0.02}, relevance_floor=0.10
    )
    outcome = retriever.retrieve("what is the capital of France")

    assert outcome.refused is True
    assert outcome.results == []
    assert "floor" in outcome.refusal_reason


def test_evidence_returned_when_the_floor_is_cleared() -> None:
    chunks = [make_chunk("a", "Evacuation guidance for riverine floods.")]
    retriever = build_retriever(
        chunks, [DenseHit("a", 0.9, "t", {})], {"a": 0.80}, relevance_floor=0.10
    )
    outcome = retriever.retrieve("what should people do when water rises")

    assert outcome.refused is False
    assert [r.chunk_id for r in outcome.results] == ["a"]
    assert outcome.best_score == pytest.approx(0.80)
    assert outcome.citations == ["NDRP, p. 1"]


def test_refusal_when_nothing_is_retrieved_at_all() -> None:
    retriever = build_retriever([make_chunk("a", "text")], [])
    outcome = retriever.retrieve("zzz-nonexistent-term")
    assert outcome.refused is True
    assert outcome.refusal_reason == "no candidates retrieved"


def test_top_k_limits_what_reaches_the_model() -> None:
    chunks = [make_chunk(f"c{i}", f"Flood guidance number {i}.") for i in range(10)]
    dense = [DenseHit(f"c{i}", 0.9 - i / 100, "t", {}) for i in range(10)]
    scores = {f"c{i}": 0.9 for i in range(10)}
    retriever = build_retriever(chunks, dense, scores, top_k=3, relevance_floor=0.1)
    assert len(retriever.retrieve("flood guidance").results) == 3


def test_timings_are_recorded_for_each_stage() -> None:
    """Latency is instrumented here because features 62 and 71 both read these numbers."""
    chunks = [make_chunk("a", "Flood guidance.")]
    retriever = build_retriever(chunks, [DenseHit("a", 0.9, "t", {})], {"a": 0.9})
    outcome = retriever.retrieve("flood guidance")
    assert outcome.retrieval_ms >= 0
    assert outcome.rerank_ms >= 0


def test_document_filter_reaches_both_retrievers() -> None:
    chunks = [
        make_chunk("a", "Flood evacuation guidance.", doc="ndrp"),
        make_chunk("b", "Flood evacuation guidance.", doc="monsoon"),
    ]
    dense = [DenseHit("a", 0.9, "t", {}), DenseHit("b", 0.8, "t", {})]
    retriever = build_retriever(chunks, dense, {"a": 0.9, "b": 0.9})
    retriever.retrieve("flood evacuation", doc_ids=["monsoon"])

    store = retriever.store
    assert store.last_where == {"doc_id": {"$in": ["monsoon"]}}  # type: ignore[attr-defined]


def test_sparse_results_are_filtered_by_document() -> None:
    """BM25 has no metadata, so scoping is applied to its results (feature 33)."""
    chunks = [
        make_chunk("a", "Heatwave thresholds for Sindh province.", doc="ndrp"),
        make_chunk("b", "Heatwave thresholds for Sindh province.", doc="monsoon"),
    ]
    retriever = build_retriever(chunks, [], {"a": 0.9, "b": 0.9})
    outcome = retriever.retrieve("heatwave thresholds", doc_ids=["monsoon"])
    assert all(r.chunk.doc_id == "monsoon" for r in outcome.results)


def test_evidence_strength_bands() -> None:
    outcome = RetrievalOutcome(query="q", rewritten_query="q")
    outcome.results = [RetrievedChunk(chunk=make_chunk("a"))]
    outcome.best_score = 0.8
    assert outcome.evidence_strength(0.6, 0.25) == "High"
    outcome.best_score = 0.3
    assert outcome.evidence_strength(0.6, 0.25) == "Moderate"
    outcome.best_score = 0.1
    assert outcome.evidence_strength(0.6, 0.25) == "Low"


def test_evidence_strength_is_none_without_results() -> None:
    outcome = RetrievalOutcome(query="q", rewritten_query="q")
    assert outcome.evidence_strength(0.6, 0.25) == "None"


# --- LLM-backed rewriting (feature 30) ----------------------------------------------


class FakeResponse:
    def __init__(self, content: object) -> None:
        self.content = content


class FakeModel:
    def __init__(self, content: object, fail: bool = False) -> None:
        self.content = content
        self.fail = fail
        self.calls = 0

    def invoke(self, prompt: str) -> FakeResponse:
        self.calls += 1
        if self.fail:
            raise RuntimeError("provider unreachable")
        return FakeResponse(self.content)


def llm_rewriter_with(model: object) -> object:
    from citara.retrieval.llm_rewriter import LLMRewriter

    rewriter = LLMRewriter(Settings(_env_file=None))  # type: ignore[call-arg]
    rewriter._client = model  # type: ignore[attr-defined]
    return rewriter


def test_llm_rewriter_reads_structured_content_blocks() -> None:
    """Regression: LangChain 1.x returns content blocks, so str() yields a list repr."""
    blocks = [{"type": "text", "text": "What were the total recovery needs in Sindh?"}]
    rewriter = llm_rewriter_with(FakeModel(blocks))
    result = rewriter.rewrite("What about Sindh?", ["What were the total recovery needs?"])  # type: ignore[attr-defined]
    assert result == "What were the total recovery needs in Sindh?"


def test_llm_rewriter_accepts_plain_string_content() -> None:
    rewriter = llm_rewriter_with(FakeModel("Standalone question about Sindh?"))
    assert rewriter.rewrite("What about Sindh?", ["earlier turn?"]) == (  # type: ignore[attr-defined]
        "Standalone question about Sindh?"
    )


def test_llm_rewriter_falls_back_when_the_provider_fails() -> None:
    """A rewriting outage must degrade retrieval, not break it."""
    rewriter = llm_rewriter_with(FakeModel(None, fail=True))
    result = rewriter.rewrite("What about Sindh?", ["What were the recovery needs?"])  # type: ignore[attr-defined]
    assert "sindh" in result.lower()
    assert "recovery" in result.lower()  # heuristic carried the previous turn


def test_llm_rewriter_rejects_an_essay() -> None:
    """A long reply means the model explained instead of rewriting."""
    rewriter = llm_rewriter_with(FakeModel("word " * 200))
    result = rewriter.rewrite("What about Sindh?", ["What were the recovery needs?"])  # type: ignore[attr-defined]
    assert len(result) < 300


def test_llm_rewriter_leaves_standalone_questions_alone() -> None:
    """No provider call at all when the question already stands on its own."""
    model = FakeModel("should not be used")
    rewriter = llm_rewriter_with(model)
    question = "What is the monsoon contingency plan for Balochistan?"
    assert rewriter.rewrite(question, ["earlier turn?"]) == question  # type: ignore[attr-defined]
    assert model.calls == 0
