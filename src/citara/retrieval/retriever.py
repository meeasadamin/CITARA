"""The retrieval pipeline (features 24-33).

Wide recall first, narrow precision second: two cheap retrievers produce roughly two dozen
candidates, fusion merges them, and an expensive cross-encoder decides which five the model
sees.

The relevance floor is what makes refusal real. A prompt asking a model to say "I don't know"
is a request it may decline; a floor is arithmetic. If nothing clears it, generation never
runs - the refusal happens before a model is involved at all, which is the only way to
guarantee it.

The ``mode`` setting selects which stages run, so the four rows of the ablation table are the
same code path under different configuration rather than four separate implementations that
might differ in some way nobody noticed.
"""

from __future__ import annotations

import time
from typing import Any

from citara.chunking.models import Chunk
from citara.config import Settings, get_settings
from citara.embeddings import Embedder
from citara.indexing.builder import load_chunks
from citara.indexing.sparse_index import SparseHit, SparseIndex
from citara.indexing.vector_store import DenseHit, VectorStore
from citara.log import get_logger, stage
from citara.retrieval.fusion import reciprocal_rank_fusion
from citara.retrieval.llm_rewriter import LLMRewriter
from citara.retrieval.models import RetrievalOutcome
from citara.retrieval.query import Rewriter, prepare
from citara.retrieval.reranker import CrossEncoderReranker

log = get_logger("retrieval")


class HybridRetriever:
    """Dense + sparse retrieval, fusion, reranking and the refusal gate."""

    def __init__(
        self,
        settings: Settings | None = None,
        embedder: Embedder | None = None,
        store: VectorStore | None = None,
        sparse: SparseIndex | None = None,
        chunks: list[Chunk] | None = None,
        rewriter: Rewriter | None = None,
        reranker: CrossEncoderReranker | None = None,
    ) -> None:
        self.settings = settings or get_settings()
        self.embedder = embedder or Embedder(self.settings.embedding)
        self.store = store or VectorStore(self.settings)
        data_dir = self.settings.paths.resolved(self.settings.paths.data_dir)
        self.sparse = sparse or SparseIndex.load(
            self.settings.paths.resolved(self.settings.paths.bm25_path)
        )
        loaded = chunks if chunks is not None else load_chunks(data_dir / "chunks.jsonl")
        self.chunks_by_id: dict[str, Chunk] = {c.chunk_id: c for c in loaded}
        # A model rewrites follow-ups far better than the heuristic, but the heuristic is what
        # keeps retrieval working when no key is configured or the provider is unreachable.
        # Construction is lazy, so this costs nothing until a follow-up actually arrives.
        self.rewriter = rewriter or LLMRewriter(self.settings)
        self.reranker = reranker or CrossEncoderReranker(self.settings.retrieval)

    # -- filtering ------------------------------------------------------------------

    def _metadata_filter(
        self, doc_ids: list[str] | None, year: int | None
    ) -> dict[str, Any] | None:
        """Chroma filter applied before the search, not after (feature 33)."""
        clauses: list[dict[str, Any]] = []
        if doc_ids:
            clauses.append({"doc_id": {"$in": doc_ids}})
        if year is not None:
            clauses.append({"year": year})
        if not clauses:
            return None
        return clauses[0] if len(clauses) == 1 else {"$and": clauses}

    def _allowed_ids(self, doc_ids: list[str] | None, year: int | None) -> set[str] | None:
        """BM25 carries no metadata, so scoping is applied to its results directly."""
        if not doc_ids and year is None:
            return None
        return {
            chunk_id
            for chunk_id, chunk in self.chunks_by_id.items()
            if (not doc_ids or chunk.doc_id in doc_ids) and (year is None or chunk.year == year)
        }

    # -- retrieval ------------------------------------------------------------------

    def retrieve(
        self,
        question: str,
        history: list[str] | None = None,
        doc_ids: list[str] | None = None,
        year: int | None = None,
    ) -> RetrievalOutcome:
        """Run the funnel and decide whether there is admissible evidence."""
        config = self.settings.retrieval
        dense_query, sparse_query = prepare(question, history or [], config, self.rewriter)
        outcome = RetrievalOutcome(query=question, rewritten_query=dense_query)

        with stage(log, "retrieve", mode=config.mode) as details:
            started = time.perf_counter()
            dense: list[DenseHit] = []
            sparse: list[SparseHit] = []

            if config.mode in {"dense", "hybrid", "hybrid_rerank"}:
                dense = self.store.search(
                    self.embedder.embed_query(dense_query),
                    k=config.dense_k,
                    where=self._metadata_filter(doc_ids, year),
                )
            if config.mode in {"sparse", "hybrid", "hybrid_rerank"}:
                allowed = self._allowed_ids(doc_ids, year)
                # Over-fetch before filtering: the sparse index cannot filter during search.
                fetch = config.sparse_k * (4 if allowed else 1)
                sparse = self.sparse.search(sparse_query, k=fetch)
                if allowed is not None:
                    sparse = [hit for hit in sparse if hit.chunk_id in allowed]
                sparse = sparse[: config.sparse_k]

            outcome.dense_hits = len(dense)
            outcome.sparse_hits = len(sparse)

            candidates = reciprocal_rank_fusion(
                dense,
                sparse,
                self.chunks_by_id,
                k=config.rrf_k,
                dense_weight=config.dense_weight if config.mode != "sparse" else 0.0,
                sparse_weight=config.sparse_weight if config.mode != "dense" else 0.0,
            )[: config.rerank_candidates]
            outcome.candidates_considered = len(candidates)
            outcome.retrieval_ms = round((time.perf_counter() - started) * 1000, 1)

            selected = candidates[: config.top_k]

            # Measured on the gold set: this cross-encoder barely reorders these candidates
            # (Hit@5 0.375 -> 0.417, MRR 0.249 -> 0.274, within noise at n=24) but separates
            # answerable from unanswerable questions cleanly (median 0.92 against 0.077). It
            # therefore serves as the admissibility gate rather than the ranker, and scoring
            # only the chunks about to be served cuts its cost by roughly five times.
            if config.mode == "hybrid_rerank" and selected:
                rerank_started = time.perf_counter()
                selected = self.reranker.rerank(dense_query, selected)
                outcome.rerank_ms = round((time.perf_counter() - rerank_started) * 1000, 1)

            outcome.best_score = selected[0].score if selected else None

            # The floor applies to reranker scores, which are calibrated; fusion scores are
            # rank-derived and carry no meaning on an absolute scale, so in the ablation modes
            # that skip reranking only emptiness can trigger a refusal.
            if config.mode == "hybrid_rerank":
                admissible = [
                    candidate
                    for candidate in selected
                    if (candidate.rerank_score or 0.0) >= config.relevance_floor
                ]
            else:
                admissible = selected

            if len(admissible) < config.min_evidence_chunks:
                outcome.refused = True
                outcome.refusal_reason = (
                    "no chunk cleared the relevance floor"
                    if selected
                    else "no candidates retrieved"
                )
                outcome.results = []
            else:
                outcome.results = admissible

            details.update(
                {
                    "dense": outcome.dense_hits,
                    "sparse": outcome.sparse_hits,
                    "candidates": outcome.candidates_considered,
                    "returned": len(outcome.results),
                    "best_score": outcome.best_score,
                    "refused": outcome.refused,
                    "retrieval_ms": outcome.retrieval_ms,
                    "rerank_ms": outcome.rerank_ms,
                }
            )

        return outcome

    def close(self) -> None:
        self.store.close()
