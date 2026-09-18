"""Weighted reciprocal rank fusion (feature 26).

Fusion works on *ranks*, not scores, and that is the point. Cosine similarity lives in
[0, 1] while BM25 is unbounded and corpus-dependent; combining them numerically would mean
inventing a normalisation and then defending it. Ranks are directly comparable, so a chunk
that both retrievers place near the top outranks one that only a single retriever liked.

    score(chunk) = w_dense / (k + rank_dense) + w_sparse / (k + rank_sparse)

The constant k damps the difference between the top ranks, so rank 1 does not overwhelm
everything a second retriever found.
"""

from __future__ import annotations

from citara.chunking.models import Chunk
from citara.indexing.sparse_index import SparseHit
from citara.indexing.vector_store import DenseHit
from citara.retrieval.models import RetrievedChunk


def reciprocal_rank_fusion(
    dense: list[DenseHit],
    sparse: list[SparseHit],
    chunks_by_id: dict[str, Chunk],
    *,
    k: int = 60,
    dense_weight: float = 0.5,
    sparse_weight: float = 0.5,
) -> list[RetrievedChunk]:
    """Merge two ranked lists into one, best first."""
    merged: dict[str, RetrievedChunk] = {}

    for rank, hit in enumerate(dense, start=1):
        chunk = chunks_by_id.get(hit.chunk_id)
        if chunk is None:
            continue
        entry = merged.setdefault(hit.chunk_id, RetrievedChunk(chunk=chunk))
        entry.dense_rank = rank
        entry.dense_score = hit.score
        entry.fusion_score += dense_weight / (k + rank)

    for rank, sparse_hit in enumerate(sparse, start=1):
        chunk = chunks_by_id.get(sparse_hit.chunk_id)
        if chunk is None:
            continue
        entry = merged.setdefault(sparse_hit.chunk_id, RetrievedChunk(chunk=chunk))
        entry.sparse_rank = rank
        entry.sparse_score = sparse_hit.score
        entry.fusion_score += sparse_weight / (k + rank)

    return sorted(
        merged.values(),
        # Ties broken deterministically so a run is reproducible.
        key=lambda item: (-item.fusion_score, item.chunk_id),
    )
