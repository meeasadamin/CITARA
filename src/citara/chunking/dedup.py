"""Near-duplicate detection (feature 17).

NDMA republishes advisories annually with small edits, so the same paragraph exists in the
2025 and 2026 monsoon plans almost verbatim. Left alone, a top-5 evidence set can be five
copies of one paragraph: the reranker is doing its job, and the LLM still receives one fact
instead of five.

The surviving copy is the most recent, and it records the ids it absorbed, so recency is
respected and the older printings remain traceable rather than vanishing silently.
"""

from __future__ import annotations

import numpy as np

from citara.chunking.models import Chunk
from citara.embeddings import Embedder
from citara.log import get_logger, stage

log = get_logger("chunking.dedup")

# Compared in blocks to keep the similarity matrix small on a constrained host.
_BLOCK = 512


def _recency_key(chunk: Chunk) -> tuple[int, str, int]:
    """Newest first; ties broken deterministically so runs are reproducible."""
    return (-(chunk.year or 0), chunk.doc_id, chunk.page_start)


def deduplicate(
    chunks: list[Chunk], embedder: Embedder, threshold: float
) -> tuple[list[Chunk], int, np.ndarray]:
    """Drop chunks that duplicate an already-kept chunk above *threshold* cosine similarity.

    Returns the survivors, the number removed, and their embeddings in the same order.
    Embedding this corpus on CPU costs about ten minutes, so the vectors computed here are
    handed to indexing rather than recomputed from scratch.
    """
    if len(chunks) < 2:
        vectors = embedder.embed_documents([c.content for c in chunks])
        return chunks, 0, vectors

    with stage(log, "deduplicate", chunks=len(chunks), threshold=threshold) as details:
        order = sorted(range(len(chunks)), key=lambda i: _recency_key(chunks[i]))
        vectors = embedder.embed_documents([chunks[i].content for i in order])

        kept_indices: list[int] = []
        kept_vectors: list[np.ndarray] = []
        absorbed: dict[int, list[str]] = {}

        for position, index in enumerate(order):
            vector = vectors[position]
            duplicate_of: int | None = None
            for start in range(0, len(kept_vectors), _BLOCK):
                block = np.stack(kept_vectors[start : start + _BLOCK])
                similarities = block @ vector
                best = int(np.argmax(similarities))
                if similarities[best] >= threshold:
                    duplicate_of = kept_indices[start + best]
                    break
            if duplicate_of is None:
                kept_indices.append(index)
                kept_vectors.append(vector)
            else:
                absorbed.setdefault(duplicate_of, []).append(chunks[index].chunk_id)

        vector_by_index = dict(zip(kept_indices, kept_vectors, strict=True))
        surviving = [i for i in range(len(chunks)) if i in vector_by_index]
        result = [
            chunks[i].model_copy(update={"duplicate_ids": tuple(absorbed.get(i, ()))})
            for i in surviving
        ]
        kept_matrix = (
            np.stack([vector_by_index[i] for i in surviving])
            if surviving
            else np.zeros((0, embedder.dimension), dtype=np.float32)
        )
        removed = len(chunks) - len(result)
        details["removed"] = removed
        details["kept"] = len(result)

    return result, removed, kept_matrix
