"""[3-4] Embedding and indexing: local BGE, ChromaDB, parallel BM25 (features 18-23).

Exports resolve on first use rather than at import, so that reaching for one light module
here - `fetch`, which the interface calls before anything else to download a published index -
does not load ChromaDB and torch first. See ``citara.generation`` for the same reasoning: an
eager re-export in a package __init__ is what once made a cold start show a blank page.
"""

from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from citara.indexing.builder import IndexManifest, build_index, load_chunks
    from citara.indexing.sparse_index import SparseHit, SparseIndex, tokenise
    from citara.indexing.vector_store import DenseHit, VectorStore, chunk_metadata

_EXPORTS = {
    "IndexManifest": "citara.indexing.builder",
    "build_index": "citara.indexing.builder",
    "load_chunks": "citara.indexing.builder",
    "SparseHit": "citara.indexing.sparse_index",
    "SparseIndex": "citara.indexing.sparse_index",
    "tokenise": "citara.indexing.sparse_index",
    "DenseHit": "citara.indexing.vector_store",
    "VectorStore": "citara.indexing.vector_store",
    "chunk_metadata": "citara.indexing.vector_store",
}

__all__ = [
    "DenseHit",
    "IndexManifest",
    "SparseHit",
    "SparseIndex",
    "VectorStore",
    "build_index",
    "chunk_metadata",
    "load_chunks",
    "tokenise",
]


def __getattr__(name: str) -> Any:
    if name in _EXPORTS:
        return getattr(import_module(_EXPORTS[name]), name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
