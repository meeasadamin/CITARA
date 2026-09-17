"""[3-4] Embedding and indexing: local BGE, ChromaDB, parallel BM25 (features 18-23)."""

from citara.indexing.builder import IndexManifest, build_index, load_chunks
from citara.indexing.sparse_index import SparseHit, SparseIndex, tokenise
from citara.indexing.vector_store import DenseHit, VectorStore, chunk_metadata

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
