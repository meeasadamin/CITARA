"""[2] Chunking: semantic + recursive fallback, overlap, dedup (features 12-17)."""

from citara.chunking.dedup import deduplicate
from citara.chunking.models import Chunk, ChunkReport
from citara.chunking.pipeline import chunk_records
from citara.chunking.semantic import recursive_split, semantic_split, split_sentences

__all__ = [
    "Chunk",
    "ChunkReport",
    "chunk_records",
    "deduplicate",
    "recursive_split",
    "semantic_split",
    "split_sentences",
]
