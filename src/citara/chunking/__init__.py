"""[2] Chunking: semantic + recursive fallback, overlap, dedup (features 12-17).

Exports resolve on first use rather than at import, so ``citara.chunking.models`` - which
every retrieved result carries - does not load the semantic splitter's embedding stack
(see ``citara.generation``).
"""

from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from citara.chunking.dedup import deduplicate
    from citara.chunking.models import Chunk, ChunkReport
    from citara.chunking.pipeline import chunk_records
    from citara.chunking.semantic import recursive_split, semantic_split, split_sentences

_EXPORTS = {
    "deduplicate": "citara.chunking.dedup",
    "Chunk": "citara.chunking.models",
    "ChunkReport": "citara.chunking.models",
    "chunk_records": "citara.chunking.pipeline",
    "recursive_split": "citara.chunking.semantic",
    "semantic_split": "citara.chunking.semantic",
    "split_sentences": "citara.chunking.semantic",
}

__all__ = [
    "Chunk",
    "ChunkReport",
    "chunk_records",
    "deduplicate",
    "recursive_split",
    "semantic_split",
    "split_sentences",
]


def __getattr__(name: str) -> Any:
    if name in _EXPORTS:
        return getattr(import_module(_EXPORTS[name]), name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
