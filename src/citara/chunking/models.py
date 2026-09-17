"""Chunk records and the chunking report (features 15, 17).

A chunk inherits its parent pages' provenance in full. It also records the page *range* it
covers: prose that runs across a page break produces a chunk citing "pp. 46-47", which is
still exactly verifiable and avoids severing a procedure at the page boundary.
"""

from __future__ import annotations

import hashlib
from datetime import UTC, datetime
from typing import Literal

from pydantic import BaseModel, Field

ChunkKind = Literal["text", "table"]
ChunkStrategy = Literal["semantic", "recursive", "table"]


class Chunk(BaseModel):
    """One indexable unit of evidence."""

    model_config = {"frozen": True}

    chunk_id: str
    doc_id: str
    title: str
    filename: str
    page_start: int = Field(ge=1)
    page_end: int = Field(ge=1)
    total_pages: int = Field(ge=1)
    year: int | None = None
    kind: ChunkKind = "text"
    strategy: ChunkStrategy = "semantic"
    ordinal: int = Field(ge=0, description="Position within the document, for neighbour lookup.")
    content: str
    source_record_ids: tuple[str, ...] = ()
    duplicate_ids: tuple[str, ...] = Field(
        default=(),
        description="Near-duplicate chunks this one absorbed, e.g. the same advisory text "
        "republished in an earlier year (feature 17).",
    )

    @property
    def char_count(self) -> int:
        return len(self.content)

    @property
    def citation(self) -> str:
        """Page-precise source reference."""
        if self.page_start == self.page_end:
            return f"{self.title}, p. {self.page_start}"
        return f"{self.title}, pp. {self.page_start}-{self.page_end}"

    @staticmethod
    def make_id(doc_id: str, page_start: int, page_end: int, ordinal: int) -> str:
        """Deterministic, location-based id: stable across re-ingestion (feature 15)."""
        raw = f"{doc_id}|{page_start}|{page_end}|{ordinal}"
        return hashlib.sha1(raw.encode("utf-8")).hexdigest()[:16]


class ChunkReport(BaseModel):
    """What chunking produced, and what it discarded."""

    generated_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    config_fingerprint: str = ""
    records_in: int = 0
    chunks_out: int = 0
    text_chunks: int = 0
    table_chunks: int = 0
    semantic_chunks: int = 0
    recursive_chunks: int = 0
    dropped_short: int = Field(default=0, description="Below the minimum length (feature 16).")
    dropped_noise: int = Field(
        default=0, description="Contents-page dot leaders and similar filler (feature 4)."
    )
    duplicates_removed: int = 0
    oversized_resplit: int = 0
    unlocated_chunks: int = Field(
        default=0,
        description="Chunks whose position in the page stream could not be found, so their "
        "page span - and therefore their citation - is approximate. Must stay at zero.",
    )
    fallback_documents: list[str] = Field(
        default_factory=list, description="Documents where semantic chunking failed."
    )
    mean_chars: float = 0.0
    p95_chars: int = 0

    def summary(self) -> dict[str, object]:
        return {
            "records_in": self.records_in,
            "chunks_out": self.chunks_out,
            "text": self.text_chunks,
            "tables": self.table_chunks,
            "semantic": self.semantic_chunks,
            "recursive": self.recursive_chunks,
            "dropped_short": self.dropped_short,
            "dropped_noise": self.dropped_noise,
            "duplicates_removed": self.duplicates_removed,
            "oversized_resplit": self.oversized_resplit,
            "unlocated_chunks": self.unlocated_chunks,
            "fallbacks": len(self.fallback_documents),
            "mean_chars": round(self.mean_chars, 1),
            "p95_chars": self.p95_chars,
        }
