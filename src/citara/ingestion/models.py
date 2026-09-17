"""Records produced by ingestion (features 2, 10, 11).

Provenance is established here, at the moment of extraction, and carried forward unchanged.
It is never reconstructed at output time - reconstruction is where citation systems start
quietly lying about which page a claim came from.
"""

from __future__ import annotations

import hashlib
from datetime import UTC, datetime
from typing import Literal

from pydantic import BaseModel, Field

RecordKind = Literal["text", "table"]


class PageRecord(BaseModel):
    """One indexable unit of text: a page's prose, or one table from that page."""

    model_config = {"frozen": True}

    record_id: str = Field(description="Deterministic id; stable across re-ingestion.")
    doc_id: str = Field(description="Slug identifying the source document.")
    title: str = Field(description="Human-readable document title, shown in citations.")
    filename: str
    page_number: int = Field(ge=1, description="1-based, matching the PDF's own numbering.")
    total_pages: int = Field(ge=1)
    year: int | None = Field(default=None, description="Publication year, filterable (feature 10).")
    kind: RecordKind = "text"
    table_index: int | None = Field(default=None, description="Ordinal of a table on its page.")
    content: str

    @property
    def char_count(self) -> int:
        return len(self.content)

    @property
    def citation(self) -> str:
        """Rendered source reference, e.g. 'NDRP 2019, p. 47'."""
        return f"{self.title}, p. {self.page_number}"

    @staticmethod
    def make_id(doc_id: str, page_number: int, kind: RecordKind, index: int = 0) -> str:
        """Deterministic record id.

        Content is deliberately excluded: the id identifies a *location*, so re-ingesting an
        updated document keeps ids stable for incremental indexing (feature 22) and lets a
        generated citation be mapped back to exact evidence (feature 15).
        """
        raw = f"{doc_id}|{page_number}|{kind}|{index}"
        return hashlib.sha1(raw.encode("utf-8")).hexdigest()[:16]


class DocumentReport(BaseModel):
    """Per-document ingestion outcome, including what was skipped and why."""

    filename: str
    doc_id: str = ""
    title: str = ""
    year: int | None = None
    sha256: str = ""
    total_pages: int = 0
    pages_indexed: int = 0
    pages_sparse: int = 0
    pages_image_only: int = 0
    tables_extracted: int = 0
    tables_flagged: int = Field(
        default=0, description="Detected but too small or malformed to trust (feature 9)."
    )
    text_records: int = 0
    font_repaired: bool = Field(
        default=False, description="Broken-font substitutions were applied to this document."
    )
    needs_ocr: bool = Field(
        default=False, description="Near-zero extractable text: reported, not silently skipped."
    )
    error: str | None = None
    duration_ms: float = 0.0

    @property
    def ok(self) -> bool:
        return self.error is None and not self.needs_ocr


class CorpusManifest(BaseModel):
    """Machine- and human-readable record of an ingestion run (feature 11)."""

    generated_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    config_fingerprint: str = ""
    documents: list[DocumentReport] = Field(default_factory=list)

    @property
    def documents_indexed(self) -> int:
        return sum(1 for d in self.documents if d.ok)

    @property
    def documents_failed(self) -> int:
        return sum(1 for d in self.documents if not d.ok)

    @property
    def total_pages(self) -> int:
        return sum(d.total_pages for d in self.documents)

    @property
    def pages_indexed(self) -> int:
        return sum(d.pages_indexed for d in self.documents)

    @property
    def total_records(self) -> int:
        return sum(d.text_records + d.tables_extracted for d in self.documents)

    @property
    def tables_extracted(self) -> int:
        return sum(d.tables_extracted for d in self.documents)

    def summary(self) -> dict[str, object]:
        """Totals for logging and for the README artifact.

        Known gaps are reported alongside successes: a corpus build that hides what it could
        not read is how a knowledge base acquires silent holes.
        """
        return {
            "documents_indexed": self.documents_indexed,
            "documents_failed": self.documents_failed,
            "total_pages": self.total_pages,
            "pages_indexed": self.pages_indexed,
            "pages_image_only": sum(d.pages_image_only for d in self.documents),
            "records": self.total_records,
            "tables": self.tables_extracted,
            "tables_flagged": sum(d.tables_flagged for d in self.documents),
            "fonts_repaired": sum(1 for d in self.documents if d.font_repaired),
            "documents_needing_ocr": sum(1 for d in self.documents if d.needs_ocr),
        }
