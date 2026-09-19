"""The boundary of the assistant's knowledge, as the sidebar shows it (feature 57).

Users hallucinate about what a system knows as readily as models hallucinate facts. The
sidebar therefore lists exactly what is searchable, read from the index that retrieval
actually uses, and separately what was collected but is not searchable - a scanned document
that needs OCR is part of the corpus on paper and absent from every answer.

Pages are counted the same way. 159 of the 1,391 collected pages are scanned images with no
extractable text - 61 of the National Disaster Response Plan's 110 - so the sidebar reports
searchable pages, not the page count of the PDF, and says which documents are incomplete.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from citara.config import Settings


@dataclass(frozen=True)
class CorpusDocument:
    """One document as the interface describes it."""

    doc_id: str
    title: str
    year: int | None
    total_pages: int
    searchable_pages: int
    chunks: int
    note: str = ""

    @property
    def searchable(self) -> bool:
        return self.chunks > 0

    @property
    def label(self) -> str:
        """Title with year, for the filter and the sidebar list."""
        return f"{self.title} ({self.year})" if self.year else self.title


@dataclass(frozen=True)
class Corpus:
    """Every collected document, split by whether answers can draw on it."""

    documents: list[CorpusDocument]

    @property
    def searchable(self) -> list[CorpusDocument]:
        return [d for d in self.documents if d.searchable]

    @property
    def unsearchable(self) -> list[CorpusDocument]:
        return [d for d in self.documents if not d.searchable]

    @property
    def searchable_pages(self) -> int:
        return sum(d.searchable_pages for d in self.searchable)

    @property
    def total_pages(self) -> int:
        return sum(d.total_pages for d in self.documents)

    @property
    def years(self) -> list[int]:
        return sorted({d.year for d in self.searchable if d.year}, reverse=True)


def _read(path: Path) -> dict[str, Any]:
    """A manifest as JSON, or empty when missing or unreadable (the health check reports it)."""
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return payload if isinstance(payload, dict) else {}


def load_corpus(settings: Settings) -> Corpus:
    """Join what was collected (corpus manifest) with what was indexed (index manifest)."""
    paths = settings.paths
    collected: list[Any] = _read(paths.resolved(paths.manifest_path)).get("documents", [])
    indexed: list[Any] = _read(paths.resolved(paths.index_manifest_path)).get("documents", [])
    chunks_by_doc = {
        str(d["doc_id"]): int(d.get("chunks", 0))
        for d in indexed
        if isinstance(d, dict) and "doc_id" in d
    }

    documents: list[CorpusDocument] = []
    for entry in collected:
        if not isinstance(entry, dict):
            continue
        doc_id = str(entry.get("doc_id", ""))
        chunks = chunks_by_doc.get(doc_id, 0)
        total = int(entry.get("total_pages") or 0)
        searchable = int(entry.get("pages_indexed") or 0) if chunks else 0
        image_only = int(entry.get("pages_image_only") or 0)
        note = ""
        if not chunks:
            note = (
                "scanned images; needs OCR before it can be searched"
                if entry.get("needs_ocr")
                else "not in the index"
            )
        elif image_only:
            note = f"{image_only} scanned page{'s' if image_only != 1 else ''} not searchable"
        year = entry.get("year")
        documents.append(
            CorpusDocument(
                doc_id=doc_id,
                title=str(entry.get("title") or doc_id),
                year=int(year) if isinstance(year, int) and year > 0 else None,
                total_pages=total,
                searchable_pages=searchable,
                chunks=chunks,
                note=note,
            )
        )

    documents.sort(key=lambda d: (-(d.year or 0), d.title.lower()))
    return Corpus(documents=documents)
