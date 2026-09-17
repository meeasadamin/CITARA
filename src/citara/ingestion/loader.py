"""PDF loading with page-level provenance (features 1-6, 10, 11).

One corrupt file must never abort a corpus build, and a document that yields no text must be
reported rather than silently contributing nothing.
"""

from __future__ import annotations

import hashlib
import re
import time
from pathlib import Path

import pymupdf

from citara.config import Settings, get_settings
from citara.ingestion.models import CorpusManifest, DocumentReport, PageRecord
from citara.ingestion.normalise import has_font_corruption, normalise_markdown, normalise_text
from citara.ingestion.tables import extract_tables
from citara.log import get_logger, stage

log = get_logger("ingestion")

_YEAR = re.compile(r"(?<!\d)(19|20)\d{2}(?!\d)")
_YEAR_RANGE = re.compile(
    # En and em dashes are intentional: NDMA titles use both in "2025-2030" ranges.
    r"(?<!\d)((?:19|20)\d{2})\s*[-–—]\s*(?:19|20)\d{2}(?!\d)"  # noqa: RUF001
)
_TITLE_NOISE = re.compile(r"^(microsoft word|untitled|document\d*|print|final)", re.IGNORECASE)
# How much of a text block must sit inside a table before it counts as table content.
_TABLE_OVERLAP = 0.6
# Characters of body text sampled when deciding whether a document has a broken font encoding.
_CORRUPTION_SAMPLE_CHARS = 400_000


def slugify(name: str) -> str:
    """Stable document id from a filename stem."""
    return re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")


def guess_year(filename: str, metadata: dict[str, str], front_text: str) -> int | None:
    """Publication year from filename, then PDF metadata, then front matter (feature 10).

    A range such as "2025-2030" is a coverage period: its first year is the publication year.
    """
    for candidate in (filename, str(metadata.get("creationDate") or "")):
        span = _YEAR_RANGE.search(candidate)
        if span:
            return int(span.group(1))
        years = [int(m.group()) for m in _YEAR.finditer(candidate)]
        if years:
            return max(years)
    years = [int(m.group()) for m in _YEAR.finditer(front_text)]
    return max(years) if years else None


def derive_title(path: Path, metadata: dict[str, str]) -> str:
    """Readable title for citations.

    Prefers the filename: NDMA PDF metadata is frequently a toolchain artefact
    ("Microsoft Word - final_v3.docx"), which would make every citation unreadable.
    """
    stem = path.stem.replace("_", " ").replace("-", " ")
    stem = re.sub(r"\s+", " ", stem).strip()
    embedded = (metadata.get("title") or "").strip()
    if embedded and len(embedded) > 12 and not _TITLE_NOISE.match(embedded):
        return embedded
    return stem


def _rect_overlap_ratio(block: tuple[float, float, float, float], table: pymupdf.Rect) -> float:
    """Fraction of *block* covered by *table*."""
    rect = pymupdf.Rect(block)
    if rect.is_empty:
        return 0.0
    intersection = rect & pymupdf.Rect(table)
    if intersection.is_empty:
        return 0.0
    return float(abs(intersection.get_area()) / abs(rect.get_area()))


def _prose_outside_tables(page: pymupdf.Page, table_rects: list[pymupdf.Rect]) -> str:
    """Page text with table regions removed.

    Without this the same figures are indexed twice: once as a clean Markdown table and once
    as the flattened digit soup the table serialisation exists to avoid.
    """
    if not table_rects:
        return str(page.get_text("text"))

    kept: list[str] = []
    for block in page.get_text("blocks"):
        bbox = (block[0], block[1], block[2], block[3])
        text = str(block[4])
        if any(_rect_overlap_ratio(bbox, rect) > _TABLE_OVERLAP for rect in table_rects):
            continue
        kept.append(text)
    return "\n".join(kept)


def file_digest(path: Path) -> str:
    """SHA-256 of the file, so re-ingestion can skip unchanged documents (feature 22)."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def load_document(path: Path, settings: Settings) -> tuple[list[PageRecord], DocumentReport]:
    """Extract one PDF into records plus a report. Never raises (feature 5)."""
    started = time.perf_counter()
    report = DocumentReport(filename=path.name, doc_id=slugify(path.stem))
    records: list[PageRecord] = []
    config = settings.ingestion

    try:
        doc = pymupdf.open(path)
    except Exception as exc:  # a bad file is a reported outcome, not a crash
        report.error = f"{type(exc).__name__}: {exc}"
        log.error("document unreadable", extra={"document": path.name, "error": report.error})
        return [], report

    try:
        with doc:
            if doc.needs_pass:
                report.error = "encrypted"
                return [], report

            report.sha256 = file_digest(path)
            report.total_pages = doc.page_count
            metadata = {k: str(v) for k, v in (doc.metadata or {}).items() if v}
            report.title = derive_title(path, metadata)

            raw_pages: list[tuple[int, str, list[str], int]] = []
            front_text = ""
            total_chars = 0

            for index in range(doc.page_count):
                page = doc.load_page(index)
                page_number = index + 1
                try:
                    tables, flagged, rects = extract_tables(page, config)
                    prose = _prose_outside_tables(page, rects)
                except Exception as exc:  # isolate the failure to this page
                    report.error = f"page {page_number}: {type(exc).__name__}: {exc}"
                    continue

                total_chars += len(prose.strip())
                if index < 5:
                    front_text += prose
                raw_pages.append((page_number, prose, tables, flagged))

            report.year = guess_year(path.stem, metadata, front_text)
            mean_chars = total_chars / max(report.total_pages, 1)
            report.needs_ocr = mean_chars < config.scanned_mean_chars
            if report.needs_ocr:
                log.warning(
                    "document appears scanned; skipped",
                    extra={"document": path.name, "mean_chars_per_page": round(mean_chars, 1)},
                )
                return [], report

            # Font repair is decided per document, from a sample spanning the whole document.
            # Front matter alone is not enough: covers and contents pages carry almost no
            # prose, so a corpus-wide scan of the first pages found nothing while the body
            # text was full of 'evacuaƟon' and 'QueƩa'.
            sample = "\n".join(text for _, text, _, _ in raw_pages)[:_CORRUPTION_SAMPLE_CHARS]
            report.font_repaired = config.normalise_ligatures and has_font_corruption(sample)

            for page_number, prose, tables, flagged in raw_pages:
                report.tables_flagged += flagged
                text = normalise_text(
                    prose,
                    repair_font=report.font_repaired,
                    dehyphenate_words=config.dehyphenate,
                )
                has_content = False

                if len(text) >= config.sparse_page_chars:
                    records.append(
                        PageRecord(
                            record_id=PageRecord.make_id(report.doc_id, page_number, "text"),
                            doc_id=report.doc_id,
                            title=report.title,
                            filename=path.name,
                            page_number=page_number,
                            total_pages=report.total_pages,
                            year=report.year,
                            kind="text",
                            content=text,
                        )
                    )
                    report.text_records += 1
                    has_content = True
                else:
                    report.pages_sparse += 1
                    if not tables:
                        report.pages_image_only += 1

                for table_index, table_markdown in enumerate(tables):
                    repaired = normalise_markdown(table_markdown, repair_font=report.font_repaired)
                    records.append(
                        PageRecord(
                            record_id=PageRecord.make_id(
                                report.doc_id, page_number, "table", table_index
                            ),
                            doc_id=report.doc_id,
                            title=report.title,
                            filename=path.name,
                            page_number=page_number,
                            total_pages=report.total_pages,
                            year=report.year,
                            kind="table",
                            table_index=table_index,
                            content=repaired,
                        )
                    )
                    report.tables_extracted += 1
                    has_content = True

                if has_content:
                    report.pages_indexed += 1
    except Exception as exc:  # last-resort isolation
        report.error = f"{type(exc).__name__}: {exc}"
        log.exception("document failed", extra={"document": path.name})

    report.duration_ms = round((time.perf_counter() - started) * 1000, 1)
    return records, report


def ingest_corpus(settings: Settings | None = None) -> tuple[list[PageRecord], CorpusManifest]:
    """Ingest every PDF in the configured docs directory (feature 1).

    Subdirectories are ignored, which is how deferred documents (``docs/_deferred_needs_ocr``)
    stay out of the index without being deleted.
    """
    settings = settings or get_settings()
    docs_dir = settings.paths.resolved(settings.paths.docs_dir)
    pdfs = sorted(docs_dir.glob("*.pdf"))

    manifest = CorpusManifest(config_fingerprint=settings.fingerprint())
    records: list[PageRecord] = []

    with stage(log, "ingest_corpus", documents=len(pdfs)) as details:
        for path in pdfs:
            doc_records, report = load_document(path, settings)
            records.extend(doc_records)
            manifest.documents.append(report)
            log.info(
                "document ingested",
                extra={
                    "document": path.name,
                    "pages": report.total_pages,
                    "indexed": report.pages_indexed,
                    "tables": report.tables_extracted,
                    "font_repaired": report.font_repaired,
                    "needs_ocr": report.needs_ocr,
                    "duration_ms": report.duration_ms,
                },
            )
        details.update(manifest.summary())

    return records, manifest
