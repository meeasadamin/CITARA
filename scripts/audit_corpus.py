"""Phase 0 — corpus audit.

Reports, per PDF in the source directory: page count, text extractability, likely-scanned
pages, detected tables, Urdu content share, and publication year. Answers the three
pre-build questions: is the corpus extractable, is it the right size, and does Urdu need
to come back into scope?

Run:  uv run python scripts/audit_corpus.py [--docs docs] [--sample-tables 0]
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path

import pymupdf

# A page below this many characters carries no indexable prose (blank, cover, image-only).
SPARSE_PAGE_CHARS = 100
# A document whose mean page yields less than this is almost certainly scanned images.
SCANNED_MEAN_CHARS = 50
# Share of Urdu/Arabic-script characters above which a page counts as Urdu.
URDU_PAGE_RATIO = 0.20

ARABIC_SCRIPT = re.compile(r"[؀-ۿݐ-ݿﭐ-﷿ﹰ-﻿]")
# Not \b-anchored: filenames like "Advisory-11Sep2026" run the year straight onto a letter.
YEAR = re.compile(r"(?<!\d)(19|20)\d{2}(?!\d)")
# "2025-2030" is a coverage period, not a publication year: the first year is the publication one.
YEAR_RANGE = re.compile(r"(?<!\d)((?:19|20)\d{2})\s*[-–—]\s*(?:19|20)\d{2}(?!\d)")  # noqa: RUF001


@dataclass
class DocAudit:
    """Audit result for a single PDF."""

    filename: str
    size_mb: float
    pages: int = 0
    encrypted: bool = False
    error: str | None = None
    total_chars: int = 0
    mean_chars_per_page: float = 0.0
    sparse_pages: int = 0
    urdu_pages: int = 0
    pages_with_tables: int = 0
    tables_found: int = 0
    table_pages_sampled: int = 0
    year_guess: int | None = None
    verdict: str = ""
    sparse_page_numbers: list[int] = field(default_factory=list)


def guess_year(doc: pymupdf.Document, filename: str, front_text: str) -> int | None:
    """Publication year from the filename, then PDF metadata, then front-matter text."""
    for candidate in (filename, str(doc.metadata.get("creationDate", "") or "")):
        span = YEAR_RANGE.search(candidate)
        if span:
            return int(span.group(1))
        years = [int(m.group()) for m in YEAR.finditer(candidate)]
        if years:
            return max(years)
    years = [int(m.group()) for m in YEAR.finditer(front_text)]
    plausible = [y for y in years if 1990 <= y <= datetime.now(UTC).year]
    return max(plausible) if plausible else None


def audit_pdf(path: Path, sample_tables: int) -> DocAudit:
    """Audit one PDF. Never raises: a bad file is reported, not fatal (feature 5)."""
    result = DocAudit(filename=path.name, size_mb=round(path.stat().st_size / 1024**2, 1))
    try:
        doc = pymupdf.open(path)
    except Exception as exc:  # any parse failure is a reportable outcome, not fatal
        result.error = f"{type(exc).__name__}: {exc}"
        result.verdict = "UNREADABLE"
        return result

    with doc:
        if doc.needs_pass:
            result.encrypted = True
            result.verdict = "ENCRYPTED"
            return result

        result.pages = doc.page_count
        table_stride = max(1, doc.page_count // sample_tables) if sample_tables else 1
        front_text = ""

        for index, page in enumerate(doc):
            try:
                text = page.get_text("text")
            except Exception as exc:
                result.error = f"page {index + 1}: {type(exc).__name__}: {exc}"
                continue

            result.total_chars += len(text.strip())
            if len(text.strip()) < SPARSE_PAGE_CHARS:
                result.sparse_pages += 1
                if len(result.sparse_page_numbers) < 25:
                    result.sparse_page_numbers.append(index + 1)
            if text and len(ARABIC_SCRIPT.findall(text)) / max(len(text), 1) > URDU_PAGE_RATIO:
                result.urdu_pages += 1
            if index < 5:
                front_text += text

            if index % table_stride == 0:
                result.table_pages_sampled += 1
                try:
                    tables = page.find_tables()
                    count = len(tables.tables)
                except Exception:  # table finder is best-effort
                    count = 0
                if count:
                    result.pages_with_tables += 1
                    result.tables_found += count

        result.mean_chars_per_page = round(result.total_chars / max(result.pages, 1), 1)
        result.year_guess = guess_year(doc, path.stem, front_text)

    if result.mean_chars_per_page < SCANNED_MEAN_CHARS:
        result.verdict = "SCANNED — needs OCR, defer"
    elif result.sparse_pages > result.pages * 0.4:
        result.verdict = "PARTIAL — many empty pages, inspect"
    else:
        result.verdict = "OK"
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description="Audit the PDF corpus before ingestion.")
    parser.add_argument("--docs", default="docs", help="source directory (default: docs)")
    parser.add_argument(
        "--sample-tables",
        type=int,
        default=0,
        help="scan only N pages per document for tables (0 = every page, slower)",
    )
    parser.add_argument("--out", default="data/corpus_audit.json", help="JSON report path")
    args = parser.parse_args()

    docs_dir = Path(args.docs)
    pdfs = sorted(p for p in docs_dir.glob("*.pdf"))
    if not pdfs:
        print(f"No PDFs found in {docs_dir.resolve()}")
        return 1

    print(f"Auditing {len(pdfs)} PDF(s) in {docs_dir.resolve()}\n")
    results = [audit_pdf(p, args.sample_tables) for p in pdfs]

    header = (
        f"{'document':42} {'pages':>6} {'chars/pg':>9} {'sparse':>7} "
        f"{'tables':>7} {'urdu':>6}  verdict"
    )
    print(header)
    print("-" * len(header))
    for r in results:
        print(
            f"{r.filename[:42]:42} {r.pages:>6} {r.mean_chars_per_page:>9.0f} "
            f"{r.sparse_pages:>7} {r.tables_found:>7} {r.urdu_pages:>6}  {r.verdict}"
        )

    total_pages = sum(r.pages for r in results)
    usable_pages = sum(r.pages - r.sparse_pages for r in results if r.verdict == "OK")
    urdu_docs = [r for r in results if r.urdu_pages >= 5]
    partial = "PARTIAL — many empty pages, inspect"
    unusable = [r for r in results if r.verdict not in {"OK", partial}]

    print("\n--- gates ---")
    scale = "PASS" if 500 <= total_pages <= 2000 else "REVIEW"
    print(f"[{scale}] scale: {total_pages} pages total ({usable_pages} usable) — target 500-2,000")
    print(
        f"[{'PASS' if not unusable else 'FAIL'}] extractability: "
        f"{len(unusable)} document(s) unusable"
        + (f" ({', '.join(u.filename for u in unusable)})" if unusable else "")
    )
    print(
        f"[{'IN SCOPE' if len(urdu_docs) >= 3 else 'DEFERRED'}] Urdu: "
        f"{len(urdu_docs)} document(s) with substantial Urdu text — threshold is 3"
    )
    print(
        f"[INFO] tables: {sum(r.tables_found for r in results)} detected across "
        f"{sum(r.pages_with_tables for r in results)} pages"
    )

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps(
            {
                "audited_at": datetime.now(UTC).isoformat(),
                "docs_dir": str(docs_dir.resolve()),
                "totals": {
                    "documents": len(results),
                    "pages": total_pages,
                    "usable_pages": usable_pages,
                    "urdu_documents": len(urdu_docs),
                },
                "documents": [asdict(r) for r in results],
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"\nJSON report: {out_path}")

    print("\n--- rows for docs/SOURCES.md ---")
    for r in results:
        print(
            f"| | {r.filename.replace('.pdf', '')} | {r.filename} | <url> | <date> | "
            f"{r.year_guess or ''} | {r.pages} | {'yes' if r.verdict == 'OK' else 'NO'} | "
            f"{'yes' if r.tables_found else 'no'} | {'Urdu' if r.urdu_pages >= 5 else 'English'} |"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
