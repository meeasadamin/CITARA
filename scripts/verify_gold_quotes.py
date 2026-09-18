"""Check every gold quote against the original PDF page it cites (feature 66).

The gold set was drafted from the chunk store, so checking it against that same store proves
nothing about chunk assembly, page spans or normalisation. This re-opens the source PDF,
extracts only the cited pages, and looks for the quote there - independent of chunking,
deduplication and the whole downstream pipeline.

What it cannot check is whether the quoted passage actually *answers* the question. That
judgement stays with a human.

Run:  uv run python scripts/verify_gold_quotes.py
"""

from __future__ import annotations

import argparse
import json
import sys

import pymupdf

from citara.config import get_settings
from citara.evaluation.gold import load_gold_set
from citara.ingestion.normalise import has_font_corruption, normalise_text


def flatten(text: str) -> str:
    """Whitespace-insensitive, case-insensitive form for comparison."""
    return " ".join(text.split()).lower()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Verify gold quotes against source PDFs.")
    parser.add_argument("--quiet", action="store_true", help="only show failures")
    args = parser.parse_args(argv)

    settings = get_settings()
    docs_dir = settings.paths.resolved(settings.paths.docs_dir)
    manifest_path = settings.paths.resolved(settings.paths.manifest_path)
    if not manifest_path.exists():
        print("No corpus manifest. Run: uv run python -m citara.ingestion")
        return 2

    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    filename_by_doc = {d["doc_id"]: d["filename"] for d in payload["documents"]}

    gold = load_gold_set(settings=settings)
    checked = confirmed = 0
    failures: list[str] = []

    for question in gold.answerable():
        for source in question.sources:
            if not source.quote:
                continue
            checked += 1
            filename = filename_by_doc.get(source.doc_id)
            if not filename or not (docs_dir / filename).exists():
                failures.append(f"{question.id}: no source PDF for {source.doc_id}")
                continue

            with pymupdf.open(docs_dir / filename) as doc:
                pages = range(source.page_start, min(source.page_end, doc.page_count) + 1)
                raw = "\n".join(doc.load_page(p - 1).get_text("text") for p in pages)

            repaired = normalise_text(raw, repair_font=has_font_corruption(raw))
            haystack = flatten(repaired)
            needle = flatten(source.quote)

            # Table quotes are rendered as Markdown rows, which do not appear verbatim in the
            # page text; fall back to checking the cell values are all present on the page.
            if needle in haystack:
                confirmed += 1
                if not args.quiet:
                    print(f"  [OK]   {question.id}  p{source.page_start}  {source.quote[:58]}...")
                continue

            cells = [flatten(c) for c in source.quote.split("|") if flatten(c)]
            if cells and all(cell in haystack for cell in cells):
                confirmed += 1
                if not args.quiet:
                    print(
                        f"  [OK*]  {question.id}  p{source.page_start}  "
                        f"table cells found: {source.quote[:48]}..."
                    )
                continue

            failures.append(
                f"{question.id}: quote not found on {source.doc_id} "
                f"p{source.page_start}-{source.page_end}: {source.quote[:70]!r}"
            )

    print(f"\n{confirmed}/{checked} quotes located on the exact page cited")
    for failure in failures:
        print(f"  [FAIL] {failure}")

    print(
        "\nStill requires a human: whether each quoted passage actually answers its question,\n"
        "and whether the expected answer is complete and correctly worded."
    )
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
