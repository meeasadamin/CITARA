"""Command-line entry point: ``uv run python -m citara.ingestion`` (feature 1).

One command turns the PDF directory into page records plus a corpus manifest.
"""

from __future__ import annotations

import argparse
import sys

from citara.config import get_settings
from citara.ingestion.loader import ingest_corpus
from citara.ingestion.store import write_manifest, write_records
from citara.log import configure_logging, get_logger

log = get_logger("ingestion.cli")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Ingest the NDMA PDF corpus.")
    parser.add_argument("--docs", help="override the source directory")
    parser.add_argument("--json-logs", action="store_true", help="emit JSON log lines")
    parser.add_argument("--quiet", action="store_true", help="only warnings and errors")
    args = parser.parse_args(argv)

    settings = get_settings()
    if args.docs:
        from pathlib import Path

        settings = settings.model_copy(
            update={"paths": settings.paths.model_copy(update={"docs_dir": Path(args.docs)})}
        )

    configure_logging("WARNING" if args.quiet else settings.log_level, args.json_logs)
    settings.ensure_directories()

    docs_dir = settings.paths.resolved(settings.paths.docs_dir)
    if not docs_dir.is_dir():
        log.error("source directory not found", extra={"path": str(docs_dir)})
        return 2

    records, manifest = ingest_corpus(settings)
    if not records:
        log.error("no records produced; nothing to index", extra={"path": str(docs_dir)})
        return 1

    records_path = settings.paths.resolved(settings.paths.data_dir) / "pages.jsonl"
    manifest_path = settings.paths.resolved(settings.paths.manifest_path)
    write_records(records, records_path)
    write_manifest(manifest, manifest_path)

    totals = manifest.summary()
    print(
        f"\nIngested {totals['documents_indexed']} document(s), "
        f"{totals['pages_indexed']}/{totals['total_pages']} pages indexed, "
        f"{totals['records']} records ({totals['tables']} tables), "
        f"{totals['fonts_repaired']} document(s) font-repaired."
    )
    print(
        f"Known gaps: {totals['pages_image_only']} image-only page(s), "
        f"{totals['tables_flagged']} table(s) flagged as unreliable, "
        f"{totals['documents_needing_ocr']} document(s) needing OCR."
    )
    failed = [d for d in manifest.documents if not d.ok]
    for document in failed:
        reason = document.error or ("scanned, needs OCR" if document.needs_ocr else "unknown")
        print(f"  skipped: {document.filename} - {reason}")
    print(f"records:  {records_path}")
    print(f"manifest: {manifest_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
