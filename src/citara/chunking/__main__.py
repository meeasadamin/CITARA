"""Command-line entry point: ``uv run python -m citara.chunking``.

Reads the page records produced by ingestion and writes chunks ready for indexing.
"""

from __future__ import annotations

import argparse
import json
import sys

import numpy as np

from citara.chunking.dedup import deduplicate
from citara.chunking.pipeline import chunk_records
from citara.config import get_settings
from citara.embeddings import Embedder
from citara.ingestion.store import read_records
from citara.log import configure_logging, get_logger

log = get_logger("chunking.cli")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Chunk ingested page records.")
    parser.add_argument("--no-dedup", action="store_true", help="skip near-duplicate removal")
    parser.add_argument("--json-logs", action="store_true")
    args = parser.parse_args(argv)

    settings = get_settings()
    configure_logging(settings.log_level, args.json_logs)
    settings.ensure_directories()

    data_dir = settings.paths.resolved(settings.paths.data_dir)
    records_path = data_dir / "pages.jsonl"
    if not records_path.exists():
        log.error(
            "no page records found; run ingestion first",
            extra={"path": str(records_path), "command": "python -m citara.ingestion"},
        )
        return 2

    records = list(read_records(records_path))
    embedder = Embedder(settings.embedding)
    chunks, report = chunk_records(records, settings, embedder)

    if not args.no_dedup:
        chunks, removed, vectors = deduplicate(
            chunks, embedder, settings.chunking.near_duplicate_threshold
        )
        report.duplicates_removed = removed
        report.chunks_out = len(chunks)
        # Indexing reuses these rather than embedding the corpus a second time (~10 min on CPU).
        np.save(data_dir / "chunk_vectors.npy", vectors)

    chunks_path = data_dir / "chunks.jsonl"
    with chunks_path.open("w", encoding="utf-8") as handle:
        for chunk in chunks:
            handle.write(chunk.model_dump_json() + "\n")

    report_path = data_dir / "chunk_report.json"
    payload = report.model_dump(mode="json")
    payload["totals"] = report.summary()
    report_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

    totals = report.summary()
    print(
        f"\n{totals['records_in']} records -> {totals['chunks_out']} chunks "
        f"({totals['text']} text, {totals['tables']} tables)."
    )
    print(
        f"Strategy: {totals['semantic']} semantic, {totals['recursive']} recursive"
        f" ({totals['fallbacks']} document(s) fell back)."
    )
    print(
        f"Filtered: {totals['dropped_short']} too short, "
        f"{totals['dropped_noise']} contents-page filler, "
        f"{totals['duplicates_removed']} near-duplicates, "
        f"{totals['oversized_resplit']} oversized re-split."
    )
    if totals["unlocated_chunks"]:
        print(f"WARNING: {totals['unlocated_chunks']} chunk(s) have an approximate page span.")
    print(f"Size: mean {totals['mean_chars']} chars, p95 {totals['p95_chars']}.")
    print(f"chunks: {chunks_path}")
    print(f"report: {report_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
