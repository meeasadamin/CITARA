"""Inspect the chunk store and assert the invariants chunking promises.

Green unit tests are not evidence that a corpus build is correct: every defect found in this
project so far passed its tests and failed on the real documents. This checks the artifact
itself.

Run:  uv run python scripts/inspect_chunks.py [--samples 3]
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

from citara.chunking.models import Chunk
from citara.config import get_settings


def load_chunks(path: Path) -> list[Chunk]:
    with path.open("r", encoding="utf-8") as handle:
        return [Chunk.model_validate_json(line) for line in handle if line.strip()]


def check(name: str, passed: bool, detail: str = "") -> bool:
    print(f"  [{'PASS' if passed else 'FAIL'}] {name}{f' - {detail}' if detail else ''}")
    return passed


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Inspect chunked corpus output.")
    parser.add_argument("--samples", type=int, default=2, help="sample chunks to print")
    args = parser.parse_args(argv)

    settings = get_settings()
    data_dir = settings.paths.resolved(settings.paths.data_dir)
    chunks_path = data_dir / "chunks.jsonl"
    report_path = data_dir / "chunk_report.json"

    if not chunks_path.exists():
        print(f"No chunks at {chunks_path}. Run: uv run python -m citara.chunking")
        return 2

    chunks = load_chunks(chunks_path)
    report: dict[str, Any] = {}
    if report_path.exists():
        report = json.loads(report_path.read_text(encoding="utf-8")).get("totals", {})

    config = settings.chunking
    text = [c for c in chunks if c.kind == "text"]
    tables = [c for c in chunks if c.kind == "table"]
    spanning = [c for c in text if c.page_end > c.page_start]
    strategies = Counter(c.strategy for c in chunks)
    lengths = sorted(c.char_count for c in chunks)

    print(f"\n{len(chunks)} chunks from {len({c.doc_id for c in chunks})} documents")
    print(f"  text {len(text)} | tables {len(tables)} | spanning pages {len(spanning)}")
    print(f"  strategies: {dict(strategies)}")
    print(
        f"  size: min {lengths[0]} | p50 {lengths[len(lengths) // 2]} | "
        f"p95 {lengths[int(len(lengths) * 0.95)]} | max {lengths[-1]}"
    )
    if report:
        print(f"  report: {report}")

    print("\ninvariants:")
    undersized = [c for c in chunks if c.kind == "text" and c.char_count < config.min_chunk_chars]
    oversized = [c for c in chunks if c.char_count > config.max_chunk_chars]
    bad_pages = [c for c in chunks if c.page_start > c.page_end or c.page_end > c.total_pages]
    bad_tables = [c for c in tables if not c.content.lstrip().startswith("|")]
    headerless = [c for c in tables if c.content.count("\n") < 2]
    ids = Counter(c.chunk_id for c in chunks)
    untitled = [c for c in chunks if not c.title.strip()]
    no_source = [c for c in chunks if not c.source_record_ids]

    results = [
        check("chunk ids unique", len(ids) == len(chunks), f"{len(chunks) - len(ids)} duplicated"),
        check("no text chunk below the minimum", not undersized, f"{len(undersized)} found"),
        check("no chunk above the maximum", not oversized, f"{len(oversized)} found"),
        check("page ranges valid", not bad_pages, f"{len(bad_pages)} invalid"),
        check("tables still Markdown", not bad_tables, f"{len(bad_tables)} malformed"),
        check("tables keep header and body", not headerless, f"{len(headerless)} truncated"),
        check("every chunk has a title", not untitled, f"{len(untitled)} missing"),
        check("every chunk traces to a record", not no_source, f"{len(no_source)} orphaned"),
        check(
            "page spans located exactly",
            report.get("unlocated_chunks", 0) == 0,
            f"{report.get('unlocated_chunks', 0)} approximate",
        ),
    ]

    if args.samples:
        print("\nsamples:")
        for chunk in (text[: args.samples] + spanning[: args.samples] + tables[: args.samples])[:6]:
            body = chunk.content.replace("\n", " ")[:130]
            print(f"  [{chunk.kind}/{chunk.strategy}] {chunk.citation}")
            print(f"      {body}...")

    print()
    return 0 if all(results) else 1


if __name__ == "__main__":
    sys.exit(main())
