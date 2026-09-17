"""Command-line entry point: ``uv run python -m citara.indexing``.

Default is incremental: only documents whose source file changed are re-indexed, which is
what makes adding one new advisory a minute of work rather than a full rebuild (feature 22).
"""

from __future__ import annotations

import argparse
import sys

from citara.config import get_settings
from citara.indexing.builder import build_index
from citara.log import configure_logging, get_logger

log = get_logger("indexing.cli")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build the dense and sparse indexes.")
    parser.add_argument(
        "--rebuild", action="store_true", help="discard the existing index and build from scratch"
    )
    parser.add_argument("--json-logs", action="store_true")
    args = parser.parse_args(argv)

    settings = get_settings()
    configure_logging(settings.log_level, args.json_logs)
    settings.ensure_directories()

    data_dir = settings.paths.resolved(settings.paths.data_dir)
    if not (data_dir / "chunks.jsonl").exists():
        log.error(
            "no chunks found; run chunking first",
            extra={"command": "uv run python -m citara.chunking"},
        )
        return 2

    try:
        manifest = build_index(settings, rebuild=args.rebuild)
    except (ValueError, RuntimeError) as error:
        log.error("index build failed", extra={"error": str(error)})
        return 1

    print(
        f"\nIndexed {manifest.chunk_count} chunks from {len(manifest.documents)} documents "
        f"using {manifest.embedding_model} ({manifest.dimension}d)."
    )
    print(f"vector store: {settings.paths.resolved(settings.paths.chroma_dir)}")
    print(f"sparse index: {settings.paths.resolved(settings.paths.bm25_path)}")
    print(f"manifest:     {settings.paths.resolved(settings.paths.index_manifest_path)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
