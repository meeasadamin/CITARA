"""Persistence for ingestion output (feature 11).

Records are written as JSON Lines so a corpus build can be inspected, diffed and re-read by
the chunking stage without re-parsing every PDF.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path

from citara.ingestion.models import CorpusManifest, PageRecord


def write_records(records: list[PageRecord], path: Path) -> None:
    """Write records as JSON Lines, creating parent directories as needed."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(record.model_dump_json() + "\n")


def read_records(path: Path) -> Iterator[PageRecord]:
    """Stream records back from disk."""
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield PageRecord.model_validate_json(line)


def write_manifest(manifest: CorpusManifest, path: Path) -> None:
    """Write the manifest as indented JSON, with totals included for humans."""
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = manifest.model_dump(mode="json")
    payload["totals"] = manifest.summary()
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def read_manifest(path: Path) -> CorpusManifest:
    """Read a manifest written by :func:`write_manifest`."""
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload.pop("totals", None)
    return CorpusManifest.model_validate(payload)
