"""Package the built index for deployment, and print the checksum to pin.

The deployed app clones the repository, which carries no index: `data/` is ignored, and the
source PDFs are not redistributed, so the app cannot rebuild one either. This packs what
retrieval actually needs into one archive to publish as a GitHub Release asset, which
`citara.indexing.fetch` downloads on first run.

Paths inside the archive are relative to the data directory, so it unpacks straight into it.

Run:  uv run python scripts/package_index.py
Then: upload dist/citara-index.tar.gz to a release, and pin the printed sha256 in config.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import tarfile
from pathlib import Path

from citara.config import get_settings

# Everything retrieval opens, and the two manifests the sidebar and health check read.
CONTENTS = (
    "chunks.jsonl",
    "index_manifest.json",
    "corpus_manifest.json",
    "bm25",
    "chroma",
)


def digest(path: Path) -> str:
    sha = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            sha.update(block)
    return sha.hexdigest()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Package the index for deployment.")
    parser.add_argument("--out", default="dist/citara-index.tar.gz")
    args = parser.parse_args(argv)

    settings = get_settings()
    data = settings.paths.resolved(settings.paths.data_dir)
    missing = [name for name in CONTENTS if not (data / name).exists()]
    if missing:
        print(f"nothing to package: {', '.join(missing)} not found in {data}")
        return 1

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(out, "w:gz") as archive:
        for name in CONTENTS:
            archive.add(data / name, arcname=name)

    checksum = digest(out)
    manifest = json.loads((data / "index_manifest.json").read_text(encoding="utf-8"))
    size_mb = out.stat().st_size / 1_048_576

    print(f"wrote {out} ({size_mb:.1f} MB)")
    print(f"  chunks indexed : {manifest.get('chunk_count')}")
    print(f"  documents      : {len(manifest.get('documents', []))}")
    print(f"  built at       : {manifest.get('built_at')}")
    print(f"  sha256         : {checksum}")
    print("\npin it in config.py as index_sha256, and publish the archive as a release asset.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
