"""Fetch the published index when a deployment starts without one.

Streamlit Community Cloud clones the repository, and the repository carries no index: `data/`
is ignored, and the source PDFs are not redistributed, so the app cannot rebuild one either.
The built index is published as a release asset instead, and downloaded once per container on
first start - about 26 MB, unpacked into the data directory.

Two rules make that safe to do automatically. The archive's checksum is pinned in
configuration, so a corrupted download or a substituted asset is refused rather than indexed;
retrieval would otherwise answer from whatever arrived. And extraction uses tarfile's "data"
filter, which rejects members that point outside the destination - an archive is untrusted
input, and a path-traversal entry would otherwise write anywhere the process can reach.
"""

from __future__ import annotations

import hashlib
import shutil
import tarfile
import tempfile
import urllib.error
import urllib.request
from collections.abc import Callable
from pathlib import Path

from citara.config import Settings
from citara.log import get_logger, stage

log = get_logger("indexing.fetch")

# What retrieval opens. Any of these missing means there is no usable index.
REQUIRED = ("chunks.jsonl", "index_manifest.json", "bm25", "chroma")

_TIMEOUT_S = 120
_BLOCK = 1024 * 256


class IndexFetchError(RuntimeError):
    """The index could not be fetched, with a message fit to show a user."""


def index_present(settings: Settings) -> bool:
    """True when the data directory already holds a usable index."""
    data = settings.paths.resolved(settings.paths.data_dir)
    return all((data / name).exists() for name in REQUIRED)


def _download(url: str, destination: Path, on_progress: Callable[[int, int], None] | None) -> None:
    request = urllib.request.Request(url, headers={"User-Agent": "citara"})
    try:
        with urllib.request.urlopen(request, timeout=_TIMEOUT_S) as response:
            total = int(response.headers.get("content-length") or 0)
            received = 0
            with destination.open("wb") as handle:
                while block := response.read(_BLOCK):
                    handle.write(block)
                    received += len(block)
                    if on_progress:
                        on_progress(received, total)
    except (urllib.error.URLError, OSError, ValueError) as error:
        raise IndexFetchError(
            f"The download did not complete ({type(error).__name__}). Check the network and "
            f"that the asset is still published, then reboot the app."
        ) from error


def _verify(archive: Path, expected: str | None) -> None:
    if not expected:
        log.warning("no checksum configured; the downloaded index cannot be verified")
        return
    sha = hashlib.sha256()
    with archive.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            sha.update(block)
    actual = sha.hexdigest()
    if actual != expected:
        raise IndexFetchError(
            "The downloaded index does not match its published checksum, so it was discarded "
            "rather than searched. Re-publish the archive or update index_sha256."
        )


def _extract(archive: Path, data: Path) -> None:
    data.mkdir(parents=True, exist_ok=True)
    try:
        with tarfile.open(archive, "r:gz") as tar:
            # "data" refuses members that escape the destination or carry unsafe attributes.
            tar.extractall(path=data, filter="data")
    except (tarfile.TarError, OSError, ValueError) as error:
        raise IndexFetchError(
            f"The downloaded index could not be unpacked ({type(error).__name__}). It may be "
            f"truncated; reboot the app to fetch it again."
        ) from error


def ensure_index(settings: Settings, on_progress: Callable[[int, int], None] | None = None) -> bool:
    """Download and unpack the published index if the data directory has none.

    Returns True when an index was fetched, False when one was already there or no source is
    configured. Raises IndexFetchError with a message the interface can show as it is.
    """
    if index_present(settings):
        return False
    url = settings.index_url
    if not url:
        log.info("no index and no index_url configured; the health check will report it")
        return False

    data = settings.paths.resolved(settings.paths.data_dir)
    with stage(log, "fetch_index", url=url), tempfile.TemporaryDirectory() as workspace:
        archive = Path(workspace) / "index.tar.gz"
        _download(url, archive, on_progress)
        _verify(archive, settings.index_sha256)
        _extract(archive, data)

    if not index_present(settings):
        missing = [name for name in REQUIRED if not (data / name).exists()]
        # Leave nothing half-unpacked: a partial index reads as a build that went wrong.
        for name in REQUIRED:
            target = data / name
            if target.is_dir():
                shutil.rmtree(target, ignore_errors=True)
            elif target.exists():
                target.unlink(missing_ok=True)
        raise IndexFetchError(
            f"The downloaded archive is missing {', '.join(missing)}. Re-package the index "
            f"with scripts/package_index.py and publish it again."
        )
    log.info("index fetched", extra={"data_dir": str(data)})
    return True
