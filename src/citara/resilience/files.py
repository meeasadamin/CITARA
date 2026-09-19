"""Crash-safe small-file writes shared by the cache and the request budget."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path


def write_atomic(path: Path, text: str) -> None:
    """Replace *path* with *text* so that no reader ever sees half a file.

    Streamlit serves each visitor on its own thread, so a reader can arrive mid-write. For
    the usage counter that is not cosmetic: a torn file reads as corrupt, a corrupt file reads
    as zero, and the next write would then reset the day's count - silently disabling the
    budget the counter exists to enforce.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, temporary = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    try:
        with os.fdopen(handle, "w", encoding="utf-8") as stream:
            stream.write(text)
        os.replace(temporary, path)
    except BaseException:
        Path(temporary).unlink(missing_ok=True)
        raise
