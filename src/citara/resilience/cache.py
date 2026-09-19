"""Response cache (feature 49).

A demo asks the same handful of showcase questions repeatedly, and every repeat is a free-tier
request spent on an answer already produced. Caching turns the second asking into a local
lookup, which both protects the quota and makes the demo feel instant.

The cache lives on disk rather than in memory because Streamlit Community Cloud sleeps idle
applications: an in-process cache is empty exactly when the panel opens the link.

Two rules keep it honest. The key includes the configuration fingerprint, so changing the
relevance floor or the model cannot serve an answer produced under the old settings. And
near-identical questions share an entry - "What were the total damages?" and "what were the
total damages" are the same question - but only through whitespace, case and punctuation
normalisation, never through anything that could change meaning.
"""

from __future__ import annotations

import hashlib
import json
import re
import time
from pathlib import Path
from typing import Any

from citara.log import get_logger
from citara.resilience.files import write_atomic

log = get_logger("resilience.cache")

_PUNCTUATION = re.compile(r"[^\w\s]")
_WHITESPACE = re.compile(r"\s+")


def normalise_question(question: str) -> str:
    """Fold trivial variations so repeats of the same question share one entry."""
    lowered = _PUNCTUATION.sub(" ", question.lower())
    return _WHITESPACE.sub(" ", lowered).strip()


def cache_key(question: str, fingerprint: str, **context: Any) -> str:
    """Key for one question under one configuration.

    The fingerprint is part of the key rather than a stored field: a changed configuration
    must miss the cache, not quietly serve an answer the current settings would not produce.
    """
    extras = "|".join(f"{name}={context[name]}" for name in sorted(context) if context[name])
    raw = f"{normalise_question(question)}|{fingerprint}|{extras}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:32]


class AnswerCache:
    """Disk-backed cache of rendered answers."""

    def __init__(self, directory: Path, ttl_s: int, max_entries: int) -> None:
        self.directory = directory
        self.ttl_s = ttl_s
        self.max_entries = max_entries

    def _path(self, key: str) -> Path:
        return self.directory / f"{key}.json"

    @staticmethod
    def _discard(path: Path) -> None:
        """Remove an entry, tolerating a file another thread still holds open on Windows."""
        try:
            path.unlink(missing_ok=True)
        except OSError:
            log.debug("could not remove cache entry", extra={"path": path.name})

    def get(self, key: str) -> dict[str, Any] | None:
        """Return a stored payload, or None when absent, expired or unreadable."""
        path = self._path(key)
        if not path.exists():
            return None
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            log.warning("discarding unreadable cache entry", extra={"key": key})
            self._discard(path)
            return None

        age = time.time() - float(payload.get("stored_at", 0))
        if age > self.ttl_s:
            self._discard(path)
            return None

        log.info("cache hit", extra={"key": key, "age_s": round(age)})
        return dict(payload.get("value", {}))

    def set(self, key: str, value: dict[str, Any]) -> None:
        """Store a payload, failing quietly.

        A cache problem must never break an answer the user already has.
        """
        try:
            write_atomic(
                self._path(key),
                json.dumps({"stored_at": time.time(), "value": value}, ensure_ascii=False),
            )
            self._evict()
        except OSError:
            log.warning("could not write the cache entry", exc_info=True)

    def _evict(self) -> None:
        """Keep the cache bounded, discarding the least recently written entries first."""
        entries = sorted(self.directory.glob("*.json"), key=lambda p: p.stat().st_mtime)
        for path in entries[: max(0, len(entries) - self.max_entries)]:
            self._discard(path)

    def clear(self) -> int:
        """Remove every entry, returning how many were removed."""
        removed = 0
        for path in self.directory.glob("*.json"):
            self._discard(path)
            removed += 1
        return removed
