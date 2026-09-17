"""Structured logging for every pipeline stage (feature 73).

Two things beyond stdlib defaults, both of which later phases depend on:

* **Structured records.** Arbitrary fields attach to a log line (``extra={...}``) and survive
  into JSON, so ingestion counts and retrieval scores are queryable rather than prose.
* **Stage timing.** :func:`stage` records ``duration_ms`` for a block of work. The per-answer
  latency display (feature 62) and the p50/p95 benchmarks (feature 71) both read these numbers,
  so timing is instrumented once, here, rather than improvised per call site.

Named ``log`` rather than ``logging`` so it can never shadow the standard library module.
"""

from __future__ import annotations

import json
import logging
import sys
import time
from collections.abc import Generator, MutableMapping
from contextlib import contextmanager
from typing import Any

_CONFIGURED_FLAG = "_citara_handler"

# Attributes present on every LogRecord; anything else was passed as `extra`.
_STANDARD_FIELDS = frozenset(
    {
        "args", "asctime", "created", "exc_info", "exc_text", "filename", "funcName",
        "levelname", "levelno", "lineno", "module", "msecs", "message", "msg", "name",
        "pathname", "process", "processName", "relativeCreated", "stack_info", "taskName",
        "thread", "threadName",
    }
)  # fmt: skip


def _extra_fields(record: logging.LogRecord) -> dict[str, Any]:
    """Fields attached via ``extra=``, excluding stdlib record attributes."""
    return {k: v for k, v in record.__dict__.items() if k not in _STANDARD_FIELDS}


class JsonFormatter(logging.Formatter):
    """One JSON object per line, with extras merged in."""

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "ts": self.formatTime(record, "%Y-%m-%dT%H:%M:%S"),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        payload.update(_extra_fields(record))
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        return json.dumps(payload, default=str, ensure_ascii=False)


class TextFormatter(logging.Formatter):
    """Human-readable line with extras appended as ``key=value`` pairs."""

    def __init__(self) -> None:
        super().__init__(fmt="%(asctime)s %(levelname)-7s %(name)-22s %(message)s")

    def format(self, record: logging.LogRecord) -> str:
        base = super().format(record)
        extras = _extra_fields(record)
        if extras:
            rendered = " ".join(f"{k}={v}" for k, v in extras.items())
            return f"{base} | {rendered}"
        return base


def configure_logging(level: str = "INFO", json_lines: bool = False) -> None:
    """Install the CITARA handler on the root logger.

    Idempotent: Streamlit re-runs the script on every interaction, so repeated calls must not
    stack duplicate handlers. Forces UTF-8 on the stream because the Windows console defaults
    to cp1252 and raises on the extracted-text characters this corpus contains (Phase 0).
    """
    root = logging.getLogger()
    root.setLevel(level)

    for existing in root.handlers:
        if getattr(existing, _CONFIGURED_FLAG, False):
            existing.setLevel(level)
            existing.setFormatter(JsonFormatter() if json_lines else TextFormatter())
            return

    stream = sys.stderr
    reconfigure = getattr(stream, "reconfigure", None)
    if callable(reconfigure):  # pragma: no cover - depends on stream type
        reconfigure(encoding="utf-8", errors="replace")

    handler: logging.Handler = logging.StreamHandler(stream)
    handler.setLevel(level)
    handler.setFormatter(JsonFormatter() if json_lines else TextFormatter())
    setattr(handler, _CONFIGURED_FLAG, True)
    root.addHandler(handler)

    # Third-party libraries in this stack are chatty at INFO.
    for noisy in ("httpx", "httpcore", "chromadb", "sentence_transformers", "urllib3"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


def get_logger(name: str) -> logging.Logger:
    """Logger for a pipeline module, namespaced under ``citara``."""
    return logging.getLogger(name if name.startswith("citara") else f"citara.{name}")


@contextmanager
def stage(
    logger: logging.Logger,
    name: str,
    level: int = logging.INFO,
    **fields: Any,
) -> Generator[MutableMapping[str, Any], None, None]:
    """Time a pipeline stage and log its outcome.

    Yields a mutable mapping; anything put in it is logged with the completion record, so a
    stage can report what it produced::

        with stage(log, "ingest", document=name) as s:
            s["pages"] = 110
            s["chunks"] = 412

    On failure the duration is still recorded, then the exception propagates untouched.
    """
    details: dict[str, Any] = dict(fields)
    logger.debug("%s started", name, extra={"stage": name, **details})
    started = time.perf_counter()
    try:
        yield details
    except Exception:
        elapsed_ms = round((time.perf_counter() - started) * 1000, 1)
        logger.exception(
            "%s failed", name, extra={"stage": name, "duration_ms": elapsed_ms, **details}
        )
        raise
    elapsed_ms = round((time.perf_counter() - started) * 1000, 1)
    logger.log(level, "%s done", name, extra={"stage": name, "duration_ms": elapsed_ms, **details})
