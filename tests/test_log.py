"""Tests for structured logging and stage timing (feature 73)."""

from __future__ import annotations

import json
import logging
from typing import Any

import pytest

from citara.log import JsonFormatter, TextFormatter, configure_logging, get_logger, stage


def _record(**extra: Any) -> logging.LogRecord:
    record = logging.LogRecord(
        name="citara.test",
        level=logging.INFO,
        pathname=__file__,
        lineno=1,
        msg="hello %s",
        args=("world",),
        exc_info=None,
    )
    for key, value in extra.items():
        setattr(record, key, value)
    return record


def test_json_formatter_includes_extras() -> None:
    payload = json.loads(JsonFormatter().format(_record(stage="ingest", pages=110)))
    assert payload["message"] == "hello world"
    assert payload["level"] == "INFO"
    assert payload["logger"] == "citara.test"
    assert payload["stage"] == "ingest"
    assert payload["pages"] == 110


def test_text_formatter_appends_extras() -> None:
    line = TextFormatter().format(_record(stage="retrieval", duration_ms=12.5))
    assert "hello world" in line
    assert "stage=retrieval" in line
    assert "duration_ms=12.5" in line


def test_get_logger_namespaces_under_citara() -> None:
    assert get_logger("ingestion").name == "citara.ingestion"
    assert get_logger("citara.retrieval").name == "citara.retrieval"


def test_configure_logging_is_idempotent() -> None:
    """Streamlit re-runs the script on every interaction; handlers must not stack."""
    root = logging.getLogger()
    original = list(root.handlers)
    try:
        root.handlers = []
        configure_logging("INFO")
        configure_logging("DEBUG")
        configure_logging("INFO", json_lines=True)
        assert len(root.handlers) == 1
    finally:
        root.handlers = original


def test_stage_records_duration(caplog: pytest.LogCaptureFixture) -> None:
    logger = get_logger("test.stage")
    with (
        caplog.at_level(logging.INFO, logger=logger.name),
        stage(logger, "retrieval", query_id="q1") as details,
    ):
        details["candidates"] = 24

    done = [r for r in caplog.records if r.getMessage() == "retrieval done"]
    assert len(done) == 1
    assert done[0].stage == "retrieval"  # type: ignore[attr-defined]
    assert done[0].candidates == 24  # type: ignore[attr-defined]
    assert done[0].duration_ms >= 0  # type: ignore[attr-defined]
    assert done[0].query_id == "q1"  # type: ignore[attr-defined]


def test_stage_logs_failure_and_reraises(caplog: pytest.LogCaptureFixture) -> None:
    logger = get_logger("test.stage.fail")
    with (
        caplog.at_level(logging.ERROR, logger=logger.name),
        pytest.raises(ValueError, match="boom"),
        stage(logger, "generation"),
    ):
        raise ValueError("boom")

    failed = [r for r in caplog.records if r.getMessage() == "generation failed"]
    assert len(failed) == 1
    assert failed[0].levelno == logging.ERROR
    assert failed[0].duration_ms >= 0  # type: ignore[attr-defined]
    assert failed[0].exc_info is not None
