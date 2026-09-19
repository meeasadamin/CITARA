"""Non-identifying query logging (feature 46).

Queries are kept so retrieval can be improved against what people actually ask - the gold set
was written by one person and cannot anticipate everything. Nothing identifying is recorded:
no user, no session, no address.

The question text itself is retained, because it is the whole point of the log, but people
put themselves into questions. "When will relief reach my area, my number is 0300-..." is a
realistic thing to type into a disaster-response tool, and a log that keeps it has captured
identifying information no matter what the surrounding fields contain. Contact details are
therefore redacted before the line is written, not afterwards.
"""

from __future__ import annotations

import json
import re
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from citara.log import get_logger

log = get_logger("guardrails.query_log")

# Patterns are deliberately broad: over-redacting a question costs a little analytical value,
# while under-redacting stores someone's phone number forever.
_REDACTIONS: tuple[tuple[re.Pattern[str], str], ...] = (
    (re.compile(r"[\w.+-]+@[\w-]+\.[\w.]+"), "[email]"),
    # Pakistani mobile and landline shapes, with or without country code and separators.
    (re.compile(r"(?<!\w)(?:\+?92[\s-]?|0)\d{2,4}[\s-]?\d{6,8}(?!\w)"), "[phone]"),
    # CNIC: 13 digits, usually 5-7-1.
    (re.compile(r"(?<!\w)\d{5}[\s-]?\d{7}[\s-]?\d(?!\w)"), "[id-number]"),
    # Any other long digit run that could be an account or identity number. Four-digit years
    # and the figures this corpus is full of stay intact.
    (re.compile(r"(?<!\w)\d{9,}(?!\w)"), "[number]"),
)

# A question longer than this is a paste, not a question, and is more likely to carry
# incidental personal content.
_MAX_LOGGED_CHARS = 500


def redact(text: str) -> str:
    """Remove contact and identity details from text bound for the log."""
    for pattern, replacement in _REDACTIONS:
        text = pattern.sub(replacement, text)
    if len(text) > _MAX_LOGGED_CHARS:
        text = text[:_MAX_LOGGED_CHARS] + "...[truncated]"
    return text


def record_query(
    path: Path,
    question: str,
    *,
    rewritten: str = "",
    mode: str = "",
    refused: bool = False,
    refusal_reason: str = "",
    best_score: float | None = None,
    evidence: int = 0,
    documents: list[str] | None = None,
    latency_ms: float = 0.0,
    screening: str = "none",
) -> None:
    """Append one line to the query log, failing quietly.

    A logging problem must never break an answer: the log is for improving the system, and
    the officer in front of it needs the answer more than the project needs the record.
    """
    entry: dict[str, Any] = {
        "ts": datetime.now(UTC).isoformat(timespec="seconds"),
        "question": redact(question),
        "rewritten": redact(rewritten) if rewritten and rewritten != question else "",
        "mode": mode,
        "refused": refused,
        "refusal_reason": refusal_reason,
        "best_score": round(best_score, 4) if best_score is not None else None,
        "evidence": evidence,
        "documents": sorted(set(documents or [])),
        "latency_ms": round(latency_ms, 1),
        "screening": screening,
    }
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(entry, ensure_ascii=False) + "\n")
    except OSError:
        log.warning("could not write the query log", exc_info=True)


def read_queries(path: Path) -> list[dict[str, Any]]:
    """Read the query log, skipping any line that cannot be parsed."""
    if not path.exists():
        return []
    entries: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            entries.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return entries
