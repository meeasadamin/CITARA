"""Non-identifying query logging (feature 46).

Queries are kept so retrieval can be improved against what people actually ask - the gold set
was written by one person and cannot anticipate everything. Nothing identifying is recorded:
no user, no session, no address. What is stored is the question, what the pipeline did with
it, and how well it went.

The question text itself is retained because it is the whole point of the log, so the
interface must say that queries are recorded, and the disclaimer does.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from citara.log import get_logger

log = get_logger("guardrails.query_log")


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
        "question": question,
        "rewritten": rewritten if rewritten != question else "",
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
