"""Request budgeting and the per-session cap (features 48, 52).

Two different problems that look alike.

**The daily budget** protects a shared free tier. It is tracked against the day rather than
the process, because Streamlit restarts an app whenever it wakes and a counter that resets on
restart protects nothing. Crossing it does not raise: the system falls back to serving cited
evidence without generation, which is the same degraded path used when providers fail. An
officer who still gets the right source pages has lost the summary, not the answer.

**The session cap** is abuse protection on a public URL, stopping one visitor from consuming
the quota everyone shares. It lives in memory, because a session is a browser tab and does
not outlive the process meaningfully.

Both surface how much is left *before* the limit is reached, so the interface can warn rather
than simply stop working.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from citara.log import get_logger

log = get_logger("resilience.budget")


@dataclass(frozen=True)
class BudgetState:
    """What remains of today's allowance."""

    day: str
    used: int
    limit: int

    @property
    def remaining(self) -> int:
        return max(0, self.limit - self.used)

    @property
    def exhausted(self) -> bool:
        return self.used >= self.limit

    @property
    def low(self) -> bool:
        """True once the remaining budget is worth warning about."""
        return not self.exhausted and self.remaining <= max(1, self.limit // 10)


class RequestBudget:
    """Counts provider requests against a daily limit, persisted across restarts."""

    def __init__(self, path: Path, daily_limit: int) -> None:
        self.path = path
        self.daily_limit = daily_limit

    @staticmethod
    def _today() -> str:
        return datetime.now(UTC).strftime("%Y-%m-%d")

    def _read(self) -> dict[str, int]:
        if not self.path.exists():
            return {}
        try:
            payload = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            log.warning("usage file unreadable; starting the day's count again")
            return {}
        return {str(day): int(count) for day, count in payload.items()}

    def state(self) -> BudgetState:
        """Today's usage without changing it."""
        day = self._today()
        return BudgetState(day=day, used=self._read().get(day, 0), limit=self.daily_limit)

    def record(self, requests: int = 1) -> BudgetState:
        """Count requests against today, keeping only recent days on disk."""
        day = self._today()
        counts = self._read()
        counts[day] = counts.get(day, 0) + requests
        # Keep a fortnight: enough to see a pattern, small enough to stay trivial.
        recent = dict(sorted(counts.items())[-14:])
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.path.write_text(json.dumps(recent), encoding="utf-8")
        except OSError:
            log.warning("could not persist request usage", exc_info=True)

        state = BudgetState(day=day, used=recent[day], limit=self.daily_limit)
        if state.exhausted:
            log.warning("daily request budget exhausted", extra={"used": state.used})
        elif state.low:
            log.info("daily request budget running low", extra={"remaining": state.remaining})
        return state


class SessionLimiter:
    """Per-session query cap, protecting a shared quota from a single visitor (feature 52)."""

    def __init__(self, cap: int) -> None:
        self.cap = cap
        self._counts: dict[str, int] = {}

    def used(self, session_id: str) -> int:
        return self._counts.get(session_id, 0)

    def remaining(self, session_id: str) -> int:
        return max(0, self.cap - self.used(session_id))

    def allows(self, session_id: str) -> bool:
        return self.used(session_id) < self.cap

    def record(self, session_id: str) -> int:
        """Count one query for this session and return the new total."""
        self._counts[session_id] = self.used(session_id) + 1
        if self._counts[session_id] == self.cap:
            log.info("session reached its query cap", extra={"cap": self.cap})
        return self._counts[session_id]

    def reset(self, session_id: str | None = None) -> None:
        if session_id is None:
            self._counts.clear()
        else:
            self._counts.pop(session_id, None)
