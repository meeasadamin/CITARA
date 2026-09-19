"""Retry with exponential backoff (feature 50).

Most rate-limit responses are survivable if you wait a moment, and a free tier will
rate-limit exactly when a room full of people is watching. Retrying buys that moment before
the system escalates to the other provider.

What matters more than the backoff is knowing when *not* to retry. A bad API key, a model
that no longer exists, or a malformed request will fail identically on the fifth attempt as
on the first; retrying them wastes the seconds a demo does not have and delays the failover
that would have worked. Only transient conditions are retried.
"""

from __future__ import annotations

import time
from collections.abc import Callable

from citara.config import ResilienceSettings
from citara.log import get_logger

log = get_logger("resilience.retry")

# Substrings that mark a failure as worth waiting out. Matching on message text is crude, but
# each provider SDK raises its own exception types and they change between versions.
_TRANSIENT_MARKERS = (
    "rate limit",
    "rate_limit",
    "429",
    "resource exhausted",
    "resource_exhausted",
    "quota",
    "timeout",
    "timed out",
    "deadline exceeded",
    "unavailable",
    "503",
    "502",
    "504",
    "overloaded",
    "connection reset",
    "connection aborted",
    "temporarily",
)

# Conditions that will not improve by waiting.
_PERMANENT_MARKERS = (
    "api key",
    "api_key",
    "unauthorized",
    "unauthenticated",
    "permission denied",
    "403",
    "401",
    "not found",
    "404",
    "invalid argument",
    "invalid_argument",
    "400",
    "no longer available",
    "does not exist",
    "safety",
    "blocked",
)


def is_transient(error: Exception) -> bool:
    """True when waiting and trying again could plausibly succeed."""
    message = f"{type(error).__name__}: {error}".lower()
    if any(marker in message for marker in _PERMANENT_MARKERS):
        return False
    return any(marker in message for marker in _TRANSIENT_MARKERS)


def with_retry[T](
    operation: Callable[[], T],
    settings: ResilienceSettings,
    *,
    label: str = "request",
    sleep: Callable[[float], None] = time.sleep,
) -> T:
    """Run *operation*, retrying transient failures with exponential backoff.

    The final failure is raised rather than swallowed, so the caller can fail over to another
    provider instead of receiving a silent None.
    """
    delay = settings.backoff_base_s
    last: Exception | None = None

    for attempt in range(settings.max_retries + 1):
        try:
            return operation()
        except Exception as error:
            last = error
            if not is_transient(error) or attempt == settings.max_retries:
                if attempt:
                    log.warning(
                        "giving up after retries",
                        extra={"label": label, "attempts": attempt + 1, "error": str(error)[:160]},
                    )
                raise
            log.info(
                "transient failure; backing off",
                extra={
                    "label": label,
                    "attempt": attempt + 1,
                    "delay_s": round(delay, 2),
                    "error": str(error)[:160],
                },
            )
            sleep(delay)
            delay = min(delay * 2, settings.backoff_max_s)

    raise last if last else RuntimeError("retry loop ended without a result")
