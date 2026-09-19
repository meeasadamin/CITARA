"""Retry with exponential backoff (feature 50).

Most rate-limit responses are survivable if you wait a moment, and a free tier will
rate-limit exactly when a room full of people is watching. Retrying buys that moment before
the system escalates to the other provider.

What matters more than the backoff is knowing when *not* to retry. A bad API key, a model
that no longer exists, or a malformed request will fail identically on the fifth attempt as
on the first; retrying them wastes the seconds a demo does not have and delays the failover
that would have worked. So does a spent *daily* quota, which a few seconds of waiting cannot
refill - it looks like any other 429 and has to be told apart. Only transient conditions are
retried, and never for longer than the retry budget: a question stuck behind a struggling
provider should reach failover or degraded mode in seconds, not minutes.

This is the only retry layer. The provider SDKs retry on their own by default - six attempts
for Gemini - which would multiply every attempt here, hide requests from the budget, and
stretch a failing call into minutes; the providers switch that off.
"""

from __future__ import annotations

import re
import time
from collections.abc import Callable
from dataclasses import dataclass

from citara.config import ResilienceSettings
from citara.log import get_logger

log = get_logger("resilience.retry")

# HTTP statuses worth waiting out: timeout, rate limit, and the server-side 5xx family.
_TRANSIENT_STATUSES = frozenset({408, 429, 500, 502, 503, 504})

# Used only when no status code can be found on the exception. Word boundaries matter: a
# rate-limit body is full of numbers, and "quotaValue: 4000000" must not read as a 400.
_TRANSIENT_TEXT = re.compile(
    r"\b(?:429|408|50[0234])\b|rate.?limit|resource.?exhausted|quota|timed? ?out|timeout"
    r"|deadline exceeded|unavailable|overloaded|connection (?:reset|aborted)|temporarily",
    re.IGNORECASE,
)
_PERMANENT_TEXT = re.compile(
    r"\b(?:400|401|403|404)\b|api.?key|unauthori[sz]ed|unauthenticated|permission denied"
    r"|not found|invalid.?argument|no longer available|does not exist|safety|blocked",
    re.IGNORECASE,
)

# A quota that resets tomorrow. Gemini names it in the quota id
# ("GenerateRequestsPerDayPerProjectPerModel-FreeTier"); Groq says "requests per day (RPD)".
_DAILY_QUOTA = re.compile(r"per ?day|\bRPD\b|\bTPD\b", re.IGNORECASE)

# How long the provider itself says to wait. Gemini: "Please retry in 34.6s." and
# "'retryDelay': '34s'"; Groq: "Please try again in 1m26.4s."
_RETRY_AFTER = (
    re.compile(r"(?:retry|try again) in (?:(\d+)m)?(\d+(?:\.\d+)?)s", re.IGNORECASE),
    re.compile(r"retryDelay['\"]?\s*:\s*['\"]?()(\d+(?:\.\d+)?)s", re.IGNORECASE),
)


@dataclass(frozen=True)
class Failure:
    """What an exception means for whether, and when, to try again."""

    status: int | None
    transient: bool
    daily_quota: bool = False
    retry_after_s: float | None = None


def _status_code(error: BaseException) -> int | None:
    """The HTTP status, from the exception or anything it wraps.

    LangChain wraps the SDK's error ("GoogleInvalidRequestError") and keeps the original as
    its cause, which is where the numeric code lives.
    """
    seen: set[int] = set()
    current: BaseException | None = error
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        for attribute in ("status_code", "code", "http_status"):
            value = getattr(current, attribute, None)
            if isinstance(value, int) and 100 <= value <= 599:
                return value
        current = current.__cause__ or current.__context__
    return None


def _retry_after(message: str) -> float | None:
    for pattern in _RETRY_AFTER:
        match = pattern.search(message)
        if match:
            minutes = float(match.group(1)) if match.group(1) else 0.0
            return minutes * 60 + float(match.group(2))
    return None


def classify(error: BaseException) -> Failure:
    """Decide whether *error* is worth retrying, and what the provider said about waiting."""
    message = f"{type(error).__name__}: {error}"
    if error.__cause__ is not None:
        message += f" | {error.__cause__}"
    status = _status_code(error)
    retry_after = _retry_after(message)

    if status is not None:
        transient = status in _TRANSIENT_STATUSES
    elif _PERMANENT_TEXT.search(message):
        transient = False
    else:
        transient = bool(_TRANSIENT_TEXT.search(message))

    rate_limited = status == 429 or (status is None and transient)
    daily = rate_limited and bool(_DAILY_QUOTA.search(message))
    return Failure(
        status=status,
        transient=transient and not daily,
        daily_quota=daily,
        retry_after_s=retry_after,
    )


def is_transient(error: BaseException) -> bool:
    """True when waiting and trying again could plausibly succeed."""
    return classify(error).transient


def with_retry[T](
    operation: Callable[[], T],
    settings: ResilienceSettings,
    *,
    label: str = "request",
    sleep: Callable[[float], None] = time.sleep,
    clock: Callable[[], float] = time.monotonic,
) -> T:
    """Run *operation*, retrying transient failures with exponential backoff.

    The wait is the longer of the backoff and whatever the provider asked for. If that is
    more than the backoff ceiling, or would overrun the retry budget, the failure is raised at
    once: a provider asking for forty seconds is telling us to use the other one.

    The final failure is raised rather than swallowed, so the caller can fail over to another
    provider instead of receiving a silent None.
    """
    started = clock()
    delay = settings.backoff_base_s

    for attempt in range(settings.max_retries + 1):
        try:
            return operation()
        except Exception as error:
            failure = classify(error)
            wait = max(delay, failure.retry_after_s or 0.0)
            elapsed = clock() - started
            reason = ""
            if not failure.transient:
                reason = "daily quota exhausted" if failure.daily_quota else "not transient"
            elif attempt == settings.max_retries:
                reason = "retries exhausted"
            elif wait > settings.backoff_max_s:
                reason = "provider asked for a longer wait than the backoff ceiling"
            elif elapsed + wait > settings.retry_budget_s:
                reason = "retry budget spent"

            if reason:
                log.warning(
                    "giving up",
                    extra={
                        "label": label,
                        "attempts": attempt + 1,
                        "reason": reason,
                        "status": failure.status,
                        "elapsed_s": round(elapsed, 2),
                        "error": str(error)[:160],
                    },
                )
                raise

            log.info(
                "transient failure; backing off",
                extra={
                    "label": label,
                    "attempt": attempt + 1,
                    "delay_s": round(wait, 2),
                    "status": failure.status,
                    "error": str(error)[:160],
                },
            )
            sleep(wait)
            delay = min(delay * 2, settings.backoff_max_s)

    raise RuntimeError("retry loop ended without a result")  # pragma: no cover
