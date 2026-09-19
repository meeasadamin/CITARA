"""Tests for retry, caching, budgeting and the session cap (features 48-52)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from citara.config import ResilienceSettings
from citara.resilience.budget import RequestBudget, SessionLimiter
from citara.resilience.cache import AnswerCache, cache_key, normalise_question
from citara.resilience.retry import is_transient, with_retry


def settings(**overrides: object) -> ResilienceSettings:
    base = {"max_retries": 2, "backoff_base_s": 0.01, "backoff_max_s": 0.05}
    return ResilienceSettings(**{**base, **overrides})  # type: ignore[arg-type]


# --- retry (feature 50) --------------------------------------------------------------


@pytest.mark.parametrize(
    "message",
    [
        "429 Too Many Requests",
        "Resource exhausted: quota exceeded",
        "Deadline exceeded",
        "503 Service Unavailable",
        "The model is overloaded, please try again",
        "Connection reset by peer",
    ],
)
def test_transient_failures_are_worth_retrying(message: str) -> None:
    assert is_transient(RuntimeError(message)) is True


@pytest.mark.parametrize(
    "message",
    [
        "401 Unauthorized: invalid api key",
        "403 permission denied",
        "404 model not found",
        "400 invalid argument",
        "This model is no longer available to new users",
    ],
)
def test_permanent_failures_are_not_retried(message: str) -> None:
    """Waiting will not fix a bad key, and retrying delays the failover that would work."""
    assert is_transient(RuntimeError(message)) is False


def test_retry_succeeds_after_a_transient_failure() -> None:
    attempts: list[int] = []
    delays: list[float] = []

    def flaky() -> str:
        attempts.append(1)
        if len(attempts) < 3:
            raise RuntimeError("429 rate limit")
        return "answer"

    result = with_retry(flaky, settings(), sleep=delays.append)
    assert result == "answer"
    assert len(attempts) == 3
    assert delays == [0.01, 0.02]  # exponential, and capped by backoff_max_s


def test_backoff_is_capped() -> None:
    delays: list[float] = []

    def always_fails() -> str:
        raise RuntimeError("503 unavailable")

    with pytest.raises(RuntimeError):
        with_retry(
            always_fails,
            settings(max_retries=5, backoff_base_s=0.02, backoff_max_s=0.05),
            sleep=delays.append,
        )
    assert max(delays) <= 0.05


def test_permanent_failure_fails_immediately() -> None:
    """No delay at all: the seconds matter when a panel is watching."""
    delays: list[float] = []
    calls: list[int] = []

    def bad_key() -> str:
        calls.append(1)
        raise RuntimeError("401 unauthorized: invalid api key")

    with pytest.raises(RuntimeError):
        with_retry(bad_key, settings(), sleep=delays.append)
    assert calls == [1]
    assert delays == []


def test_the_final_error_propagates_for_failover() -> None:
    def always_fails() -> str:
        raise RuntimeError("429 rate limit")

    with pytest.raises(RuntimeError, match="rate limit"):
        with_retry(always_fails, settings(max_retries=1), sleep=lambda _: None)


# --- cache (feature 49) --------------------------------------------------------------


def test_trivial_variations_share_a_cache_entry() -> None:
    """A demo asks the same question with different punctuation and capitalisation."""
    assert normalise_question("What were the TOTAL damages?") == normalise_question(
        "what were the total damages"
    )


def test_different_questions_do_not_collide() -> None:
    assert cache_key("flood damages", "fp1") != cache_key("heatwave thresholds", "fp1")


def test_configuration_change_misses_the_cache() -> None:
    """A changed relevance floor must not serve an answer the new settings would not give."""
    assert cache_key("flood damages", "fp1") != cache_key("flood damages", "fp2")


def test_document_filters_are_part_of_the_key() -> None:
    assert cache_key("q", "fp", docs="ndrp") != cache_key("q", "fp", docs="pdna")


def test_cache_roundtrip(tmp_path: Path) -> None:
    cache = AnswerCache(tmp_path, ttl_s=60, max_entries=10)
    assert cache.get("missing") is None
    cache.set("k", {"text": "answer [1]"})
    assert cache.get("k") == {"text": "answer [1]"}


def test_expired_entries_are_discarded(tmp_path: Path) -> None:
    cache = AnswerCache(tmp_path, ttl_s=0, max_entries=10)
    cache.set("k", {"text": "stale"})
    assert cache.get("k") is None


def test_cache_is_bounded(tmp_path: Path) -> None:
    cache = AnswerCache(tmp_path, ttl_s=60, max_entries=3)
    for index in range(6):
        cache.set(f"k{index}", {"text": str(index)})
    assert len(list(tmp_path.glob("*.json"))) <= 3


def test_corrupt_entries_are_dropped_not_raised(tmp_path: Path) -> None:
    cache = AnswerCache(tmp_path, ttl_s=60, max_entries=10)
    (tmp_path / "bad.json").write_text("{not json", encoding="utf-8")
    assert cache.get("bad") is None


def test_cache_write_failure_is_survivable(tmp_path: Path) -> None:
    """A cache problem must never break an answer the user already has."""
    blocked = tmp_path / "cache"
    blocked.mkdir()
    (blocked / "k.json").mkdir()  # a directory where the entry file belongs
    AnswerCache(blocked, ttl_s=60, max_entries=10).set("k", {"text": "x"})


# --- budget and session cap (features 48, 52) ----------------------------------------


def test_budget_counts_and_reports_what_remains(tmp_path: Path) -> None:
    budget = RequestBudget(tmp_path / "usage.json", daily_limit=10)
    assert budget.state().remaining == 10

    for _ in range(4):
        state = budget.record()
    assert state.used == 4
    assert state.remaining == 6
    assert state.exhausted is False


def test_budget_survives_a_restart(tmp_path: Path) -> None:
    """Streamlit restarts the app when it wakes; a counter that resets protects nothing."""
    path = tmp_path / "usage.json"
    RequestBudget(path, daily_limit=10).record(3)
    assert RequestBudget(path, daily_limit=10).state().used == 3


def test_budget_warns_before_it_stops(tmp_path: Path) -> None:
    budget = RequestBudget(tmp_path / "usage.json", daily_limit=10)
    budget.record(9)
    assert budget.state().low is True
    budget.record(1)
    assert budget.state().exhausted is True


def test_budget_keeps_only_recent_days(tmp_path: Path) -> None:
    path = tmp_path / "usage.json"
    path.write_text(json.dumps({f"2026-01-{day:02d}": 1 for day in range(1, 21)}), encoding="utf-8")
    RequestBudget(path, daily_limit=10).record()
    assert len(json.loads(path.read_text(encoding="utf-8"))) <= 14


def test_unreadable_usage_file_does_not_break_answering(tmp_path: Path) -> None:
    path = tmp_path / "usage.json"
    path.write_text("{corrupt", encoding="utf-8")
    assert RequestBudget(path, daily_limit=10).state().used == 0


def test_session_cap_limits_one_visitor(tmp_path: Path) -> None:
    limiter = SessionLimiter(cap=3)
    for _ in range(3):
        assert limiter.allows("a") is True
        limiter.record("a")
    assert limiter.allows("a") is False
    assert limiter.remaining("a") == 0


def test_sessions_are_counted_separately() -> None:
    limiter = SessionLimiter(cap=2)
    limiter.record("a")
    limiter.record("a")
    assert limiter.allows("a") is False
    assert limiter.allows("b") is True


def test_session_can_be_reset() -> None:
    limiter = SessionLimiter(cap=1)
    limiter.record("a")
    limiter.reset("a")
    assert limiter.allows("a") is True
