"""Tests for retry, caching, budgeting and the session cap (features 48-52)."""

from __future__ import annotations

import json
import threading
from collections.abc import Iterator
from pathlib import Path
from types import SimpleNamespace

import pytest

from citara.config import ResilienceSettings, Settings
from citara.generation.providers import _LangChainProvider
from citara.resilience.budget import RequestBudget, SessionLimiter
from citara.resilience.cache import AnswerCache, cache_key, normalise_question
from citara.resilience.files import write_atomic
from citara.resilience.retry import is_transient, with_retry
from citara.retrieval.llm_rewriter import LLMRewriter


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


def test_concurrent_requests_are_all_counted(tmp_path: Path) -> None:
    """Streamlit serves each visitor on a thread; an interleaved read-modify-write drops counts."""
    path = tmp_path / "usage.json"

    def burst() -> None:
        budget = RequestBudget(path, daily_limit=10_000)
        for _ in range(25):
            budget.record()

    threads = [threading.Thread(target=burst) for _ in range(8)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert RequestBudget(path, daily_limit=10_000).state().used == 200


# --- atomic writes -------------------------------------------------------------------


def test_atomic_write_replaces_and_leaves_nothing_behind(tmp_path: Path) -> None:
    target = tmp_path / "usage.json"
    write_atomic(target, '{"a": 1}')
    write_atomic(target, '{"a": 2}')
    assert json.loads(target.read_text(encoding="utf-8")) == {"a": 2}
    assert [p.name for p in tmp_path.iterdir()] == ["usage.json"]


def test_a_failed_write_keeps_the_previous_file(tmp_path: Path) -> None:
    """A write that dies part-way must not leave a torn file that reads as a zero count."""
    target = tmp_path / "usage.json"
    write_atomic(target, '{"2026-01-01": 7}')

    class Unserialisable:
        def __str__(self) -> str:
            raise RuntimeError("disk full")

    with pytest.raises(TypeError):
        write_atomic(target, Unserialisable())  # type: ignore[arg-type]
    assert json.loads(target.read_text(encoding="utf-8")) == {"2026-01-01": 7}
    assert [p.name for p in tmp_path.iterdir()] == ["usage.json"]


# --- counting where requests leave the process (feature 48) --------------------------


class FakeChat:
    """Stands in for a LangChain chat model; fails transiently a set number of times."""

    def __init__(self, failures: int = 0, text: str = "answer") -> None:
        self.failures = failures
        self.text = text
        self.invocations = 0

    def invoke(self, messages: object) -> SimpleNamespace:
        self.invocations += 1
        if self.invocations <= self.failures:
            raise RuntimeError("429 rate limit")
        return SimpleNamespace(content=self.text)

    def stream(self, messages: object) -> Iterator[SimpleNamespace]:
        self.invocations += 1
        yield SimpleNamespace(content=self.text)


def fast_settings() -> Settings:
    return Settings(  # type: ignore[call-arg]
        _env_file=None,
        resilience=ResilienceSettings(max_retries=3, backoff_base_s=0.001, backoff_max_s=0.001),
    )


def provider_with(chat: FakeChat | None, settings: Settings) -> _LangChainProvider:
    class Provider(_LangChainProvider):
        name = "fake"

        def _build(self) -> object:
            if chat is None:
                raise RuntimeError("GOOGLE_API_KEY is not configured")
            return chat

    return Provider(settings)


def test_every_attempt_is_counted_including_retries() -> None:
    """Two rate-limited attempts and a success are three requests, not one answer."""
    settings = fast_settings()
    chat = FakeChat(failures=2)
    assert provider_with(chat, settings).generate("system", "user") == "answer"
    assert chat.invocations == 3
    assert RequestBudget.from_settings(settings).state().used == 3


def test_a_request_that_never_left_is_not_counted() -> None:
    settings = fast_settings()
    with pytest.raises(RuntimeError, match="not configured"):
        provider_with(None, settings).generate("system", "user")
    assert RequestBudget.from_settings(settings).state().used == 0


def test_a_stream_counts_as_one_request() -> None:
    settings = fast_settings()
    assert "".join(provider_with(FakeChat(), settings).stream("system", "user")) == "answer"
    assert RequestBudget.from_settings(settings).state().used == 1


def test_provider_warm_up_spends_nothing() -> None:
    settings = fast_settings()
    chat = FakeChat()
    provider = provider_with(chat, settings)
    provider.warm_up()
    assert chat.invocations == 0
    assert RequestBudget.from_settings(settings).state().used == 0


def test_query_rewriting_counts_against_the_budget() -> None:
    settings = fast_settings()
    rewriter = LLMRewriter(settings)
    rewriter._client = FakeChat(text="What were the total flood damages in Sindh?")

    rewritten = rewriter.rewrite("What about Sindh?", ["What were the total flood damages?"])

    assert rewritten == "What were the total flood damages in Sindh?"
    assert RequestBudget.from_settings(settings).state().used == 1


def test_query_rewriting_stops_spending_once_the_budget_is_gone() -> None:
    """The free heuristic is the better use of nothing."""
    settings = fast_settings()
    budget = RequestBudget.from_settings(settings)
    budget.record(settings.resilience.daily_request_budget)
    chat = FakeChat(text="should not be asked")
    rewriter = LLMRewriter(settings)
    rewriter._client = chat

    rewritten = rewriter.rewrite("What about Sindh?", ["What were the total flood damages?"])

    assert chat.invocations == 0
    assert rewritten != "should not be asked"
    assert "Sindh" in rewritten
