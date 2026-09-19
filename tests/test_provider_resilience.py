"""Provider-facing resilience: error classification, retry timing, one retry layer on the
wire, streaming retries and quota cooldowns (features 48, 50).

Error messages here are shaped on real responses: the wrapped-exception layout is what a live
bad-key call to Gemini produced, and the quota bodies follow Gemini's and Groq's formats.
"""

from __future__ import annotations

import json
import threading
from collections.abc import Iterator
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from types import SimpleNamespace

import pytest

from citara.config import ResilienceSettings, Settings
from citara.generation.providers import (
    GeminiProvider,
    GroqProvider,
    ProviderUnavailable,
    _LangChainProvider,
)
from citara.resilience.budget import COOLDOWNS, QuotaCooldowns, RequestBudget
from citara.resilience.retry import classify, is_transient, with_retry
from citara.retrieval.llm_rewriter import LLMRewriter


def retry_settings(**overrides: object) -> ResilienceSettings:
    base = {"max_retries": 2, "backoff_base_s": 0.01, "backoff_max_s": 0.05}
    return ResilienceSettings(**{**base, **overrides})  # type: ignore[arg-type]


def fast_settings(max_retries: int = 3) -> Settings:
    return Settings(  # type: ignore[call-arg]
        _env_file=None,
        resilience=ResilienceSettings(
            max_retries=max_retries, backoff_base_s=0.001, backoff_max_s=0.05
        ),
    )


def used(settings: Settings) -> int:
    return RequestBudget.from_settings(settings).state().used


# --- real error shapes ---------------------------------------------------------------


class SdkError(Exception):
    """Shaped like google.genai's ClientError: the status lives on ``code``."""

    def __init__(self, code: int, message: str) -> None:
        super().__init__(f"{code} {message}")
        self.code = code


class WrappedError(Exception):
    """Shaped like LangChain's GoogleInvalidRequestError, which keeps the SDK error as cause."""


def wrapped(code: int, message: str) -> WrappedError:
    try:
        try:
            raise SdkError(code, message)
        except SdkError as inner:
            raise WrappedError(f"Error calling model 'gemini': {inner}") from inner
    except WrappedError as outer:
        return outer


GEMINI_PER_MINUTE = (
    "RESOURCE_EXHAUSTED. You exceeded your current quota. Quota exceeded for metric: "
    "generate_content_free_tier_input_token_count, limit: 4000000. "
    "'quotaId': 'GenerateContentInputTokensPerModelPerMinute-FreeTier', "
    "'quotaValue': '4000000'. Please retry in 3.2s."
)
GEMINI_PER_DAY = (
    "RESOURCE_EXHAUSTED. You exceeded your current quota. "
    "'quotaId': 'GenerateRequestsPerDayPerProjectPerModel-FreeTier', 'quotaValue': '20'. "
    "'retryDelay': '34s'"
)
GROQ_PER_DAY = (
    "Rate limit reached for model `openai/gpt-oss-120b` on requests per day (RPD): "
    "Limit 1000, Used 1000, Requested 1. Please try again in 1m26.4s."
)


def test_the_status_code_is_read_through_the_langchain_wrapper() -> None:
    """Regression from a live bad-key call: the code sits on the wrapped cause."""
    failure = classify(wrapped(400, "INVALID_ARGUMENT. API key not valid."))
    assert failure.status == 400
    assert failure.transient is False


def test_numbers_in_a_rate_limit_body_are_not_status_codes() -> None:
    """Regression: substring matching read 'quotaValue: 4000000' as a 400 and gave up."""
    assert is_transient(RuntimeError(f"rate limit hit. {GEMINI_PER_MINUTE}")) is True
    failure = classify(wrapped(429, GEMINI_PER_MINUTE))
    assert failure.transient is True
    assert failure.daily_quota is False
    assert failure.retry_after_s == pytest.approx(3.2)


def test_a_spent_daily_quota_is_not_worth_retrying() -> None:
    """It looks like any other 429, and no amount of waiting today will refill it."""
    failure = classify(wrapped(429, GEMINI_PER_DAY))
    assert failure.transient is False
    assert failure.daily_quota is True
    assert failure.retry_after_s == pytest.approx(34.0)


def test_groq_daily_quota_and_its_wait_are_recognised() -> None:
    error = RuntimeError(GROQ_PER_DAY)
    error.status_code = 429  # type: ignore[attr-defined]
    failure = classify(error)
    assert failure.daily_quota is True
    assert failure.retry_after_s == pytest.approx(86.4)


def test_server_errors_are_classified_by_status() -> None:
    assert classify(wrapped(503, "UNAVAILABLE")).transient is True
    assert classify(wrapped(404, "model not found")).transient is False


# --- retry timing --------------------------------------------------------------------


def test_the_provider_requested_wait_is_honoured() -> None:
    delays: list[float] = []
    calls: list[int] = []

    def rate_limited_once() -> str:
        calls.append(1)
        if len(calls) == 1:
            raise RuntimeError("429 rate limit. Please retry in 0.03s.")
        return "answer"

    assert with_retry(rate_limited_once, retry_settings(), sleep=delays.append) == "answer"
    assert delays == [0.03]  # longer than the 0.01 backoff, because the provider asked


def test_a_wait_beyond_the_backoff_ceiling_fails_over_at_once() -> None:
    """A provider asking for forty seconds is saying: use the other one."""
    delays: list[float] = []
    calls: list[int] = []

    def rate_limited() -> str:
        calls.append(1)
        raise RuntimeError("429 rate limit. Please retry in 40s.")

    with pytest.raises(RuntimeError):
        with_retry(rate_limited, retry_settings(), sleep=delays.append)
    assert calls == [1]
    assert delays == []


def test_retrying_stops_when_the_budget_is_spent() -> None:
    """A slow, failing provider must hand over to failover in seconds, not minutes."""
    now = [0.0]
    calls: list[int] = []

    def slow_failure() -> str:
        calls.append(1)
        now[0] += 10.0  # each attempt takes ten seconds to fail
        raise RuntimeError("503 service unavailable")

    with pytest.raises(RuntimeError):
        with_retry(
            slow_failure,
            retry_settings(max_retries=10, retry_budget_s=25.0),
            sleep=lambda _: None,
            clock=lambda: now[0],
        )
    assert len(calls) == 3  # at 10 s and 20 s it retries; at 30 s it would overrun 25 s


def test_a_daily_quota_is_not_retried() -> None:
    calls: list[int] = []

    def spent() -> str:
        calls.append(1)
        raise wrapped(429, GEMINI_PER_DAY)

    with pytest.raises(WrappedError):
        with_retry(spent, retry_settings(), sleep=lambda _: None)
    assert calls == [1]


# --- one retry layer, measured on the wire -------------------------------------------


@pytest.fixture
def rate_limited_endpoint() -> Iterator[tuple[str, list[str]]]:
    """A local HTTP endpoint that answers every request with a per-minute 429."""
    hits: list[str] = []
    body = json.dumps(
        {
            "error": {
                "code": 429,
                "message": "Resource exhausted. Please retry in 0.01s.",
                "status": "RESOURCE_EXHAUSTED",
                "type": "requests",
            }
        }
    ).encode()

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self) -> None:
            hits.append(self.path)
            self.rfile.read(int(self.headers.get("content-length", 0)))
            self.send_response(429)
            self.send_header("content-type", "application/json")
            self.send_header("content-length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *args: object) -> None: ...

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{server.server_address[1]}", hits
    server.shutdown()
    server.server_close()


def point_gemini_at(url: str, monkeypatch: pytest.MonkeyPatch) -> None:
    """Build clients exactly as the code does, but aimed at the local endpoint."""
    import langchain_google_genai

    real = langchain_google_genai.ChatGoogleGenerativeAI

    def aimed(**kwargs: object) -> object:
        return real(**kwargs, base_url=url)

    monkeypatch.setattr(langchain_google_genai, "ChatGoogleGenerativeAI", aimed)
    monkeypatch.setenv("GOOGLE_API_KEY", "test-key")


def test_gemini_sends_one_request_per_counted_attempt(
    rate_limited_endpoint: tuple[str, list[str]], monkeypatch: pytest.MonkeyPatch
) -> None:
    """Regression: the SDK retried up to six times inside every attempt, uncounted.

    Three attempts must mean three requests on the wire and three in the budget.
    """
    url, hits = rate_limited_endpoint
    point_gemini_at(url, monkeypatch)
    settings = fast_settings(max_retries=2)

    with pytest.raises(Exception, match="429"):
        GeminiProvider(settings).generate("system", "user")

    assert len(hits) == 3
    assert used(settings) == 3


def test_groq_sends_one_request_per_counted_attempt(
    rate_limited_endpoint: tuple[str, list[str]], monkeypatch: pytest.MonkeyPatch
) -> None:
    import langchain_groq

    url, hits = rate_limited_endpoint
    real = langchain_groq.ChatGroq

    def aimed(**kwargs: object) -> object:
        return real(**kwargs, groq_api_base=url)

    monkeypatch.setattr(langchain_groq, "ChatGroq", aimed)
    monkeypatch.setenv("GROQ_API_KEY", "test-key")
    settings = fast_settings(max_retries=2)

    with pytest.raises(Exception, match="429"):
        GroqProvider(settings).generate("system", "user")

    assert len(hits) == 3
    assert used(settings) == 3


def test_the_rewriter_sends_one_request_then_falls_back(
    rate_limited_endpoint: tuple[str, list[str]], monkeypatch: pytest.MonkeyPatch
) -> None:
    url, hits = rate_limited_endpoint
    point_gemini_at(url, monkeypatch)

    rewritten = LLMRewriter(fast_settings()).rewrite(
        "What about Sindh?", ["What were the total flood damages?"]
    )

    assert len(hits) == 1
    assert "Sindh" in rewritten


# --- streaming retries (feature 50 on the path the interface uses) -------------------


class FlakyStreamChat:
    """Rate-limits the first *failures* stream requests, then streams *pieces*."""

    def __init__(self, failures: int, pieces: list[str], fail_after: int | None = None) -> None:
        self.failures = failures
        self.pieces = pieces
        self.fail_after = fail_after
        self.requests = 0

    def stream(self, messages: object) -> Iterator[SimpleNamespace]:
        self.requests += 1
        if self.requests <= self.failures:
            raise RuntimeError("429 rate limit")
        for index, piece in enumerate(self.pieces):
            if index == self.fail_after:
                raise RuntimeError("503 connection reset mid-stream")
            yield SimpleNamespace(content=piece)


def provider_with(chat: object, settings: Settings) -> _LangChainProvider:
    class Provider(_LangChainProvider):
        name = "fake"

        def _build(self) -> object:
            return chat

    return Provider(settings)


def test_a_rate_limited_stream_is_retried_before_anything_is_shown() -> None:
    """Regression: stream() had no retry, so one 429 on the UI path skipped to failover."""
    settings = fast_settings()
    chat = FlakyStreamChat(failures=2, pieces=["Evacuate ", "now [1]."])

    assert "".join(provider_with(chat, settings).stream("system", "user")) == "Evacuate now [1]."
    assert chat.requests == 3
    assert used(settings) == 3


def test_a_stream_that_fails_part_way_is_not_restarted() -> None:
    """Text is already on screen; a restart would repeat it."""
    chat = FlakyStreamChat(failures=0, pieces=["Evacuate ", "now ", "[1]."], fail_after=1)
    shown: list[str] = []

    with pytest.raises(RuntimeError, match="mid-stream"):
        for piece in provider_with(chat, fast_settings()).stream("system", "user"):
            shown.append(piece)

    assert shown == ["Evacuate "]
    assert chat.requests == 1


# --- quota cooldowns -----------------------------------------------------------------


class DailyQuotaChat:
    def __init__(self) -> None:
        self.invocations = 0

    def invoke(self, messages: object) -> SimpleNamespace:
        self.invocations += 1
        raise wrapped(429, GEMINI_PER_DAY)


def gemini_with(chat: object, settings: Settings) -> GeminiProvider:
    class Stubbed(GeminiProvider):
        def _build(self) -> object:
            return chat

    return Stubbed(settings)


def test_a_provider_out_of_daily_quota_is_skipped_without_a_request() -> None:
    """Every later question would otherwise pay a round trip to rediscover it."""
    settings = fast_settings()
    chat = DailyQuotaChat()
    provider = gemini_with(chat, settings)

    with pytest.raises(WrappedError):
        provider.generate("system", "user")
    with pytest.raises(ProviderUnavailable):
        provider.generate("system", "user")

    assert chat.invocations == 1
    assert used(settings) == 1


def test_the_rewriter_respects_a_quota_the_provider_discovered() -> None:
    """One model, one quota: the rewriter should not spend a request finding out again."""
    settings = fast_settings()
    with pytest.raises(WrappedError):
        gemini_with(DailyQuotaChat(), settings).generate("system", "user")

    class Unused:
        invocations = 0

        def invoke(self, prompt: object) -> SimpleNamespace:
            self.invocations += 1
            return SimpleNamespace(content="should not be asked")

    rewriter = LLMRewriter(settings)
    client = Unused()
    rewriter._client = client
    rewritten = rewriter.rewrite("What about Sindh?", ["What were the total flood damages?"])

    assert client.invocations == 0
    assert "Sindh" in rewritten


def test_cooldowns_lapse() -> None:
    now = [100.0]
    cooldowns = QuotaCooldowns(clock=lambda: now[0])
    cooldowns.mark("gemini:model", 60)
    assert cooldowns.remaining("gemini:model") == pytest.approx(60)
    now[0] += 61
    assert cooldowns.remaining("gemini:model") == 0.0
    assert COOLDOWNS.remaining("gemini:model") == 0.0  # the shared registry is untouched
