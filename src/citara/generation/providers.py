"""LLM providers and failover (features 36, 38, 39, 40, 41).

Two independent providers, because a demo on institutional WiFi cannot depend on one endpoint
staying healthy, and free tiers rate-limit exactly when a room full of people is watching.
Groq is not a second Gemini endpoint: different company, different network path, different
quota - which is the entire point of redundancy.

Generation is near-deterministic by configuration. A policy assistant must not be creative:
the same question on Tuesday must produce the answer it produced on Monday, because an officer
may act on it.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator
from itertools import chain
from typing import Protocol

from citara.config import Settings
from citara.log import get_logger
from citara.resilience.budget import COOLDOWNS, RequestBudget
from citara.resilience.retry import classify, with_retry

log = get_logger("generation.provider")


def extract_text(content: object) -> str:
    """Plain text from a chat response.

    LangChain 1.x returns a list of content blocks, so stringifying the response yields a
    list repr with metadata instead of the sentence the model wrote.
    """
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = [
            str(block.get("text", ""))
            for block in content
            if isinstance(block, dict) and block.get("type", "text") == "text"
        ]
        return "".join(parts)
    return str(content)


class Provider(Protocol):
    """A chat model that can answer with a system prompt."""

    name: str
    model: str

    def generate(self, system: str, user: str) -> str: ...
    def stream(self, system: str, user: str) -> Iterator[str]: ...
    def warm_up(self) -> None: ...


class ProviderUnavailable(RuntimeError):
    """Raised without sending anything, for a provider that reported its daily quota spent."""


class _LangChainProvider:
    """Shared behaviour for the chat providers."""

    name = "unknown"

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.model = ""
        self._client: object | None = None
        self.budget = RequestBudget.from_settings(settings)

    def _build(self) -> object:
        raise NotImplementedError

    def _chat(self) -> object:
        if self._client is None:
            self._client = self._build()
        return self._client

    def _messages(self, system: str, user: str) -> list[tuple[str, str]]:
        return [("system", system), ("human", user)]

    def warm_up(self) -> None:
        """Build the client ahead of the first question. Sends nothing, so spends no quota."""
        try:
            self._chat()
        except Exception:
            log.warning("could not prepare provider", extra={"provider": self.name}, exc_info=True)

    @property
    def quota_key(self) -> str:
        """Identity for quota tracking, shared with anything else calling the same model."""
        return f"{self.name}:{self.model}"

    def _call[T](self, attempt: Callable[[], T], label: str) -> T:
        """One logical request: skipped while the quota is spent, retried while transient.

        Each attempt counts itself against the budget as it is sent (feature 48), so retries
        are counted as the requests they are.
        """
        cooling = COOLDOWNS.remaining(self.quota_key)
        if cooling:
            raise ProviderUnavailable(f"{self.name} quota spent; next attempt in {cooling:.0f} s")
        try:
            return with_retry(attempt, self.settings.resilience, label=f"{self.name}.{label}")
        except Exception as error:
            if classify(error).daily_quota:
                COOLDOWNS.mark(self.quota_key, self.settings.resilience.quota_cooldown_s)
            raise

    def generate(self, system: str, user: str) -> str:
        """Ask the model, waiting out transient failures before giving up (feature 50)."""

        def attempt() -> str:
            chat = self._chat()
            self.budget.record()
            response = chat.invoke(self._messages(system, user))  # type: ignore[attr-defined]
            return extract_text(getattr(response, "content", "")).strip()

        return self._call(attempt, "generate")

    def stream(self, system: str, user: str) -> Iterator[str]:
        """Stream the answer, retrying only until the first piece arrives.

        The request goes out on the first read, so that is where a rate limit surfaces - and
        while nothing has been shown, a retry is invisible to the reader. After that, a
        failure propagates: restarting would repeat text already on screen.
        """

        def attempt() -> tuple[object | None, Iterator[object]]:
            chat = self._chat()
            self.budget.record()
            pieces = iter(chat.stream(self._messages(system, user)))  # type: ignore[attr-defined]
            return next(pieces, None), pieces

        first, rest = self._call(attempt, "stream")
        for piece in chain([first] if first is not None else [], rest):
            text = extract_text(getattr(piece, "content", ""))
            if text:
                yield text


class GeminiProvider(_LangChainProvider):
    """Primary provider."""

    name = "gemini"

    def __init__(self, settings: Settings) -> None:
        super().__init__(settings)
        self.model = settings.generation.primary_model

    def _build(self) -> object:
        from langchain_google_genai import ChatGoogleGenerativeAI

        key = self.settings.google_api_key
        if key is None:
            raise RuntimeError("GOOGLE_API_KEY is not configured")
        config = self.settings.generation
        log.info("initialising primary provider", extra={"model": self.model})
        return ChatGoogleGenerativeAI(
            model=self.model,
            google_api_key=key.get_secret_value(),
            temperature=config.temperature,
            max_output_tokens=config.max_output_tokens,
            timeout=config.request_timeout_s,
            # The SDK would otherwise retry up to six times inside every attempt made here,
            # uncounted and unbounded; resilience.retry is the one retry layer.
            max_retries=0,
        )


class GroqProvider(_LangChainProvider):
    """Failover provider: separate company, network path and quota."""

    name = "groq"

    def __init__(self, settings: Settings) -> None:
        super().__init__(settings)
        self.model = settings.generation.fallback_model

    def _build(self) -> object:
        from langchain_groq import ChatGroq

        key = self.settings.groq_api_key
        if key is None:
            raise RuntimeError("GROQ_API_KEY is not configured")
        config = self.settings.generation
        log.info("initialising failover provider", extra={"model": self.model})
        return ChatGroq(
            model=self.model,
            api_key=key,
            temperature=config.temperature,
            max_tokens=config.max_output_tokens,
            timeout=config.request_timeout_s,
            max_retries=0,  # see GeminiProvider: one retry layer, in resilience.retry
        )


def available_providers(settings: Settings) -> list[Provider]:
    """Configured providers, primary first.

    A provider with no key is omitted rather than left to fail at request time, so the
    failover chain reflects what can actually be called.
    """
    providers: list[Provider] = []
    if settings.google_api_key is not None:
        providers.append(GeminiProvider(settings))
    if settings.groq_api_key is not None and settings.resilience.enable_failover:
        providers.append(GroqProvider(settings))
    return providers
