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

from collections.abc import Iterator
from typing import Protocol

from citara.config import Settings
from citara.log import get_logger
from citara.resilience.retry import with_retry

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


class _LangChainProvider:
    """Shared behaviour for the chat providers."""

    name = "unknown"

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.model = ""
        self._client: object | None = None

    def _build(self) -> object:
        raise NotImplementedError

    def _chat(self) -> object:
        if self._client is None:
            self._client = self._build()
        return self._client

    def _messages(self, system: str, user: str) -> list[tuple[str, str]]:
        return [("system", system), ("human", user)]

    def generate(self, system: str, user: str) -> str:
        """Ask the model, waiting out transient failures before giving up (feature 50)."""

        def call() -> str:
            response = self._chat().invoke(self._messages(system, user))  # type: ignore[attr-defined]
            return extract_text(getattr(response, "content", "")).strip()

        return with_retry(call, self.settings.resilience, label=f"{self.name}.generate")

    def stream(self, system: str, user: str) -> Iterator[str]:
        for piece in self._chat().stream(self._messages(system, user)):  # type: ignore[attr-defined]
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
