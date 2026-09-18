"""LLM-backed history-aware query rewriting (feature 30).

The heuristic rewriter carries the previous turn's content words into the question, which is
free and never fails but produces clumsy queries: "What about Sindh? recovery needs total
2022 floods" retrieves a table of contents. A model rewrites the same turn as "What were the
total recovery needs for Sindh after the 2022 floods?", which is what the retriever needed.

Every failure path falls back to the heuristic rather than raising. A rewriting outage must
degrade retrieval, not break it - the same reasoning that puts a deterministic splitter
behind the semantic chunker.
"""

from __future__ import annotations

from citara.config import GenerationSettings, Settings, get_settings
from citara.log import get_logger
from citara.retrieval.query import HeuristicRewriter, looks_like_follow_up

log = get_logger("retrieval.rewriter")

_PROMPT = """Rewrite the user's latest question as a standalone search query.

Rules:
- Resolve pronouns and references using the conversation, so the query makes sense alone.
- Keep the user's intent and any named places, documents, years or figures.
- Do not answer the question, explain, or add information that is not implied.
- Reply with the rewritten question only, on one line.

Conversation so far:
{history}

Latest question: {question}

Standalone question:"""

# A rewrite longer than this is the model explaining rather than rewriting.
_MAX_REWRITE_CHARS = 300


def _extract_text(content: object) -> str:
    """Pull plain text out of a chat response.

    LangChain 1.x returns a list of content blocks rather than a string, so stringifying the
    response yields the repr of a list - complete with block metadata - instead of the
    sentence the model wrote.
    """
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = [
            str(block.get("text", ""))
            for block in content
            if isinstance(block, dict) and block.get("type", "text") == "text"
        ]
        return " ".join(part for part in parts if part)
    return str(content)


class LLMRewriter:
    """Condenses conversation history into a standalone query using the primary provider."""

    def __init__(self, settings: Settings | None = None) -> None:
        self.settings: Settings = settings or get_settings()
        self.generation: GenerationSettings = self.settings.generation
        self._fallback = HeuristicRewriter()
        self._client: object | None = None
        self._unavailable = False

    def _model(self) -> object | None:
        """Lazily build the chat client; None when no key is configured."""
        if self._unavailable:
            return None
        if self._client is None:
            key = self.settings.google_api_key
            if key is None:
                self._unavailable = True
                log.info("no provider key; using heuristic query rewriting")
                return None
            try:
                from langchain_google_genai import ChatGoogleGenerativeAI

                self._client = ChatGoogleGenerativeAI(
                    model=self.generation.primary_model,
                    google_api_key=key.get_secret_value(),
                    temperature=0.0,
                    max_output_tokens=128,
                    timeout=self.generation.request_timeout_s,
                )
            except Exception:
                self._unavailable = True
                log.exception("could not initialise the rewriting model")
                return None
        return self._client

    def rewrite(self, question: str, history: list[str]) -> str:
        """Return a standalone query, falling back to the heuristic on any failure."""
        if not history or not looks_like_follow_up(question):
            return question

        model = self._model()
        if model is None:
            return self._fallback.rewrite(question, history)

        prompt = _PROMPT.format(
            history="\n".join(f"- {turn}" for turn in history), question=question
        )
        try:
            response = model.invoke(prompt)  # type: ignore[attr-defined]
            text = _extract_text(getattr(response, "content", "")).strip().strip('"')
        except Exception:
            log.warning("query rewriting failed; using the heuristic", exc_info=True)
            return self._fallback.rewrite(question, history)

        first_line = text.splitlines()[0].strip() if text else ""
        if not first_line or len(first_line) > _MAX_REWRITE_CHARS:
            return self._fallback.rewrite(question, history)

        log.info("query rewritten by model", extra={"original": question, "rewritten": first_line})
        return first_line
