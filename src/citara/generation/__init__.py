"""[7-8] Grounded generation, citations, refusal, provider failover, streaming (34-41).

Exports resolve on first use rather than at import. Importing ``citara.generation.models``
runs this file first, and an eager ``from citara.generation.answerer import Answerer`` here
made that light import load the retriever, the embedder and torch - which kept the
interface's loading screen blank until the heaviest part of startup had finished.
"""

from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from citara.generation.answerer import Answerer
    from citara.generation.citations import build_citations, markers_in, validate
    from citara.generation.models import Citation, GeneratedAnswer
    from citara.generation.prompts import SYSTEM_PROMPT, build_user_prompt, format_evidence
    from citara.generation.providers import GeminiProvider, GroqProvider, available_providers

_EXPORTS = {
    "Answerer": "citara.generation.answerer",
    "build_citations": "citara.generation.citations",
    "markers_in": "citara.generation.citations",
    "validate": "citara.generation.citations",
    "Citation": "citara.generation.models",
    "GeneratedAnswer": "citara.generation.models",
    "SYSTEM_PROMPT": "citara.generation.prompts",
    "build_user_prompt": "citara.generation.prompts",
    "format_evidence": "citara.generation.prompts",
    "GeminiProvider": "citara.generation.providers",
    "GroqProvider": "citara.generation.providers",
    "available_providers": "citara.generation.providers",
}

__all__ = [
    "SYSTEM_PROMPT",
    "Answerer",
    "Citation",
    "GeminiProvider",
    "GeneratedAnswer",
    "GroqProvider",
    "available_providers",
    "build_citations",
    "build_user_prompt",
    "format_evidence",
    "markers_in",
    "validate",
]


def __getattr__(name: str) -> Any:
    if name in _EXPORTS:
        return getattr(import_module(_EXPORTS[name]), name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
