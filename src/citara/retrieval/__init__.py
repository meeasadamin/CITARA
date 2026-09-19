"""[5-6] Query prep, hybrid retrieval, RRF fusion, reranking, relevance floor (features 24-33).

Exports resolve on first use rather than at import, so ``citara.retrieval.models`` stays a
light import: an eager re-export of the retriever here loaded torch, ChromaDB and the model
clients for anything that only needed the result dataclasses (see ``citara.generation``).
"""

from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from citara.retrieval.fusion import reciprocal_rank_fusion
    from citara.retrieval.llm_rewriter import LLMRewriter
    from citara.retrieval.models import RetrievalOutcome, RetrievedChunk
    from citara.retrieval.query import (
        ACRONYMS,
        HeuristicRewriter,
        expand_acronyms,
        looks_like_follow_up,
        prepare,
    )
    from citara.retrieval.reranker import CrossEncoderReranker
    from citara.retrieval.retriever import HybridRetriever

_EXPORTS = {
    "reciprocal_rank_fusion": "citara.retrieval.fusion",
    "LLMRewriter": "citara.retrieval.llm_rewriter",
    "RetrievalOutcome": "citara.retrieval.models",
    "RetrievedChunk": "citara.retrieval.models",
    "ACRONYMS": "citara.retrieval.query",
    "HeuristicRewriter": "citara.retrieval.query",
    "expand_acronyms": "citara.retrieval.query",
    "looks_like_follow_up": "citara.retrieval.query",
    "prepare": "citara.retrieval.query",
    "CrossEncoderReranker": "citara.retrieval.reranker",
    "HybridRetriever": "citara.retrieval.retriever",
}

__all__ = [
    "ACRONYMS",
    "CrossEncoderReranker",
    "HeuristicRewriter",
    "HybridRetriever",
    "LLMRewriter",
    "RetrievalOutcome",
    "RetrievedChunk",
    "expand_acronyms",
    "looks_like_follow_up",
    "prepare",
    "reciprocal_rank_fusion",
]


def __getattr__(name: str) -> Any:
    if name in _EXPORTS:
        return getattr(import_module(_EXPORTS[name]), name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
