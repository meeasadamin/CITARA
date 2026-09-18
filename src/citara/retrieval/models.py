"""Retrieval results and their provenance (features 24-29).

Every result carries where it came from - dense rank, sparse rank, fusion score, reranker
score - so a retrieval can be explained after the fact rather than being a black box that
emitted five chunks.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from citara.chunking.models import Chunk


@dataclass
class RetrievedChunk:
    """One candidate, annotated with how it was found."""

    chunk: Chunk
    dense_rank: int | None = None
    dense_score: float | None = None
    sparse_rank: int | None = None
    sparse_score: float | None = None
    fusion_score: float = 0.0
    rerank_score: float | None = None

    @property
    def chunk_id(self) -> str:
        return self.chunk.chunk_id

    @property
    def score(self) -> float:
        """The score this chunk is finally ranked by."""
        return self.rerank_score if self.rerank_score is not None else self.fusion_score

    @property
    def found_by(self) -> str:
        """Which retriever surfaced it - the evidence for hybrid retrieval being worth it."""
        if self.dense_rank is not None and self.sparse_rank is not None:
            return "both"
        if self.dense_rank is not None:
            return "dense"
        if self.sparse_rank is not None:
            return "sparse"
        return "none"


@dataclass
class RetrievalOutcome:
    """Everything one retrieval produced, including why it refused."""

    query: str
    rewritten_query: str
    results: list[RetrievedChunk] = field(default_factory=list)
    candidates_considered: int = 0
    dense_hits: int = 0
    sparse_hits: int = 0
    refused: bool = False
    refusal_reason: str = ""
    best_score: float | None = None
    retrieval_ms: float = 0.0
    rerank_ms: float = 0.0

    @property
    def has_evidence(self) -> bool:
        return bool(self.results) and not self.refused

    @property
    def citations(self) -> list[str]:
        return [r.chunk.citation for r in self.results]

    def evidence_strength(self, high: float, moderate: float) -> str:
        """High / Moderate / Low from the reranker score (feature 56).

        Deliberately a band, not a percentage: an invented confidence number is a lie with a
        decimal point.
        """
        if self.best_score is None or not self.results:
            return "None"
        if self.best_score >= high:
            return "High"
        if self.best_score >= moderate:
            return "Moderate"
        return "Low"
