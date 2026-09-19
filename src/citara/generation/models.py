"""Generated answers and their provenance (features 35, 37, 39, 41, 51)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

from citara.retrieval.models import RetrievedChunk

AnswerMode = Literal["generated", "refused", "degraded", "out_of_scope", "blocked"]

# Why a turn ended without a generated answer - so the interface can say so specifically,
# instead of inferring it from message text.
Reason = Literal[
    "",
    "no_evidence",
    "too_long",
    "session_cap",
    "injection",
    "out_of_scope",
    "budget_exhausted",
    "providers_unavailable",
]


@dataclass
class Citation:
    """One evidence block, as the answer refers to it."""

    marker: int
    citation: str
    chunk_id: str
    doc_id: str
    page_start: int
    page_end: int
    used: bool = False


@dataclass
class GeneratedAnswer:
    """An answer plus everything needed to verify and explain it."""

    text: str
    mode: AnswerMode = "generated"
    citations: list[Citation] = field(default_factory=list)
    provider: str = ""
    model: str = ""
    evidence: list[RetrievedChunk] = field(default_factory=list)
    invalid_markers: list[int] = field(default_factory=list)
    uncited_sentences: int = 0
    retrieval_ms: float = 0.0
    generation_ms: float = 0.0
    failover_used: bool = False
    error: str = ""
    cached: bool = False
    screening: str = "none"
    injection_flags: list[str] = field(default_factory=list)
    reason: Reason = ""

    @property
    def refused(self) -> bool:
        return self.mode in {"refused", "out_of_scope", "blocked"}

    @property
    def total_ms(self) -> float:
        return round(self.retrieval_ms + self.generation_ms, 1)

    @property
    def cited(self) -> list[Citation]:
        """Only the evidence the answer actually referenced."""
        return [c for c in self.citations if c.used]

    @property
    def is_fully_cited(self) -> bool:
        """True when every marker resolves and no factual sentence is left uncited.

        An uncited claim is treated as a defect rather than a stylistic lapse: the whole
        promise of the system is that a claim can be checked against its source.
        """
        return not self.invalid_markers and self.uncited_sentences == 0
