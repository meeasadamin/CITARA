"""Gold question set (feature 66).

This is the artifact every later number depends on. Hit Rate, MRR, faithfulness and the
ablation table are all measured against it, so two design decisions are load-bearing.

**Gold is keyed on pages, not chunk ids.** A chunk id changes whenever a chunking parameter
changes, which would silently invalidate the whole set. A document and page number is what a
human can verify against the PDF, and it survives re-chunking.

**The category mix is the experiment design.** A set made only of conceptual paraphrases would
show hybrid retrieval performing identically to dense retrieval - not because fusion is
useless, but because the questions never exercised the case it exists for. Identifier
lookups, table figures and procedures have to be represented deliberately, and so do
questions the corpus cannot answer, since refusal is a behaviour under test rather than an
absence of one.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, model_validator

QuestionCategory = Literal[
    "conceptual",  # paraphrase with no shared keywords - the dense retriever's case
    "identifier",  # district, section number, figure - the sparse retriever's case
    "table",  # the answer lives in a table, not prose
    "procedural",  # a sequence that must not be returned half-complete
    "multi_hop",  # needs evidence from more than one place
    "follow_up",  # only makes sense given the preceding turn
    "unanswerable",  # in-domain and plausible, but absent from the corpus
    "out_of_scope",  # not disaster management at all
]

VerificationStatus = Literal["drafted", "verified", "rejected"]


class GoldSource(BaseModel):
    """Where the answer can be read, in a form a human can check against the PDF."""

    model_config = {"frozen": True}

    doc_id: str
    page_start: int = Field(ge=1)
    page_end: int = Field(ge=1)
    quote: str = Field(
        default="",
        description="Short verbatim extract supporting the answer, for verification.",
    )

    @model_validator(mode="after")
    def _ordered(self) -> GoldSource:
        if self.page_end < self.page_start:
            raise ValueError("page_end must not precede page_start")
        return self

    def covers(self, doc_id: str, page_start: int, page_end: int) -> bool:
        """True when a retrieved chunk overlaps this gold location."""
        return self.doc_id == doc_id and page_start <= self.page_end and page_end >= self.page_start


class GoldQuestion(BaseModel):
    """One evaluation item."""

    model_config = {"frozen": True}

    id: str
    question: str
    category: QuestionCategory
    answerable: bool = True
    expected_answer: str = Field(
        default="",
        description="What a correct answer must convey. Not a string to match literally.",
    )
    sources: list[GoldSource] = Field(default_factory=list)
    context: list[str] = Field(
        default_factory=list,
        description="Preceding turns, for follow-up questions that are meaningless alone.",
    )
    status: VerificationStatus = "drafted"
    notes: str = ""

    @model_validator(mode="after")
    def _consistent(self) -> GoldQuestion:
        if self.answerable and not self.sources:
            raise ValueError(f"{self.id}: an answerable question needs at least one source")
        if not self.answerable and self.sources:
            raise ValueError(f"{self.id}: an unanswerable question must not cite sources")
        if not self.answerable and self.expected_answer:
            raise ValueError(
                f"{self.id}: an unanswerable question expects a refusal, not an answer"
            )
        if self.category == "follow_up" and not self.context:
            raise ValueError(f"{self.id}: a follow-up question needs its preceding turn")
        return self


class GoldSet(BaseModel):
    """The curated set, plus what it is for."""

    version: int = 1
    description: str = ""
    questions: list[GoldQuestion] = Field(default_factory=list)

    @model_validator(mode="after")
    def _unique_ids(self) -> GoldSet:
        seen = {q.id for q in self.questions}
        if len(seen) != len(self.questions):
            raise ValueError("question ids must be unique")
        return self

    def category_counts(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for question in self.questions:
            counts[question.category] = counts.get(question.category, 0) + 1
        return counts

    def answerable(self) -> list[GoldQuestion]:
        return [q for q in self.questions if q.answerable]

    def refusal_cases(self) -> list[GoldQuestion]:
        """Questions where the correct behaviour is to refuse (features 37, 45)."""
        return [q for q in self.questions if not q.answerable]

    def verified(self) -> list[GoldQuestion]:
        return [q for q in self.questions if q.status == "verified"]
