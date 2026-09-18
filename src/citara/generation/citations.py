"""Citation parsing and validation (feature 35).

Every factual claim must carry a marker that resolves to a specific retrieved chunk, and
therefore to a document and page. This module makes that checkable by machine rather than by
reading the answer and hoping.

Two failures are detected:

* **Invalid markers** - the model cited [7] when only five evidence blocks exist. Left alone,
  a reader following that citation finds nothing, which is worse than no citation at all.
* **Uncited claims** - a sentence that asserts something and carries no marker.
"""

from __future__ import annotations

import re

from citara.generation.models import Citation
from citara.retrieval.models import RetrievedChunk

_MARKER = re.compile(r"\[(\d+)\]")
# Sentences that assert nothing factual do not need a source.
_NON_CLAIM = re.compile(
    r"^\s*(here (is|are)|the following|in summary|note that|according to the evidence|"
    r"this (answer|information)|sources?:|based on|however[,]?\s*(the)?)",
    re.IGNORECASE,
)
# Statements *about* the evidence rather than claims drawn from it. A refusal or a caveat
# cannot cite a source, and flagging it as an uncited claim would penalise the system for
# doing the right thing - saying the corpus does not contain something.
_ABOUT_THE_EVIDENCE = re.compile(
    r"(evidence|documents?|corpus|sources?|passages?)\s+"
    r"(provided\s+)?(does|do|did)\s+not\s+(contain|include|state|specify|mention|provide)"
    r"|(does|do)\s+not\s+(contain|include|state|specify|mention|provide)\s+"
    r"(the\s+)?(specific|any|an?)\b"
    r"|(is|are)\s+not\s+(stated|specified|mentioned|given|provided)\s+in\s+the\s+"
    r"(evidence|documents?|corpus)"
    r"|i\s+(could|can)\s?not\s+find",
    re.IGNORECASE,
)
_SENTENCE = re.compile(r"(?<=[.!?])\s+")
# A fragment this short is a heading or a list label, not a claim.
_MIN_CLAIM_CHARS = 25


def build_citations(results: list[RetrievedChunk]) -> list[Citation]:
    """Number the evidence blocks exactly as the prompt presents them."""
    return [
        Citation(
            marker=index,
            citation=result.chunk.citation,
            chunk_id=result.chunk.chunk_id,
            doc_id=result.chunk.doc_id,
            page_start=result.chunk.page_start,
            page_end=result.chunk.page_end,
        )
        for index, result in enumerate(results, start=1)
    ]


def markers_in(text: str) -> list[int]:
    """Every citation marker referenced in *text*, in order of appearance."""
    return [int(match.group(1)) for match in _MARKER.finditer(text)]


def count_uncited_claims(text: str) -> int:
    """Sentences that assert something without citing evidence."""
    uncited = 0
    for sentence in _SENTENCE.split(text.strip()):
        body = sentence.strip()
        if len(body) < _MIN_CLAIM_CHARS or _NON_CLAIM.match(body):
            continue
        if _ABOUT_THE_EVIDENCE.search(body):
            continue
        if not _MARKER.search(body):
            uncited += 1
    return uncited


def validate(text: str, citations: list[Citation]) -> tuple[list[int], int]:
    """Mark which citations the answer used and report its citation defects.

    Returns ``(invalid_markers, uncited_sentences)``.
    """
    valid = {citation.marker for citation in citations}
    used = markers_in(text)
    for citation in citations:
        citation.used = citation.marker in used
    invalid = sorted({marker for marker in used if marker not in valid})
    return invalid, count_uncited_claims(text)
