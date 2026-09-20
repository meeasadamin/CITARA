"""The Streamlit interface (features 47, 53-65).

Layout and wiring only. What an answer looks like is decided in ``presenters``, what the
corpus contains in ``corpus``, and what can go wrong at startup in ``health`` - each testable
without a browser. This module arranges them and talks to the Answerer.

Streamlit reruns this script top to bottom on every interaction, so state lives in
``st.session_state`` and anything expensive - the models, the index - is loaded once per
process through ``st.cache_resource``.
"""

from __future__ import annotations

import html
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING

import streamlit as st

from citara.config import Settings, get_settings
from citara.generation.models import GeneratedAnswer
from citara.indexing import fetch
from citara.log import configure_logging, get_logger
from citara.ui import health, logo, presenters
from citara.ui.corpus import Corpus, load_corpus
from citara.ui.styles import CSS

if TYPE_CHECKING:
    from citara.generation.answerer import Answerer

log = get_logger("ui")

SUBTITLE = "Citation-grounded answers from NDMA's published disaster-management documents"
ANY_YEAR = "Any year"
FAVICON = Path(__file__).resolve().parents[3] / "assets" / "favicon.png"

# What each kind of turn is called above its rule.
_NOTICE_LABEL = {
    "refusal": "Refused",
    "degraded": "Degraded",
    "blocked": "Declined",
    "scope": "Out of scope",
    "limit": "Limit reached",
}

# Curated for a ninety-second demo (feature 58) and chosen by running candidates live, not by
# guessing: each of the first three returned High evidence and cited answers on 2026-09-20.
# They show, in order, two documents disagreeing (35,000 against 30,000 dead) with each figure
# cited rather than one silently chosen; figures read out of a PDNA damage table; and the early
# warning chain. The last is deliberately unanswerable - the corpus has heatwave plans but no
# Gwadar-specific threshold - because showing the refusal unprompted beats hoping it is asked.
# Rejected on the same run: agriculture damage (the table holding the figure was not
# retrieved) and district control-room preparation (answered only in part).
STARTERS: tuple[str, ...] = (
    "How many people died in the Quetta earthquake?",
    "How much damage did the water resources and irrigation sector suffer in 2022?",
    "How does NDMA disseminate early warnings to communities?",
    "Which temperature threshold triggers a district-level heatwave alert specifically for Gwadar?",
)

_LOADING = (
    "Loading the document index and the two search models. After the app has been idle this "
    "takes up to a minute, because the hosting service puts sleeping apps to disk; every "
    "question after it is answered in seconds."
)

_FETCHING = (
    "This deployment starts without the document index, so it is being downloaded once. "
    "About 26 MB; it stays until the app is rebooted."
)

_ANSWER_FAILED = (
    "Something went wrong while answering, and the error has been logged. Please try again; "
    "if it persists, rephrase the question or narrow it with the document filter."
)


@st.cache_resource(show_spinner=False)
def load_answerer() -> Answerer:
    """One answerer per process, with models loaded before the first question (feature 64).

    The import lives here, behind the loading state, on purpose. Importing the answerer pulls
    in torch, sentence-transformers and ChromaDB; done at the top of this module, that ran
    before Streamlit drew anything, and a cold start showed a blank page for a minute - the
    exact failure feature 64 exists to prevent.
    """
    from citara.generation.answerer import Answerer

    answerer = Answerer()
    answerer.warm_up()
    return answerer


@st.cache_resource(show_spinner=False)
def fetch_index(_settings: Settings, on_progress: object = None) -> bool:
    """Download the published index once per process, if this deployment has none."""
    return fetch.ensure_index(_settings, on_progress if callable(on_progress) else None)


@st.cache_resource(show_spinner=False)
def corpus_boundary() -> Corpus:
    return load_corpus(get_settings())


def check_health(settings: Settings) -> list[health.Problem]:
    return health.check(settings)


# -- entry point ------------------------------------------------------------------------


def main() -> None:
    settings = get_settings()
    configure_logging(settings.log_level)
    st.set_page_config(
        page_title=settings.ui.page_title,
        page_icon=str(FAVICON) if FAVICON.is_file() else ":material/menu_book:",
        layout="centered",
        initial_sidebar_state="auto",
    )
    st.markdown(CSS, unsafe_allow_html=True)
    _header(settings)

    if not _bootstrap_index(settings):
        st.stop()

    problems = check_health(settings)
    for problem in problems:
        if problem.severity != "info":
            _show_problem(problem)
    if any(problem.blocking for problem in problems):
        st.stop()

    answerer = _start()
    if answerer is None:
        st.stop()
        return

    corpus = corpus_boundary()
    state = st.session_state
    if "session_id" not in state:
        state.session_id = uuid.uuid4().hex
    if "turns" not in state:
        state.turns = []

    for turn in state.turns:
        _render_turn(turn, settings)

    pending = state.pop("pending", None)
    if not state.turns and pending is None:
        _empty_state()

    scope = _scope_text(corpus, list(state.get("doc_filter") or []), _selected_year(state))
    if scope:
        st.markdown(
            f'<div class="citara-scope">Searching only: {html.escape(scope)}</div>',
            unsafe_allow_html=True,
        )

    typed = st.chat_input(
        "Ask about NDMA plans, advisories and guidelines",
        max_chars=settings.guardrails.max_query_chars,
    )
    question = (pending or typed or "").strip()
    if question:
        _ask(question, answerer, settings, scope)

    # Rendered last so the transcript and the remaining-question count include this turn.
    _sidebar(settings, corpus, answerer, problems)


# -- page furniture ---------------------------------------------------------------------


def _header(settings: Settings) -> None:
    """Brand bar and the prototype disclaimer, visible on every screen size (feature 47)."""
    st.markdown(
        f'<div class="citara-header">{logo.lockup(SUBTITLE)}</div>'
        f'<div class="citara-disclaimer">{html.escape(settings.guardrails.prototype_disclaimer)}'
        "</div>",
        unsafe_allow_html=True,
    )


def _label(text: str, first: bool = False, count: str = "") -> None:
    """A small-caps section label over a hairline rule."""
    suffix = f'<span class="count">{html.escape(count)}</span>' if count else ""
    st.markdown(
        f'<span class="section-label{" first" if first else ""}">{html.escape(text)}{suffix}'
        "</span>",
        unsafe_allow_html=True,
    )


def _question(text: str) -> None:
    st.markdown(
        f'<div class="citara-question">{presenters.safe_markdown(text)}</div>',
        unsafe_allow_html=True,
    )


def _show_problem(problem: health.Problem) -> None:
    """A specific, actionable message instead of a stack trace (feature 65)."""
    body = f"**{problem.title}.** {problem.detail}"
    if problem.severity == "error":
        st.error(body, icon=":material/error:")
    else:
        st.warning(body, icon=":material/warning:")


def _bootstrap_index(settings: Settings) -> bool:
    """Fetch the published index before anything decides it is missing (feature 65).

    False means the app cannot continue: the download failed and the reason is on screen.
    """
    if fetch.index_present(settings) or not settings.index_url:
        return True

    with st.status("Fetching the document index", expanded=True) as status:
        st.write(_FETCHING)
        progress = st.empty()

        def report(received: int, total: int) -> None:
            megabytes = received / 1_048_576
            if total:
                progress.markdown(
                    f'<div class="citara-searching">{megabytes:.0f} MB of '
                    f"{total / 1_048_576:.0f} MB</div>",
                    unsafe_allow_html=True,
                )

        try:
            fetch_index(settings, report)
        except fetch.IndexFetchError as error:
            log.exception("index fetch failed")
            status.update(label="Index download failed", state="error", expanded=True)
            st.error(
                f"**The document index could not be downloaded.** {error}", icon=":material/error:"
            )
            return False
        progress.empty()
        status.update(label="Index ready", state="complete", expanded=False)
    return True


def _start() -> Answerer | None:
    """Load the answerer behind an honest loading state (feature 64)."""
    if st.session_state.get("ready"):
        return load_answerer()
    slot = st.empty()
    with slot.status("Starting CITARA", expanded=True) as status:
        st.write(_LOADING)
        try:
            answerer = load_answerer()
        except Exception as error:
            log.exception("startup failed")
            status.update(label="Startup failed", state="error", expanded=True)
            st.error(
                f"**The search models could not be loaded** ({type(error).__name__}). On a "
                "fresh deployment the model download may have been interrupted: reboot the app "
                "to retry. Locally, run `uv sync` and check the logs.",
                icon=":material/error:",
            )
            return None
    # Gone once loaded rather than left as a "Ready" stub: on a phone that stub pushed the
    # starter questions below the fold and scrolled the brand bar off the top.
    slot.empty()
    st.session_state.ready = True
    return answerer


def _queue(question: str) -> None:
    st.session_state.pending = question


def _clear() -> None:
    st.session_state.turns = []


def _empty_state() -> None:
    st.markdown(
        '<div class="citara-intro">Ask an operational question about Pakistan\'s disaster '
        "management plans, advisories and assessments. Every answer cites the document and "
        "page it comes from, and the assistant says so plainly when the documents do not "
        "contain the answer.</div>",
        unsafe_allow_html=True,
    )
    st.caption("Suggested questions. The last one shows how the assistant refuses.")
    for index, question in enumerate(STARTERS):
        st.button(
            question,
            key=f"starter-{index}",
            on_click=_queue,
            args=(question,),
            width="stretch",
        )


def _selected_year(state: object) -> int | None:
    """The year filter as a number, or None while it reads 'Any year'."""
    value = st.session_state.get("year_filter")
    return int(value) if isinstance(value, str) and value.isdigit() else None


def _scope_text(corpus: Corpus, doc_ids: list[str], year: int | None) -> str:
    parts = []
    if doc_ids:
        titles = {d.doc_id: d.title for d in corpus.documents}
        parts.append(", ".join(titles.get(doc_id, doc_id) for doc_id in doc_ids))
    if year:
        parts.append(f"published {year}")
    return "; ".join(parts)


# -- a question and its answer ----------------------------------------------------------


def _ask(question: str, answerer: Answerer, settings: Settings, scope: str) -> None:
    """Stream an answer, then replace the stream with the finished, validated answer."""
    state = st.session_state
    doc_ids = list(state.get("doc_filter") or []) or None
    year = _selected_year(state)
    history = [turn.question for turn in state.turns]

    with st.chat_message("user"):
        _label("Question", first=True)
        _question(question)

    final: GeneratedAnswer | None = None
    with st.chat_message("assistant"):
        live = st.empty()
        # Retrieval and reranking take seconds before the first word can stream, and an empty
        # answer area for that long reads as a crash (feature 41). Replaced by the first piece.
        live.markdown(
            '<div class="citara-searching">Searching the indexed documents…</div>',
            unsafe_allow_html=True,
        )
        streamed = ""
        try:
            for piece, answer in answerer.stream(
                question,
                history=history,
                doc_ids=doc_ids,
                year=year,
                session_id=state.session_id,
            ):
                if answer is None:
                    streamed += piece
                    live.markdown(presenters.safe_markdown(streamed) + " ▌")
                else:
                    final = answer
        except Exception:
            log.exception("answering failed", extra={"question": question[:120]})
            live.empty()
            st.error(_ANSWER_FAILED, icon=":material/error:")
            return
        # The finished answer is authoritative. A provider that failed part-way is discarded
        # by the answerer, so what was streamed may belong to an answer that no longer exists.
        live.empty()
        if final is None:
            st.error(_ANSWER_FAILED, icon=":material/error:")
            return
        _render_answer(final, settings)

    turn = presenters.Turn(question=question, answer=final, asked_at=datetime.now(UTC), scope=scope)
    state.turns = [*state.turns, turn][-settings.ui.max_history_messages :]


def _render_turn(turn: presenters.Turn, settings: Settings) -> None:
    with st.chat_message("user"):
        _label("Question", first=True)
        _question(turn.question)
    with st.chat_message("assistant"):
        _render_answer(turn.answer, settings)


def _render_answer(answer: GeneratedAnswer, settings: Settings) -> None:
    heading = presenters.notice(answer)
    if heading is not None:
        _label(_NOTICE_LABEL[heading.kind])
        # A degraded answer's text is the notice followed by the passages, which the source
        # panel below shows properly; only the explanation belongs in the box.
        body = answer.text.split("\n\n", 1)[0] if answer.mode == "degraded" else answer.text
        st.markdown(
            f'<div class="citara-notice notice-{heading.kind}"><strong>'
            f"{html.escape(heading.heading)}</strong>{presenters.safe_markdown(body)}</div>",
            unsafe_allow_html=True,
        )
    if answer.mode == "generated":
        _label("Answer")
        st.markdown(presenters.render_answer_html(answer), unsafe_allow_html=True)

    _label("Evidence")
    meta = []
    band = presenters.evidence_band(answer, settings.ui)
    if band is not None:
        explanation = html.escape(presenters.BAND_EXPLANATION[band])
        meta.append(f'<span class="band band-{band}" title="{explanation}">{band}</span>')
        if band == "Low":
            meta.append(f"<span>{explanation}</span>")
    if settings.ui.show_latency:
        meta.append(f"<span>{html.escape(presenters.latency_line(answer))}</span>")
    st.markdown(f'<div class="citara-meta">{"".join(meta)}</div>', unsafe_allow_html=True)

    rows = presenters.source_rows(answer)
    if not rows:
        return
    if answer.mode == "generated":
        cited = sum(row.cited for row in rows)
        count = f"{cited} cited of {len(rows)} retrieved"
    else:
        count = f"{len(rows)} retrieved"
    _label("Sources", count=count)
    with st.expander("Show the passages", expanded=answer.mode == "degraded"):
        for row in rows:
            score = f"relevance {row.score:.2f}" if row.score is not None else ""
            status = ""
            if answer.mode == "generated":
                status = "cited in the answer" if row.cited else "retrieved, not cited"
            st.markdown(
                f'<div class="source-head"><span class="cite-chip">{row.marker}</span> '
                f"<strong>{html.escape(row.citation)}</strong> "
                f'<span class="score">{score}</span> <span class="state">{status}</span></div>',
                unsafe_allow_html=True,
            )
            st.markdown(presenters.safe_markdown(row.text))


# -- sidebar ----------------------------------------------------------------------------


def _sidebar(
    settings: Settings, corpus: Corpus, answerer: Answerer, problems: list[health.Problem]
) -> None:
    state = st.session_state
    with st.sidebar:
        _label("Search within", first=True)
        labels = {d.doc_id: d.label for d in corpus.searchable}
        st.multiselect(
            "Documents",
            options=list(labels),
            format_func=lambda doc_id: labels.get(doc_id, doc_id),
            key="doc_filter",
            placeholder="All documents",
        )
        # "Any year" as a real option rather than None: Streamlit reads a None option as
        # nothing selected and shows its own "Choose an option" placeholder instead.
        st.selectbox(
            "Year of publication",
            options=[ANY_YEAR, *(str(year) for year in corpus.years)],
            key="year_filter",
        )

        _label("This session")
        cap = answerer.sessions.cap
        st.caption(f"{answerer.sessions.remaining(state.session_id)} of {cap} questions left")
        budget = answerer.budget.state()
        if budget.exhausted:
            st.warning(
                "Today's model allowance is used up. Answers show cited source passages "
                "without a written summary until it resets."
            )
        elif budget.low:
            st.caption(f"Model allowance running low: {budget.remaining} requests left today.")
        now = datetime.now(UTC)
        st.download_button(
            "Download transcript",
            data=presenters.transcript(
                state.turns, settings.guardrails.prototype_disclaimer, settings.ui, now
            ),
            file_name=f"{settings.ui.transcript_prefix}-{now:%Y%m%d-%H%M}.txt",
            mime="text/plain",
            disabled=not state.turns,
            width="stretch",
        )
        st.button("Clear conversation", on_click=_clear, disabled=not state.turns, width="stretch")

        _label("Knowledge boundary")
        st.caption(
            f"{len(corpus.searchable)} documents · {corpus.searchable_pages:,} of "
            f"{corpus.total_pages:,} pages searchable. Answers come only from these."
        )
        st.markdown(_corpus_list(corpus), unsafe_allow_html=True)

        for problem in problems:
            if problem.severity == "info":
                st.caption(f"**{problem.title}.** {problem.detail}")
        st.caption(settings.guardrails.prototype_disclaimer)


def _corpus_list(corpus: Corpus) -> str:
    items = []
    for document in corpus.documents:
        if document.searchable:
            pages = (
                f"{document.searchable_pages} of {document.total_pages} pages"
                if document.searchable_pages < document.total_pages
                else f"{document.total_pages} pages"
            )
            gap = (
                f'<br><span class="gap">{html.escape(document.note)}</span>'
                if document.note
                else ""
            )
        else:
            pages = "not searchable"
            gap = f'<br><span class="gap">{html.escape(document.note)}</span>'
        items.append(
            f'<li>{html.escape(document.label)} <span class="pages">· {pages}</span>{gap}</li>'
        )
    return f'<ul class="corpus-list">{"".join(items)}</ul>'
