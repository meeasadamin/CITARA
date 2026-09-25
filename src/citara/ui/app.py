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
import json
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING

import streamlit as st
from streamlit.delta_generator import DeltaGenerator

from citara.config import Settings, get_settings
from citara.generation.models import GeneratedAnswer
from citara.indexing import fetch
from citara.log import configure_logging, get_logger
from citara.ui import health, logo, markup, presenters
from citara.ui.corpus import Corpus, load_corpus
from citara.ui.styles import CSS

if TYPE_CHECKING:
    from citara.generation.answerer import Answerer

log = get_logger("ui")

SUBTITLE = "Citation-grounded answers from NDMA's published disaster-management documents"
ANY_YEAR = "Any year"
REPOSITORY = "https://github.com/meeasadamin/CITARA"

# Streamlit's shell has no main landmark and no complementary one, so every element on the
# page counts as content outside any region - a screen reader cannot jump to the conversation
# or skip the filters. These attributes are the only thing this script sets, it re-applies
# harmlessly on every rerun, and it is the one place the interface needs JavaScript.
_LANDMARKS = """<script>
(() => {
  const win = window.parent || window;
  const doc = win.document;
  const apply = () => {
    const main = doc.querySelector('section.stMain, [data-testid="stMain"]');
    if (main && main.getAttribute('role') !== 'main') {
      main.setAttribute('role', 'main');
      main.setAttribute('aria-label', 'Questions and answers');
      main.setAttribute('tabindex', '-1');
      main.id = 'citara-main';
    }
    const sidebar = doc.querySelector('[data-testid="stSidebar"]');
    if (sidebar && !sidebar.getAttribute('aria-label')) {
      sidebar.setAttribute('aria-label', 'Filters, session and corpus');
    }
  };
  // Retried rather than observed: this component is torn down and rebuilt on every rerun,
  // which takes any MutationObserver with it, and the main container is not always in the
  // document at the moment the script first runs.
  [0, 80, 250, 800, 2000].forEach((delay) => win.setTimeout(apply, delay));
})();
</script>"""
FAVICON = Path(__file__).resolve().parents[3] / "assets" / "favicon.png"


# Curated for a ninety-second demo (feature 58) and chosen by running candidates live, not by
# guessing: each of the first three returned High evidence and cited answers on 2026-09-20.
# They show, in order, two documents disagreeing (35,000 against 30,000 dead) with each figure
# cited rather than one silently chosen; figures read out of a PDNA damage table; and the early
# warning chain. The last is deliberately unanswerable - the corpus has heatwave plans but no
# Gwadar-specific threshold - because showing the refusal unprompted beats hoping it is asked.
# Rejected on the same run: agriculture damage (the table holding the figure was not
# retrieved) and district control-room preparation (answered only in part).
# Everything the picker offers, the starters included. Each was run against the real index
# on 2026-09-20 and answered with citations, except the last, which is the refusal.
SUGGESTIONS: tuple[str, ...] = (
    "How many people died in the Quetta earthquake?",
    "How much damage did the water resources and irrigation sector suffer in 2022?",
    "How does NDMA disseminate early warnings to communities?",
    "What does NDMA advise people to do during a heatwave?",
    "What is NDMA's legal mandate in a national emergency?",
    "Which months does NDMA treat as the monsoon period?",
    "What must a district do to prepare its control room before the heatwave season?",
    "What preparations does NDMA require before the monsoon arrives?",
    "Which temperature threshold triggers a district-level heatwave alert specifically for Gwadar?",
)

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
        layout="wide",
        initial_sidebar_state="auto",
    )
    st.markdown(CSS, unsafe_allow_html=True)
    st.html(_LANDMARKS, unsafe_allow_javascript=True)
    st.markdown(
        '<a class="skip-link" href="#citara-main">Skip to the conversation</a>',
        unsafe_allow_html=True,
    )
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

    for index, turn in enumerate(state.turns, start=1):
        _render_turn(turn, settings, index)

    # Created here, filled by _ask: a new turn belongs with the conversation above, not
    # below the picker that comes after it in the script.
    answer_slot = st.empty()

    pending = state.pop("pending", None)
    if not state.turns and pending is None:
        _empty_state()

    scope = _scope_text(corpus, list(state.get("doc_filter") or []), _selected_year(state))
    if scope:
        st.markdown(
            f'<div class="citara-scope">Searching only: {html.escape(scope)}</div>',
            unsafe_allow_html=True,
        )

    _question_picker(state)
    typed = st.chat_input(
        "Ask about NDMA plans, advisories and guidelines",
        max_chars=settings.guardrails.max_query_chars,
    )
    question = (pending or typed or "").strip()
    if question:
        _ask(question, answerer, settings, scope, answer_slot)

    _footer(corpus, settings)
    # Rendered last so the transcript and the remaining-question count include this turn.
    _sidebar(settings, corpus, answerer, problems)


# -- page furniture ---------------------------------------------------------------------


def _header(settings: Settings) -> None:
    """Brand bar and the prototype disclaimer, visible on every screen size (feature 47)."""
    st.markdown(
        markup.header(SUBTITLE, settings.guardrails.prototype_disclaimer, logo.on_dark()),
        unsafe_allow_html=True,
    )


def _label(text: str) -> None:
    """A sidebar section heading: small caps over a hairline rule, and a real heading."""
    st.markdown(f'<h2 class="section-label">{html.escape(text)}</h2>', unsafe_allow_html=True)


def _footer(corpus: Corpus, settings: Settings) -> None:
    """Site footer: what the assistant knows, where the code is, and what it is not."""
    built = ""
    manifest = settings.paths.resolved(settings.paths.index_manifest_path)
    if manifest.is_file():
        try:
            built = str(json.loads(manifest.read_text(encoding="utf-8")).get("built_at", ""))
        except (OSError, ValueError):
            built = ""
    st.markdown(markup.footer(corpus, built, REPOSITORY), unsafe_allow_html=True)


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


def _pick() -> None:
    """Ask the question chosen in the picker, then clear it for the next one."""
    chosen = st.session_state.get("picker")
    if chosen:
        st.session_state.pending = chosen
        st.session_state.picker = None


def _question_picker(state: object) -> None:
    """A searchable list of questions this corpus answers (feature 58).

    Streamlit's chat box reports nothing until it is submitted, so it cannot suggest while
    someone types. A selectbox can: it filters its options on every keystroke, which is the
    same help in the one place the framework allows it.
    """
    asked = [turn.question for turn in st.session_state.turns]
    options = list(dict.fromkeys([*SUGGESTIONS, *asked]))
    st.selectbox(
        "Find a question",
        options=options,
        index=None,
        key="picker",
        on_change=_pick,
        placeholder="Type to filter questions this corpus answers, or ask your own below",
        label_visibility="collapsed",
    )


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


def _ask(
    question: str,
    answerer: Answerer,
    settings: Settings,
    scope: str,
    slot: DeltaGenerator,
) -> None:
    """Stream an answer into *slot*, then replace it with the finished, validated answer."""
    state = st.session_state
    doc_ids = list(state.get("doc_filter") or []) or None
    year = _selected_year(state)
    history = [turn.question for turn in state.turns]

    live = slot
    # Retrieval and reranking take seconds before the first word can stream, and an empty
    # answer area for that long reads as a crash (feature 41).
    live.markdown(markup.streaming_article(question, ""), unsafe_allow_html=True)

    final: GeneratedAnswer | None = None
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
                # Deliberately not a live region while it streams: a screen reader fed token
                # by token stutters through half-words. The finished article announces itself
                # once, through the status line markup adds to it.
                live.markdown(markup.streaming_article(question, streamed), unsafe_allow_html=True)
            else:
                final = answer
    except Exception:
        log.exception("answering failed", extra={"question": question[:120]})
        live.empty()
        st.error(_ANSWER_FAILED, icon=":material/error:")
        return

    if final is None:
        live.empty()
        st.error(_ANSWER_FAILED, icon=":material/error:")
        return

    turn = presenters.Turn(question=question, answer=final, asked_at=datetime.now(UTC), scope=scope)
    state.turns = [*state.turns, turn][-settings.ui.max_history_messages :]
    # The finished answer is authoritative: a provider that failed part-way is discarded by
    # the answerer, so what was streamed may belong to an answer that no longer exists.
    live.markdown(markup.turn_article(turn, settings, len(state.turns)), unsafe_allow_html=True)


def _render_turn(turn: presenters.Turn, settings: Settings, index: int) -> None:
    st.markdown(markup.turn_article(turn, settings, index), unsafe_allow_html=True)


# -- sidebar ----------------------------------------------------------------------------


def _sidebar(
    settings: Settings, corpus: Corpus, answerer: Answerer, problems: list[health.Problem]
) -> None:
    state = st.session_state
    with st.sidebar:
        st.markdown(logo.wordmark(), unsafe_allow_html=True)
        _label("Search within")
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
        st.markdown(markup.corpus_list(corpus), unsafe_allow_html=True)

        for problem in problems:
            if problem.severity == "info":
                st.caption(f"**{problem.title}.** {problem.detail}")
        st.caption(settings.guardrails.prototype_disclaimer)
