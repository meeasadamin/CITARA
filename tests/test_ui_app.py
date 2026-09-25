"""The Streamlit app run headless: flows, not pixels (features 47, 53-65).

A fake answerer stands in for the models, so these exercise what the interface does with each
kind of answer - chips, refusals, failures, filters - in seconds rather than minutes.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pytest
from streamlit.testing.v1 import AppTest

from citara.chunking.models import Chunk
from citara.generation.citations import build_citations
from citara.generation.models import GeneratedAnswer
from citara.resilience.budget import RequestBudget, SessionLimiter
from citara.retrieval.models import RetrievedChunk
from citara.ui import health
from citara.ui.app import STARTERS
from citara.ui.corpus import Corpus, CorpusDocument

APP = str(Path(__file__).resolve().parents[1] / "streamlit_app.py")
REFUSAL = STARTERS[-1]


def evidence(page: int = 47) -> RetrievedChunk:
    chunk = Chunk(
        chunk_id=f"c{page}",
        doc_id="ndrp",
        title="NDRP 2019",
        filename="ndrp.pdf",
        page_start=page,
        page_end=page,
        total_pages=110,
        year=2019,
        ordinal=0,
        content="Evacuate low-lying areas when the river crosses the danger mark.",
    )
    return RetrievedChunk(chunk=chunk, rerank_score=0.92, fusion_score=0.5)


def cited_answer(text: str = "Evacuate low-lying areas [1].") -> GeneratedAnswer:
    results = [evidence()]
    citations = build_citations(results)
    citations[0].used = "[1]" in text
    return GeneratedAnswer(
        text=text,
        citations=citations,
        evidence=results,
        provider="gemini",
        model="gemini-test",
        retrieval_ms=3400.0,
        generation_ms=1500.0,
    )


@dataclass
class FakeAnswerer:
    """The slice of Answerer the interface uses."""

    budget: RequestBudget
    sessions: SessionLimiter = field(default_factory=lambda: SessionLimiter(cap=40))
    calls: list[dict[str, Any]] = field(default_factory=list)
    pieces: list[str] = field(default_factory=lambda: ["Evacuate ", "low-lying areas [1]."])
    final: GeneratedAnswer | None = None
    fail: bool = False

    def stream(self, question: str, **kwargs: Any) -> Iterator[tuple[str, GeneratedAnswer | None]]:
        self.calls.append({"question": question, **kwargs})
        if self.fail:
            raise RuntimeError("chromadb exploded somewhere deep")
        if question == REFUSAL:
            refusal = GeneratedAnswer(
                text="I could not find this in the indexed NDMA corpus.",
                mode="refused",
                reason="no_evidence",
                retrieval_ms=3100.0,
            )
            yield refusal.text, refusal
            return
        for piece in self.pieces:
            yield piece, None
        yield "", self.final or cited_answer()


CORPUS = Corpus(
    documents=[
        CorpusDocument("ndrp", "NDRP 2019", 2019, 110, 49, 50, "61 scanned pages not searchable"),
        CorpusDocument("pdna", "PDNA 2022 Floods Main Report", 2022, 184, 182, 575),
    ]
)


@pytest.fixture
def fake(tmp_path: Path) -> FakeAnswerer:
    return FakeAnswerer(budget=RequestBudget(tmp_path / "usage.json", daily_limit=1000))


@pytest.fixture
def app(fake: FakeAnswerer, monkeypatch: pytest.MonkeyPatch) -> AppTest:
    import citara.ui.app as ui

    monkeypatch.setattr(ui, "load_answerer", lambda: fake)
    monkeypatch.setattr(ui, "corpus_boundary", lambda: CORPUS)
    monkeypatch.setattr(ui, "check_health", lambda settings: [])
    return AppTest.from_file(APP, default_timeout=60)


def markdown_blocks(at: AppTest) -> list[str]:
    return [str(element.value) for element in at.markdown]


def all_markdown(at: AppTest) -> str:
    return "\n".join(markdown_blocks(at))


def ask(at: AppTest, question: str) -> AppTest:
    at.chat_input[0].set_value(question)
    return at.run()


# --- first screen --------------------------------------------------------------------


def test_the_disclaimer_is_on_screen_before_anything_is_asked(app: AppTest) -> None:
    """Feature 47: in the main area, not only in a sidebar a phone hides."""
    app.run()
    assert not app.exception
    assert "Not an official NDMA system." in all_markdown(app)


def test_starter_questions_include_a_deliberate_refusal(app: AppTest) -> None:
    app.run()
    labels = [button.label for button in app.button]
    for starter in STARTERS:
        assert starter in labels
    assert any("refuses" in caption.value for caption in app.caption)


def test_the_sidebar_shows_the_knowledge_boundary(app: AppTest) -> None:
    app.run()
    sidebar = "\n".join(str(e.value) for e in app.sidebar.markdown)
    captions = "\n".join(str(c.value) for c in app.sidebar.caption)
    assert "NDRP 2019 (2019)" in sidebar
    assert "49 of 110 pages" in sidebar
    assert "61 scanned pages not searchable" in sidebar
    assert "231 of 294 pages searchable" in captions


# --- asking --------------------------------------------------------------------------


def test_a_starter_is_answered_with_citation_chips(app: AppTest, fake: FakeAnswerer) -> None:
    app.run()
    next(b for b in app.button if b.label == STARTERS[0]).click().run()

    assert not app.exception
    assert fake.calls[0]["question"] == STARTERS[0]
    rendered = all_markdown(app)
    assert 'class="cite-chip"' in rendered
    assert "1 · p. 47" in rendered
    assert '<h3 class="section-label label-answer" id="a-1">Answer</h3>' in rendered
    assert '<h2 class="question" id="q-1">' in rendered
    assert '<article class="turn" aria-labelledby="q-1">' in rendered
    assert '<span class="band band-High"' in rendered
    assert "Retrieval 3.4 s" in rendered
    # Once asked, the starters give way to the conversation.
    assert STARTERS[1] not in [b.label for b in app.button]


def test_the_refusal_is_stated_plainly(app: AppTest) -> None:
    ask(app.run(), REFUSAL)
    rendered = all_markdown(app)
    assert "No supporting evidence in the indexed documents" in rendered
    assert "no model called" in rendered
    # The turn itself, not the whole page: the stylesheet names these classes too.
    turn = next(block for block in markdown_blocks(app) if "<article" in block)
    assert 'class="citara-notice' in turn
    assert "cite-chip" not in turn


def test_the_final_answer_replaces_what_was_streamed(app: AppTest, fake: FakeAnswerer) -> None:
    """A provider that died mid-stream is discarded; its words must not stay on screen."""
    fake.pieces = ["Half an answer from a provider that then failed "]
    fake.final = cited_answer("Evacuate low-lying areas [1].")

    ask(app.run(), "What should people do when water rises?")

    rendered = all_markdown(app)
    assert "Half an answer" not in rendered
    assert "Evacuate low-lying areas" in rendered
    assert "Searching the indexed documents" not in rendered  # the interim line is replaced


def test_the_conversation_persists_and_feeds_follow_ups(app: AppTest, fake: FakeAnswerer) -> None:
    app.run()
    ask(app, "What were the flood damages?")
    ask(app, "What about Sindh?")

    assert fake.calls[1]["history"] == ["What were the flood damages?"]
    rendered = all_markdown(app)
    assert "What were the flood damages?" in rendered
    assert "What about Sindh?" in rendered
    assert fake.calls[0]["session_id"] == fake.calls[1]["session_id"] != ""


def test_filters_reach_retrieval_and_are_visible(app: AppTest, fake: FakeAnswerer) -> None:
    app.run()
    app.sidebar.multiselect[0].set_value(["ndrp"])
    app.sidebar.selectbox[0].set_value("2019")
    app.run()
    ask(app, "What should people do?")

    assert fake.calls[-1]["doc_ids"] == ["ndrp"]
    assert fake.calls[-1]["year"] == 2019
    assert "Searching only: NDRP 2019; published 2019" in all_markdown(app)


def test_model_output_cannot_inject_markup(app: AppTest, fake: FakeAnswerer) -> None:
    fake.final = cited_answer('<img src=x onerror="alert(1)"> Evacuate [1].')
    ask(app.run(), "What should people do?")
    rendered = all_markdown(app)
    assert "<img" not in rendered
    assert "&lt;img" in rendered


# --- failures (feature 65) -----------------------------------------------------------


def test_a_failure_while_answering_is_explained_not_dumped(
    app: AppTest, fake: FakeAnswerer
) -> None:
    fake.fail = True
    ask(app.run(), "What should people do?")

    assert not app.exception
    assert any("Something went wrong while answering" in e.value for e in app.error)
    assert "chromadb exploded" not in all_markdown(app)


def test_a_missing_index_stops_the_app_with_the_fix(
    fake: FakeAnswerer, monkeypatch: pytest.MonkeyPatch
) -> None:
    import citara.ui.app as ui

    problem = health.Problem("error", "The document index is missing", "Run the build.")
    monkeypatch.setattr(ui, "check_health", lambda settings: [problem])
    monkeypatch.setattr(ui, "load_answerer", lambda: pytest.fail("must not load models"))
    at = AppTest.from_file(APP, default_timeout=60).run()

    assert not at.exception
    assert any("The document index is missing" in e.value for e in at.error)
    assert not at.chat_input


def test_a_model_load_failure_is_explained(monkeypatch: pytest.MonkeyPatch) -> None:
    import citara.ui.app as ui

    def broken() -> None:
        raise OSError("connection reset while downloading weights")

    monkeypatch.setattr(ui, "check_health", lambda settings: [])
    monkeypatch.setattr(ui, "load_answerer", broken)
    at = AppTest.from_file(APP, default_timeout=60).run()

    assert not at.exception
    assert any("could not be loaded" in e.value and "OSError" in e.value for e in at.error)
    assert not at.chat_input


def test_a_missing_key_is_a_warning_and_the_app_still_runs(
    app: AppTest, monkeypatch: pytest.MonkeyPatch
) -> None:
    import citara.ui.app as ui

    problem = health.Problem("warning", "No language-model key is configured", "Add a key.")
    monkeypatch.setattr(ui, "check_health", lambda settings: [problem])
    app.run()

    assert any("No language-model key" in w.value for w in app.warning)
    assert app.chat_input  # degraded mode still serves cited sources


def test_the_picker_offers_questions_this_corpus_answers(app: AppTest) -> None:
    """Streamlit's chat box cannot suggest while typing; the picker filters as you type."""
    from citara.ui.app import SUGGESTIONS

    app.run()
    picker = app.selectbox[0]
    assert picker.options[: len(SUGGESTIONS)] == list(SUGGESTIONS)
    for starter in STARTERS:
        assert starter in picker.options


def test_picking_a_question_asks_it(app: AppTest, fake: FakeAnswerer) -> None:
    app.run()
    app.selectbox[0].set_value(STARTERS[2]).run()

    assert fake.calls[-1]["question"] == STARTERS[2]
    assert "cite-chip" in all_markdown(app)
    # Cleared afterwards, so the same question can be picked again.
    assert app.session_state["picker"] is None


def test_questions_already_asked_join_the_picker(app: AppTest) -> None:
    app.run()
    ask(app, "What were the flood damages in Sindh?")
    # The picker is built before the turn is appended, so it gains the question on the next
    # run - which is the next thing the visitor does anyway.
    app.run()
    assert "What were the flood damages in Sindh?" in app.selectbox[0].options
