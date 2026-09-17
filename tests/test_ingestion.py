"""Tests for PDF loading, provenance and the corpus manifest (features 1-11).

Synthetic PDFs keep these fast and deterministic; the real corpus is exercised separately.
"""

from __future__ import annotations

from pathlib import Path

import pymupdf
import pytest

from citara.config import Paths, Settings
from citara.ingestion.loader import derive_title, guess_year, ingest_corpus, load_document, slugify
from citara.ingestion.models import CorpusManifest, DocumentReport, PageRecord
from citara.ingestion.store import read_manifest, read_records, write_manifest, write_records
from citara.ingestion.tables import _collapse_duplicate_columns, to_markdown


def make_pdf(path: Path, pages: list[str], title: str | None = None) -> Path:
    """Write a simple text PDF.

    Uses a wrapping text box: a single insert_text line runs off the page and much of it is
    lost, which would make page-length assertions meaningless.
    """
    doc = pymupdf.open()
    for body in pages:
        page = doc.new_page()
        if body:
            page.insert_textbox(pymupdf.Rect(50, 50, 545, 780), body, fontsize=11)
    if title:
        doc.set_metadata({"title": title})
    doc.save(path)
    doc.close()
    return path


def settings_for(tmp_path: Path) -> Settings:
    return Settings(
        _env_file=None,  # type: ignore[call-arg]
        paths=Paths(docs_dir=tmp_path / "docs", data_dir=tmp_path / "data"),
    )


@pytest.fixture
def corpus(tmp_path: Path) -> Path:
    docs = tmp_path / "docs"
    docs.mkdir(parents=True)
    make_pdf(
        docs / "NDRP-2019-Response-Plan.pdf",
        [
            "Evacuation protocol for inundation events. " * 6,
            "Responsibilities of the district administration. " * 6,
        ],
    )
    make_pdf(docs / "Advisory-2026.pdf", ["Heatwave advisory for Sindh province. " * 6])
    return docs


def test_record_ids_are_deterministic_and_location_based() -> None:
    first = PageRecord.make_id("ndrp-2019", 47, "text")
    assert first == PageRecord.make_id("ndrp-2019", 47, "text")
    assert first != PageRecord.make_id("ndrp-2019", 48, "text")
    assert first != PageRecord.make_id("ndrp-2019", 47, "table", 0)


def test_citation_renders_document_and_page() -> None:
    record = PageRecord(
        record_id="x",
        doc_id="ndrp",
        title="NDRP 2019",
        filename="ndrp.pdf",
        page_number=47,
        total_pages=110,
        content="text",
    )
    assert record.citation == "NDRP 2019, p. 47"


def test_page_provenance_captured_at_extraction(corpus: Path, tmp_path: Path) -> None:
    settings = settings_for(tmp_path)
    records, report = load_document(corpus / "NDRP-2019-Response-Plan.pdf", settings)

    assert report.error is None
    assert report.total_pages == 2
    assert report.year == 2019
    assert len(records) == 2
    for index, record in enumerate(records, start=1):
        assert record.page_number == index
        assert record.total_pages == 2
        assert record.filename == "NDRP-2019-Response-Plan.pdf"
        assert record.doc_id == "ndrp-2019-response-plan"
        assert record.year == 2019


def test_sparse_pages_are_skipped_and_counted(tmp_path: Path) -> None:
    docs = tmp_path / "docs"
    docs.mkdir(parents=True)
    pdf = make_pdf(docs / "sparse-2024.pdf", ["Real content here. " * 20, "x", ""])
    records, report = load_document(pdf, settings_for(tmp_path))

    assert len(records) == 1
    assert report.pages_sparse == 2
    assert report.pages_indexed == 1
    assert report.total_pages == 3


def test_corrupt_file_is_isolated_not_fatal(tmp_path: Path) -> None:
    docs = tmp_path / "docs"
    docs.mkdir(parents=True)
    (docs / "broken.pdf").write_bytes(b"%PDF-1.4 this is not a real pdf")
    make_pdf(docs / "good-2025.pdf", ["Valid content for indexing. " * 20])

    records, manifest = ingest_corpus(settings_for(tmp_path))

    assert len(records) == 1  # the good document still made it
    assert manifest.documents_failed == 1
    broken = next(d for d in manifest.documents if d.filename == "broken.pdf")
    assert broken.error is not None


def test_scanned_document_reported_not_silently_empty(tmp_path: Path) -> None:
    """A document with no text layer must be flagged, not quietly contribute nothing."""
    docs = tmp_path / "docs"
    docs.mkdir(parents=True)
    doc = pymupdf.open()
    for _ in range(3):
        doc.new_page()
    path = docs / "scanned-2026.pdf"
    doc.save(path)
    doc.close()

    records, report = load_document(path, settings_for(tmp_path))
    assert records == []
    assert report.needs_ocr is True
    assert report.ok is False


def test_subdirectories_are_not_ingested(corpus: Path, tmp_path: Path) -> None:
    """Deferred documents live in docs/_deferred_needs_ocr and must stay out."""
    deferred = corpus / "_deferred_needs_ocr"
    deferred.mkdir()
    make_pdf(deferred / "scanned-plan.pdf", ["Should never be indexed. " * 20])

    _, manifest = ingest_corpus(settings_for(tmp_path))
    assert all(d.filename != "scanned-plan.pdf" for d in manifest.documents)


def test_manifest_totals_and_roundtrip(corpus: Path, tmp_path: Path) -> None:
    settings = settings_for(tmp_path)
    records, manifest = ingest_corpus(settings)

    assert manifest.documents_indexed == 2
    assert manifest.total_pages == 3
    assert manifest.total_records == len(records)
    assert manifest.config_fingerprint == settings.fingerprint()

    manifest_path = tmp_path / "data" / "corpus_manifest.json"
    write_manifest(manifest, manifest_path)
    restored = read_manifest(manifest_path)
    assert restored.summary() == manifest.summary()


def test_records_roundtrip_through_jsonl(corpus: Path, tmp_path: Path) -> None:
    records, _ = ingest_corpus(settings_for(tmp_path))
    path = tmp_path / "data" / "pages.jsonl"
    write_records(records, path)
    restored = list(read_records(path))
    assert [r.record_id for r in restored] == [r.record_id for r in records]
    assert restored[0].content == records[0].content


def test_font_repair_flag_recorded(tmp_path: Path) -> None:
    docs = tmp_path / "docs"
    docs.mkdir(parents=True)
    # Latin-1 characters only: PyMuPDF's base-14 fonts cannot encode U+019F, so a Ɵ written
    # here would never survive the round trip and the test would prove nothing.
    pdf = make_pdf(
        docs / "broken-font-2019.pdf",
        ["ïrst response to ðooding in the district. " * 8],
    )
    records, report = load_document(pdf, settings_for(tmp_path))

    assert report.font_repaired is True
    assert "first response to flooding" in records[0].content
    assert "ï" not in records[0].content
    assert "ð" not in records[0].content


def test_font_corruption_detected_beyond_the_front_matter(tmp_path: Path) -> None:
    """Regression: detection once sampled only the first pages.

    Real NDMA documents open with a cover, foreword and contents pages that carry almost no
    prose, so the corruption signature does not appear until well into the body. Sampling
    the front matter alone reported zero broken documents while 'evacuaƟon' and 'QueƩa'
    were being indexed verbatim.
    """
    docs = tmp_path / "docs"
    docs.mkdir(parents=True)
    pages = ["Cover page", "Contents", "Foreword", "Acknowledgements", "Abbreviations"]
    pages += ["ïrst response arrangements for ðooding in the district. " * 8] * 3
    pdf = make_pdf(docs / "late-corruption-2020.pdf", pages)

    records, report = load_document(pdf, settings_for(tmp_path))

    assert report.font_repaired is True
    body = [r for r in records if "response" in r.content]
    assert body, "expected body pages to be indexed"
    assert all("ï" not in r.content and "ð" not in r.content for r in body)


def test_clean_document_not_font_repaired(corpus: Path, tmp_path: Path) -> None:
    _, report = load_document(corpus / "Advisory-2026.pdf", settings_for(tmp_path))
    assert report.font_repaired is False


def test_year_from_range_uses_publication_year() -> None:
    assert guess_year("NDRR-Strategy-2025-2030", {}, "") == 2025
    assert guess_year("Sendai-Framework-2015-2030", {}, "") == 2015
    assert guess_year("Advisory-11Sep2026", {}, "") == 2026
    assert guess_year("no-year-here", {}, "published in 2021") == 2021
    assert guess_year("no-year-here", {}, "") is None


def test_descriptive_filename_always_wins_over_metadata(tmp_path: Path) -> None:
    """Citations must not read 'Orange and White Illustrated Fire Safety Tips Poster'.

    Both of these titles are real metadata from this corpus: a design-tool template name on
    the fire safety guidelines, and a CorelDRAW filename on the mitigation plan.
    """
    fire = tmp_path / "NDMA-Urban-Fire-Safety-Guidelines-2026.pdf"
    assert derive_title(
        fire, {"title": "Orange and White Illustrated Fire Safety Tips Poster"}
    ) == ("NDMA Urban Fire Safety Guidelines 2026")
    plan = tmp_path / "NDMP-II-2023-National-Disaster-Mitigation-Plan.pdf"
    assert derive_title(plan, {"title": "NDMA Report Final.cdr"}).startswith("NDMP")

    ndrp = tmp_path / "NDRP-2019-National-Disaster-Response-Plan.pdf"
    assert derive_title(ndrp, {}) == "NDRP 2019 National Disaster Response Plan"


def test_opaque_filename_falls_back_to_metadata(tmp_path: Path) -> None:
    """NDMA's own download URLs are upload hashes, so metadata is the better name there."""
    path = tmp_path / "6a27bf8ac0ab0.pdf"
    assert derive_title(path, {"title": "Monsoon Contingency Plan 2026"}) == (
        "Monsoon Contingency Plan 2026"
    )
    assert derive_title(path, {"title": "Microsoft Word - final_v3.docx"}) == "6a27bf8ac0ab0"


def test_slugify() -> None:
    assert slugify("NDRP-2019 National_Plan") == "ndrp-2019-national-plan"


def test_duplicate_columns_from_merged_cells_collapse() -> None:
    rows = [
        ["Province", "Province", "Province", "Total", "Total"],
        ["Sindh", "Sindh", "Sindh", "353,594", "353,594"],
    ]
    assert _collapse_duplicate_columns(rows) == [["Province", "Total"], ["Sindh", "353,594"]]


def test_empty_columns_dropped() -> None:
    rows = [["A", "", "B"], ["1", "", "2"]]
    assert _collapse_duplicate_columns(rows) == [["A", "B"], ["1", "2"]]


def test_staggered_columns_are_merged() -> None:
    """Regression: PyMuPDF splits one logical column across two physical ones.

    Shape taken from PDNA 2022 p.24, where header labels land in odd columns and their
    figures in even ones. Rendered as-is, every other column is empty and the row is unusable.
    """
    from citara.ingestion.tables import _merge_complementary_columns

    rows = [
        ["", "Province/Region", "", "CD*", "", "PD**", "", "Total"],
        ["Balochistan", "", "65,542", "", "117,432", "", "192,605", ""],
        ["KP", "", "10,929", "", "17,779", "", "91,932", ""],
    ]
    merged = _merge_complementary_columns(rows)
    assert merged == [
        ["Province/Region", "CD*", "PD**", "Total"],
        ["Balochistan", "65,542", "117,432", "192,605"],
        ["KP", "10,929", "17,779", "91,932"],
    ]


def test_genuinely_distinct_columns_are_not_merged() -> None:
    """Columns filled together in the same row are real columns."""
    from citara.ingestion.tables import _merge_complementary_columns

    rows = [["District", "Deaths"], ["Dadu", "120"], ["Thatta", "45"]]
    assert _merge_complementary_columns(rows) == rows


def test_table_markdown_keeps_its_line_structure() -> None:
    """Regression: the prose normaliser folded table rows into a single line."""
    from citara.ingestion.normalise import normalise_markdown

    table = "| District | Deaths |\n| --- | --- |\n| QueƩa | 120 |"
    out = normalise_markdown(table, repair_font=True)
    assert out.count("\n") == 2
    assert "| Quetta | 120 |" in out


def test_markdown_serialisation_merges_split_headers() -> None:
    rows = [
        ["Province", "Value of Damaged Assets"],
        ["", "(PKR Million)"],
        ["Sindh", "353,594"],
    ]
    markdown = to_markdown(rows)
    assert "| Province | Value of Damaged Assets (PKR Million) |" in markdown
    assert "| Sindh | 353,594 |" in markdown


def test_text_only_table_keeps_its_body_rows() -> None:
    """Regression: an unbounded header swallowed every row of a table containing no digits.

    Responsibility matrices in the INSaR and contingency plans are pure text, so there is no
    numeric row to mark where the header ends. 56 tables were indexed as a header and nothing
    else.
    """
    rows = [
        ["Serial", "Team", "Area"],
        ["1", "Navy", "Karachi coast"],
        ["2", "Army", "Northern districts"],
    ]
    markdown = to_markdown(rows)
    assert "| 1 | Navy | Karachi coast |" in markdown
    assert "| 2 | Army | Northern districts |" in markdown


def test_cell_repairs_line_broken_compounds() -> None:
    """Regression: 540 compounds stayed broken inside table cells.

    Cells are flattened to one line here rather than by the prose normaliser, so the prose
    fix did not reach them - "Medium-\\nterm" became "Medium- term" in the DRR Strategy.
    """
    from citara.ingestion.tables import _clean_cell

    assert _clean_cell("Medium-\nterm") == "Medium-term"
    assert _clean_cell("high-\n  risk") == "high-risk"
    assert _clean_cell("Improved  resilience\nof assets") == "Improved resilience of assets"


def test_header_only_table_is_rejected() -> None:
    """A table with no data rows carries no information; the caller flags it instead."""
    assert to_markdown([["District", "Deaths"]]) == ""


def test_placeholder_headers_discarded() -> None:
    rows = [["Col1", "Col2"], ["Province", "Total"], ["Sindh", "42"]]
    markdown = to_markdown(rows)
    assert "Col1" not in markdown
    assert "Province" in markdown


def test_document_report_ok_semantics() -> None:
    assert DocumentReport(filename="a.pdf").ok is True
    assert DocumentReport(filename="a.pdf", error="boom").ok is False
    assert DocumentReport(filename="a.pdf", needs_ocr=True).ok is False


def test_empty_manifest_summary() -> None:
    assert CorpusManifest().summary()["documents_indexed"] == 0
