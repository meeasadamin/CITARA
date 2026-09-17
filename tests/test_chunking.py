"""Tests for chunking (features 12-17).

A fake embedder keeps these deterministic and fast: the real model would make every
assertion depend on a 130 MB download and its similarity quirks.
"""

from __future__ import annotations

import numpy as np
import pytest

from citara.chunking.dedup import deduplicate
from citara.chunking.models import Chunk
from citara.chunking.pipeline import _apply_overlap, _page_streams, _split_table, chunk_records
from citara.chunking.semantic import breakpoint_threshold, recursive_split, split_sentences
from citara.config import ChunkingSettings, Settings
from citara.ingestion.models import PageRecord


class FakeEmbedder:
    """Deterministic embeddings: identical text embeds identically, different text differs.

    Sentences sharing their first word are placed close together, which lets a test state
    exactly where a topic shift should be detected.
    """

    def __init__(self, dimension: int = 16) -> None:
        self.dimension = dimension

    def embed_documents(self, texts: list[str]) -> np.ndarray:
        vectors = []
        for text in texts:
            token = text.strip().split(" ")[0].lower() if text.strip() else ""
            rng = np.random.default_rng(abs(hash(token)) % (2**32))
            vector = rng.normal(size=self.dimension)
            vectors.append(vector / np.linalg.norm(vector))
        return np.asarray(vectors, dtype=np.float32)

    def embed_query(self, text: str) -> np.ndarray:
        return self.embed_documents([text])[0]


def make_settings(**chunking: object) -> Settings:
    return Settings(
        _env_file=None,  # type: ignore[call-arg]
        chunking=ChunkingSettings(**chunking),  # type: ignore[arg-type]
    )


def page(doc: str, number: int, content: str, kind: str = "text", year: int = 2026) -> PageRecord:
    return PageRecord(
        record_id=PageRecord.make_id(doc, number, kind),  # type: ignore[arg-type]
        doc_id=doc,
        title=doc.upper(),
        filename=f"{doc}.pdf",
        page_number=number,
        total_pages=50,
        year=year,
        kind=kind,  # type: ignore[arg-type]
        content=content,
    )


def test_chunk_ids_are_deterministic_and_location_based() -> None:
    first = Chunk.make_id("ndrp", 46, 47, 3)
    assert first == Chunk.make_id("ndrp", 46, 47, 3)
    assert first != Chunk.make_id("ndrp", 46, 47, 4)
    assert first != Chunk.make_id("ndrp", 46, 48, 3)


def test_citation_renders_single_page_and_range() -> None:
    base = dict(
        chunk_id="c",
        doc_id="d",
        title="NDRP 2019",
        filename="f.pdf",
        total_pages=110,
        ordinal=0,
        content="x",
    )
    assert Chunk(page_start=47, page_end=47, **base).citation == "NDRP 2019, p. 47"  # type: ignore[arg-type]
    assert Chunk(page_start=46, page_end=47, **base).citation == "NDRP 2019, pp. 46-47"  # type: ignore[arg-type]


def test_sentence_splitting() -> None:
    text = "Evacuate low ground. Move to higher ground.\n\nContact the DDMA office."
    assert split_sentences(text) == [
        "Evacuate low ground.",
        "Move to higher ground.",
        "Contact the DDMA office.",
    ]


def test_consecutive_pages_form_one_stream() -> None:
    records = [page("d", 1, "a"), page("d", 2, "b"), page("d", 4, "c")]
    streams = _page_streams(records)
    assert [[r.page_number for r in s] for s in streams] == [[1, 2], [4]]


def test_prose_can_span_a_page_boundary() -> None:
    """A procedure split by a page break must be able to stay in one chunk."""
    settings = make_settings(min_chunk_chars=10, fallback_chunk_chars=4000, max_chunk_chars=8000)
    records = [
        page("ndrp", 46, "Evacuation step one: sound the siren and alert the union council."),
        page("ndrp", 47, "Evacuation step two: move residents to the designated relief camp."),
    ]
    chunks, _ = chunk_records(records, settings, FakeEmbedder())  # type: ignore[arg-type]
    spanning = [c for c in chunks if c.page_start != c.page_end]
    assert spanning, "expected at least one chunk covering both pages"
    assert spanning[0].citation.startswith("NDRP, pp. 46-47")
    assert len(spanning[0].source_record_ids) == 2


def test_citation_pages_survive_paragraph_breaks() -> None:
    """Regression: a chunk spanning a blank line could not be located in the page stream.

    Semantic splitting joins sentences with a single space while the stream keeps the
    original paragraph breaks, so a plain string search failed and the page span fell back
    to a guess - meaning a citation could name the wrong page.
    """
    settings = make_settings(min_chunk_chars=10, fallback_chunk_chars=4000, max_chunk_chars=8000)
    records = [
        page("d", 3, "Alpha paragraph one.\n\nAlpha paragraph two about relief camps."),
        page("d", 4, "Beta paragraph about heatwave thresholds in southern districts."),
    ]
    chunks, report = chunk_records(records, settings, FakeEmbedder())  # type: ignore[arg-type]

    assert report.unlocated_chunks == 0
    for chunk in chunks:
        first_words = chunk.content.split()[0].lower()
        if first_words.startswith("alpha"):
            assert chunk.page_start == 3
        if first_words.startswith("beta"):
            assert chunk.page_start == 4


def test_page_gap_breaks_the_stream() -> None:
    """A skipped image-only page is unknown content, not continuity."""
    settings = make_settings(min_chunk_chars=10, fallback_chunk_chars=4000, max_chunk_chars=8000)
    records = [page("d", 10, "Text about floods. " * 4), page("d", 20, "Text about heat. " * 4)]
    chunks, _ = chunk_records(records, settings, FakeEmbedder())  # type: ignore[arg-type]
    assert all(c.page_start == c.page_end for c in chunks)


def test_tables_are_never_split_below_the_ceiling() -> None:
    table = "| District | Deaths |\n| --- | --- |\n| Dadu | 120 |\n| Thatta | 45 |"
    settings = make_settings(min_chunk_chars=10)
    chunks, _ = chunk_records([page("d", 5, table, kind="table")], settings, FakeEmbedder())  # type: ignore[arg-type]
    assert len(chunks) == 1
    assert chunks[0].kind == "table"
    assert chunks[0].strategy == "table"
    assert chunks[0].content == table


def test_oversized_table_splits_by_rows_repeating_the_header() -> None:
    """Rows without their header are meaningless, so every part carries it."""
    header = "| District | Deaths |\n| --- | --- |"
    rows = "\n".join(f"| District{i} | {i} |" for i in range(200))
    settings = make_settings(max_chunk_chars=600, fallback_chunk_chars=500)
    parts = _split_table(f"{header}\n{rows}", settings)
    assert len(parts) > 1
    assert all(p.startswith("| District | Deaths |\n| --- | --- |") for p in parts)
    assert all(len(p) <= 700 for p in parts)


def test_short_fragments_are_dropped() -> None:
    settings = make_settings(min_chunk_chars=200, fallback_chunk_chars=400, max_chunk_chars=800)
    records = [page("d", 1, "Too short.")]
    chunks, report = chunk_records(records, settings, FakeEmbedder())  # type: ignore[arg-type]
    assert chunks == []
    assert report.dropped_short == 1


def test_overlap_shares_a_margin() -> None:
    pieces = ["First piece ends here.", "Second piece starts here."]
    overlapped = _apply_overlap(pieces, overlap=10)
    assert overlapped[0] == pieces[0]
    assert overlapped[1].startswith("ends here.")
    assert overlapped[1].endswith(pieces[1])


def test_overlap_disabled() -> None:
    pieces = ["a", "b"]
    assert _apply_overlap(pieces, overlap=0) == pieces


def test_overlap_respects_the_size_ceiling() -> None:
    """Regression: overlap was added after the limit was enforced, pushing 25 chunks over."""
    pieces = ["a" * 90, "b" * 95]
    overlapped = _apply_overlap(pieces, overlap=50, ceiling=100)
    assert all(len(piece) <= 100 for piece in overlapped)


def test_contents_page_filler_is_dropped() -> None:
    """Dot leaders embed as plausible prose and match almost any query (feature 4)."""
    from citara.chunking.pipeline import is_low_information

    assert is_low_information("Introduction ................................ 5") is True
    assert is_low_information("......................................") is True
    assert is_low_information("   ") is True
    assert is_low_information("Evacuate residents when the river crosses the danger mark.") is False
    assert is_low_information("| District | Deaths |") is False


def test_noise_chunks_counted_in_the_report() -> None:
    settings = make_settings(min_chunk_chars=20, fallback_chunk_chars=400, max_chunk_chars=800)
    records = [page("d", 1, "Contents ....................................................... 12")]
    chunks, report = chunk_records(records, settings, FakeEmbedder())  # type: ignore[arg-type]
    assert chunks == []
    assert report.dropped_noise == 1


def test_recursive_fallback_is_deterministic() -> None:
    settings = ChunkingSettings(fallback_chunk_chars=100, chunk_overlap_chars=10)
    text = "Sentence number one. " * 30
    assert recursive_split(text, settings) == recursive_split(text, settings)
    assert all(len(p) <= 130 for p in recursive_split(text, settings))


def test_recursive_strategy_recorded_in_report() -> None:
    settings = make_settings(
        strategy="recursive", min_chunk_chars=20, fallback_chunk_chars=120, chunk_overlap_chars=20
    )
    records = [page("d", 1, "Flood response guidance for district officers. " * 10)]
    _, report = chunk_records(records, settings, FakeEmbedder())  # type: ignore[arg-type]
    assert report.recursive_chunks > 0
    assert report.semantic_chunks == 0
    assert report.fallback_documents == ["d.pdf"]


@pytest.mark.parametrize(
    ("kind", "amount"),
    [("percentile", 90.0), ("standard_deviation", 1.0), ("interquartile", 1.5)],
)
def test_threshold_types_produce_a_finite_cut(kind: str, amount: float) -> None:
    settings = ChunkingSettings(
        breakpoint_threshold_type=kind,  # type: ignore[arg-type]
        breakpoint_threshold_amount=amount,
    )
    distances = np.array([0.1, 0.2, 0.9, 0.15, 0.12])
    threshold = breakpoint_threshold(distances, settings)
    assert np.isfinite(threshold)
    assert threshold > distances.min()


def test_empty_distances_never_cut() -> None:
    assert breakpoint_threshold(np.array([]), ChunkingSettings()) == float("inf")


def _chunk(chunk_id: str, content: str, year: int, doc: str = "d") -> Chunk:
    return Chunk(
        chunk_id=chunk_id,
        doc_id=doc,
        title=doc,
        filename=f"{doc}.pdf",
        page_start=1,
        page_end=1,
        total_pages=10,
        year=year,
        ordinal=0,
        content=content,
    )


def test_near_duplicates_removed_keeping_the_newest() -> None:
    """The 2025 and 2026 monsoon advisories repeat paragraphs almost verbatim."""
    text = "Districts must pre-position boats before the monsoon season begins."
    chunks = [
        _chunk("old", text, 2024, doc="advisory-2024"),
        _chunk("new", text, 2026, doc="advisory-2026"),
    ]
    kept, removed, vectors = deduplicate(chunks, FakeEmbedder(), threshold=0.97)  # type: ignore[arg-type]
    assert removed == 1
    assert [c.chunk_id for c in kept] == ["new"]
    assert kept[0].duplicate_ids == ("old",)
    assert vectors.shape[0] == 1


def test_distinct_chunks_survive_deduplication() -> None:
    chunks = [
        _chunk("a", "Evacuation procedures for riverine flooding.", 2026),
        _chunk("b", "Heatwave thresholds for Sindh province.", 2026),
    ]
    kept, removed, _ = deduplicate(chunks, FakeEmbedder(), threshold=0.97)  # type: ignore[arg-type]
    assert removed == 0
    assert len(kept) == 2


def test_deduplication_of_single_chunk_is_a_noop() -> None:
    chunks = [_chunk("only", "text", 2026)]
    kept, removed, vectors = deduplicate(chunks, FakeEmbedder(), 0.97)  # type: ignore[arg-type]
    assert (kept, removed) == (chunks, 0)
    assert vectors.shape[0] == 1


def test_deduplication_returns_vectors_for_reuse() -> None:
    """Indexing reuses these rather than embedding the corpus again (~10 min on CPU)."""
    chunks = [
        _chunk("a", "Evacuation procedures for riverine flooding.", 2026),
        _chunk("b", "Heatwave thresholds for Sindh province.", 2026),
    ]
    kept, _, vectors = deduplicate(chunks, FakeEmbedder(), 0.97)  # type: ignore[arg-type]
    assert vectors.shape[0] == len(kept)


def test_metadata_inherited_from_parent_page() -> None:
    settings = make_settings(min_chunk_chars=10)
    records = [page("ndrp", 47, "Relief camp activation criteria for district authorities. " * 3)]
    chunks, _ = chunk_records(records, settings, FakeEmbedder())  # type: ignore[arg-type]
    chunk = chunks[0]
    assert (chunk.doc_id, chunk.title, chunk.filename) == ("ndrp", "NDRP", "ndrp.pdf")
    assert (chunk.year, chunk.total_pages) == (2026, 50)
    assert chunk.source_record_ids == (records[0].record_id,)


def test_report_totals_are_consistent() -> None:
    settings = make_settings(min_chunk_chars=10)
    records = [
        page("d", 1, "Flood response guidance. " * 8),
        page("d", 2, "| A | B |\n| --- | --- |\n| 1 | 2 |", kind="table"),
    ]
    chunks, report = chunk_records(records, settings, FakeEmbedder())  # type: ignore[arg-type]
    assert report.chunks_out == len(chunks)
    assert report.text_chunks + report.table_chunks == len(chunks)
    assert report.mean_chars > 0
