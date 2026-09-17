"""Tests for dense and sparse indexing (features 19-23)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from citara.chunking.models import Chunk
from citara.config import EmbeddingSettings, Paths, Settings
from citara.indexing.builder import build_index, load_vectors
from citara.indexing.sparse_index import SparseIndex, tokenise
from citara.indexing.vector_store import VectorStore, chunk_metadata


class FakeEmbedder:
    """Token-overlap embeddings: deterministic, and similar text lands nearby."""

    dimension = 32

    def embed_documents(self, texts: list[str]) -> np.ndarray:
        vectors = np.zeros((len(texts), self.dimension), dtype=np.float32)
        for row, text in enumerate(texts):
            for token in tokenise(text):
                vectors[row, hash(token) % self.dimension] += 1.0
            norm = np.linalg.norm(vectors[row]) or 1.0
            vectors[row] /= norm
        return vectors

    def embed_query(self, text: str) -> np.ndarray:
        return self.embed_documents([text])[0]


def make_chunk(chunk_id: str, content: str, doc: str = "ndrp", page: int = 1) -> Chunk:
    return Chunk(
        chunk_id=chunk_id,
        doc_id=doc,
        title=doc.upper(),
        filename=f"{doc}.pdf",
        page_start=page,
        page_end=page,
        total_pages=110,
        year=2019,
        ordinal=page,
        content=content,
        source_record_ids=("r1",),
    )


def settings_for(tmp_path: Path) -> Settings:
    return Settings(
        _env_file=None,  # type: ignore[call-arg]
        paths=Paths(docs_dir=tmp_path / "docs", data_dir=tmp_path / "data"),
        embedding=EmbeddingSettings(collection_name="test_corpus"),
    )


def write_corpus(tmp_path: Path, chunks: list[Chunk], vectors: np.ndarray | None = None) -> Path:
    data = tmp_path / "data"
    data.mkdir(parents=True, exist_ok=True)
    with (data / "chunks.jsonl").open("w", encoding="utf-8") as handle:
        for chunk in chunks:
            handle.write(chunk.model_dump_json() + "\n")
    if vectors is not None:
        np.save(data / "chunk_vectors.npy", vectors)
    manifest = {
        "documents": [{"doc_id": doc, "sha256": f"sha-{doc}"} for doc in {c.doc_id for c in chunks}]
    }
    (data / "corpus_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    return data


# --- sparse index -----------------------------------------------------------------


def test_tokeniser_preserves_identifiers() -> None:
    """The exact tokens embeddings lose: section numbers, figures, coded names."""
    assert tokenise("Section 4.2 covers Dadu") == ["section", "4.2", "covers", "dadu"]
    assert "14.9" in tokenise("USD 14.9 billion in damages")
    assert "covid-19" in tokenise("COVID-19 response")


def filler_chunks(count: int = 8) -> list[Chunk]:
    """Background corpus.

    BM25 inverse document frequency is zero for a term present in half the corpus, so a
    two-document test would score everything zero and prove nothing about ranking.
    """
    topics = [
        "General disaster management coordination arrangements.",
        "Provincial authority responsibilities during response.",
        "Relief goods logistics and warehousing guidance.",
        "Early warning dissemination to union councils.",
        "Post-disaster damage assessment methodology.",
        "Community awareness and preparedness campaigns.",
        "Emergency operations centre staffing rosters.",
        "Standard operating procedures for rescue teams.",
    ]
    return [make_chunk(f"f{i}", topics[i % len(topics)], page=i + 10) for i in range(count)]


def test_sparse_search_finds_exact_identifiers() -> None:
    """The case dense retrieval structurally loses: a district name and a section number."""
    chunks = [
        *filler_chunks(),
        make_chunk("a", "Evacuation protocol for inundation events in riverine areas."),
        make_chunk("b", "Section 4.2 assigns relief camp duties to the Dadu district."),
    ]
    index = SparseIndex.build(chunks)
    assert index.search("Dadu", k=5)[0].chunk_id == "b"
    assert index.search("4.2", k=5)[0].chunk_id == "b"


def test_sparse_search_handles_empty_and_unmatched_queries() -> None:
    index = SparseIndex.build([make_chunk("a", "Flood response guidance for districts.")])
    assert index.search("") == []
    assert index.search("zzzz-nonexistent-token") == []


def test_sparse_index_roundtrip_without_pickle(tmp_path: Path) -> None:
    chunks = [
        *filler_chunks(),
        make_chunk("a", "Heatwave thresholds for Sindh province in summer."),
        make_chunk("b", "Flood warning sirens at district headquarters."),
    ]
    path = tmp_path / "bm25" / "index.json"
    SparseIndex.build(chunks).save(path)

    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["version"] == 1  # plain JSON, nothing executable

    restored = SparseIndex.load(path)
    assert restored.chunk_ids[-2:] == ["a", "b"]
    assert restored.search("heatwave")[0].chunk_id == "a"


def test_sparse_index_rejects_mismatched_lengths() -> None:
    with pytest.raises(ValueError, match="same length"):
        SparseIndex(["a", "b"], [["only-one"]])


# --- vector store -----------------------------------------------------------------


def test_metadata_is_scalar_and_carries_the_citation() -> None:
    chunk = make_chunk("a", "text", page=47)
    metadata = chunk_metadata(chunk)
    assert metadata["citation"] == "NDRP, p. 47"
    assert all(isinstance(v, str | int | float | bool) for v in metadata.values())


def test_missing_year_becomes_a_sentinel_not_none() -> None:
    """Chroma metadata cannot hold None, and a dropped key breaks year filtering."""
    chunk = make_chunk("a", "text").model_copy(update={"year": None})
    assert chunk_metadata(chunk)["year"] == 0


def test_vector_store_roundtrip_and_cosine_scores(tmp_path: Path) -> None:
    settings = settings_for(tmp_path)
    embedder = FakeEmbedder()
    chunks = [
        make_chunk("a", "Evacuation protocol for inundation events."),
        make_chunk("b", "Heatwave thresholds for southern districts.", page=2),
    ]
    vectors = embedder.embed_documents([c.content for c in chunks])

    store = VectorStore(settings)
    try:
        store.upsert(chunks, vectors)
        assert store.count() == 2

        hits = store.search(embedder.embed_query("Evacuation protocol for inundation"), k=2)
        assert hits[0].chunk_id == "a"
        assert 0.0 <= hits[0].score <= 1.0001  # similarity, not distance
        assert hits[0].score > hits[1].score
        assert hits[0].metadata["citation"] == "NDRP, p. 1"
    finally:
        store.close()


def test_vector_store_metadata_filter(tmp_path: Path) -> None:
    """Document scoping is applied before search, not after (feature 33)."""
    settings = settings_for(tmp_path)
    embedder = FakeEmbedder()
    chunks = [
        make_chunk("a", "Flood evacuation guidance.", doc="ndrp"),
        make_chunk("b", "Flood evacuation guidance.", doc="monsoon"),
    ]
    vectors = embedder.embed_documents([c.content for c in chunks])

    store = VectorStore(settings)
    try:
        store.upsert(chunks, vectors)
        hits = store.search(
            embedder.embed_query("flood evacuation"), k=5, where={"doc_id": "monsoon"}
        )
        assert [h.chunk_id for h in hits] == ["b"]
    finally:
        store.close()


def test_upsert_rejects_mismatched_vectors(tmp_path: Path) -> None:
    settings = settings_for(tmp_path)
    store = VectorStore(settings)
    try:
        with pytest.raises(ValueError, match="mismatch"):
            store.upsert([make_chunk("a", "text")], np.zeros((2, 32), dtype=np.float32))
    finally:
        store.close()


# --- embedding reuse --------------------------------------------------------------


def test_stored_vectors_are_reused_when_they_match(tmp_path: Path) -> None:
    embedder = FakeEmbedder()
    chunks = [make_chunk("a", "Flood response."), make_chunk("b", "Heat response.")]
    vectors = embedder.embed_documents([c.content for c in chunks])
    path = tmp_path / "chunk_vectors.npy"
    np.save(path, vectors)

    loaded, recomputed = load_vectors(path, chunks, embedder)  # type: ignore[arg-type]
    assert recomputed is False
    assert np.allclose(loaded, vectors)


def test_stale_vectors_are_recomputed(tmp_path: Path) -> None:
    """A vector file left over from different chunks would attach every chunk to the wrong point."""
    embedder = FakeEmbedder()
    chunks = [make_chunk("a", "Flood response."), make_chunk("b", "Heat response.")]
    stale = embedder.embed_documents(["something else entirely", "and another thing"])
    path = tmp_path / "chunk_vectors.npy"
    np.save(path, stale)

    loaded, recomputed = load_vectors(path, chunks, embedder)  # type: ignore[arg-type]
    assert recomputed is True
    assert not np.allclose(loaded, stale)


def test_wrong_shape_vectors_are_recomputed(tmp_path: Path) -> None:
    embedder = FakeEmbedder()
    chunks = [make_chunk("a", "Flood response.")]
    path = tmp_path / "chunk_vectors.npy"
    np.save(path, np.zeros((5, 32), dtype=np.float32))

    _, recomputed = load_vectors(path, chunks, embedder)  # type: ignore[arg-type]
    assert recomputed is True


# --- build orchestration ----------------------------------------------------------


def test_build_creates_both_indexes_in_sync(tmp_path: Path) -> None:
    settings = settings_for(tmp_path)
    embedder = FakeEmbedder()
    chunks = [make_chunk(f"c{i}", f"Flood guidance number {i}.", page=i + 1) for i in range(5)]
    vectors = embedder.embed_documents([c.content for c in chunks])
    write_corpus(tmp_path, chunks, vectors)

    manifest = build_index(settings, rebuild=True, embedder=embedder)  # type: ignore[arg-type]
    assert manifest.chunk_count == 5
    assert manifest.dimension == 32

    sparse = SparseIndex.load(settings.paths.resolved(settings.paths.bm25_path))
    store = VectorStore(settings)
    try:
        assert store.count() == len(sparse) == 5
    finally:
        store.close()


def test_incremental_build_skips_unchanged_documents(tmp_path: Path) -> None:
    settings = settings_for(tmp_path)
    embedder = FakeEmbedder()
    chunks = [make_chunk("a", "Flood guidance.", doc="ndrp")]
    vectors = embedder.embed_documents([c.content for c in chunks])
    write_corpus(tmp_path, chunks, vectors)
    build_index(settings, rebuild=True, embedder=embedder)  # type: ignore[arg-type]

    # Same corpus, unchanged digests: nothing should be re-indexed, and nothing lost.
    manifest = build_index(settings, embedder=embedder)  # type: ignore[arg-type]
    store = VectorStore(settings)
    try:
        assert store.count() == 1
    finally:
        store.close()
    assert manifest.chunk_count == 1


def test_new_document_is_added_incrementally(tmp_path: Path) -> None:
    settings = settings_for(tmp_path)
    embedder = FakeEmbedder()
    first = [make_chunk("a", "Flood guidance.", doc="ndrp")]
    write_corpus(tmp_path, first, embedder.embed_documents([c.content for c in first]))
    build_index(settings, rebuild=True, embedder=embedder)  # type: ignore[arg-type]

    both = [*first, make_chunk("b", "Monsoon advisory guidance.", doc="monsoon")]
    write_corpus(tmp_path, both, embedder.embed_documents([c.content for c in both]))
    manifest = build_index(settings, embedder=embedder)  # type: ignore[arg-type]

    assert manifest.chunk_count == 2
    store = VectorStore(settings)
    try:
        assert store.count() == 2
    finally:
        store.close()


def test_removed_document_is_dropped_from_the_index(tmp_path: Path) -> None:
    settings = settings_for(tmp_path)
    embedder = FakeEmbedder()
    both = [
        make_chunk("a", "Flood guidance.", doc="ndrp"),
        make_chunk("b", "Monsoon advisory.", doc="monsoon"),
    ]
    write_corpus(tmp_path, both, embedder.embed_documents([c.content for c in both]))
    build_index(settings, rebuild=True, embedder=embedder)  # type: ignore[arg-type]

    remaining = [both[0]]
    write_corpus(tmp_path, remaining, embedder.embed_documents([c.content for c in remaining]))
    build_index(settings, embedder=embedder)  # type: ignore[arg-type]

    store = VectorStore(settings)
    try:
        assert store.count() == 1
    finally:
        store.close()


def test_build_without_chunks_fails_clearly(tmp_path: Path) -> None:
    settings = settings_for(tmp_path)
    (tmp_path / "data").mkdir(parents=True, exist_ok=True)
    (tmp_path / "data" / "chunks.jsonl").write_text("", encoding="utf-8")
    with pytest.raises(ValueError, match="no chunks"):
        build_index(settings, embedder=FakeEmbedder())  # type: ignore[arg-type]
