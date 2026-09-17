"""Index construction (features 20-23).

Both indexes are built from one chunk list in a single pass. That is the whole point of
feature 23: if the dense and sparse sides are built separately they will eventually disagree
about what exists, and hybrid fusion will merge two different corpora.

Embeddings are reused from chunking, where deduplication already computed them - about ten
minutes of CPU work on this corpus. Reuse is verified rather than assumed: a sample of stored
vectors is re-embedded and compared, and any mismatch triggers a full recompute.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
from pydantic import BaseModel, Field

from citara.chunking.models import Chunk
from citara.config import Settings, get_settings
from citara.embeddings import Embedder
from citara.indexing.sparse_index import SparseIndex
from citara.indexing.vector_store import VectorStore
from citara.log import get_logger, stage

log = get_logger("indexing")

# Stored vectors are trusted only after this many spot checks agree with a fresh embedding.
_VERIFY_SAMPLES = 3
_VERIFY_TOLERANCE = 0.999


class IndexedDocument(BaseModel):
    doc_id: str
    filename: str
    sha256: str = ""
    chunks: int = 0


class IndexManifest(BaseModel):
    """What is currently indexed, so a rebuild can skip unchanged documents (feature 22)."""

    built_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    config_fingerprint: str = ""
    embedding_model: str = ""
    dimension: int = 0
    chunk_count: int = 0
    documents: list[IndexedDocument] = Field(default_factory=list)

    def by_doc(self) -> dict[str, IndexedDocument]:
        return {d.doc_id: d for d in self.documents}


def load_chunks(path: Path) -> list[Chunk]:
    with path.open("r", encoding="utf-8") as handle:
        return [Chunk.model_validate_json(line) for line in handle if line.strip()]


def load_vectors(path: Path, chunks: list[Chunk], embedder: Embedder) -> tuple[np.ndarray, bool]:
    """Load chunk embeddings, verifying they still match the chunks.

    Returns the vectors and whether they had to be recomputed. A stale or misaligned vector
    file would attach every chunk to the wrong point in the space - retrieval would keep
    working, and keep being wrong.
    """
    if path.exists():
        vectors = np.load(path)
        if vectors.shape == (len(chunks), embedder.dimension):
            step = max(1, len(chunks) // _VERIFY_SAMPLES)
            sample = list(range(0, len(chunks), step))[:_VERIFY_SAMPLES]
            fresh = embedder.embed_documents([chunks[i].content for i in sample])
            agreement = [float(fresh[n] @ vectors[i]) for n, i in enumerate(sample)]
            if all(score >= _VERIFY_TOLERANCE for score in agreement):
                log.info(
                    "reusing chunk embeddings",
                    extra={"path": str(path), "verified": len(sample)},
                )
                return vectors.astype(np.float32), False
            log.warning(
                "stored embeddings do not match the chunks; recomputing",
                extra={"agreement": [round(a, 4) for a in agreement]},
            )
        else:
            log.warning(
                "stored embeddings have the wrong shape; recomputing",
                extra={
                    "stored": list(vectors.shape),
                    "expected": [len(chunks), embedder.dimension],
                },
            )

    vectors = embedder.embed_documents([c.content for c in chunks])
    return vectors, True


def build_index(
    settings: Settings | None = None,
    *,
    rebuild: bool = False,
    embedder: Embedder | None = None,
) -> IndexManifest:
    """Build or refresh both indexes from the chunk store."""
    settings = settings or get_settings()
    embedder = embedder or Embedder(settings.embedding)
    data_dir = settings.paths.resolved(settings.paths.data_dir)

    chunks = load_chunks(data_dir / "chunks.jsonl")
    if not chunks:
        raise ValueError("no chunks found; run: uv run python -m citara.chunking")

    vectors, recomputed = load_vectors(data_dir / "chunk_vectors.npy", chunks, embedder)

    corpus_manifest_path = settings.paths.resolved(settings.paths.manifest_path)
    digests: dict[str, str] = {}
    if corpus_manifest_path.exists():
        payload = json.loads(corpus_manifest_path.read_text(encoding="utf-8"))
        digests = {d["doc_id"]: d.get("sha256", "") for d in payload.get("documents", [])}

    manifest_path = settings.paths.resolved(settings.paths.index_manifest_path)
    previous = IndexManifest()
    if manifest_path.exists() and not rebuild:
        previous = IndexManifest.model_validate_json(manifest_path.read_text(encoding="utf-8"))

    store = VectorStore(settings)
    with stage(log, "build_index", chunks=len(chunks), rebuild=rebuild) as details:
        if rebuild or previous.embedding_model not in ("", settings.embedding.model_name):
            store.reset()
            previous = IndexManifest()

        known = previous.by_doc()
        by_doc: dict[str, list[int]] = {}
        for index, chunk in enumerate(chunks):
            by_doc.setdefault(chunk.doc_id, []).append(index)

        changed: list[str] = []
        for doc_id, indices in by_doc.items():
            digest = digests.get(doc_id, "")
            existing = known.get(doc_id)
            unchanged = (
                existing is not None
                and existing.sha256 == digest
                and existing.chunks == len(indices)
                and not rebuild
            )
            if unchanged:
                continue
            changed.append(doc_id)
            if existing is not None:
                store.delete_document(doc_id)
            selection = np.array(indices)
            store.upsert([chunks[i] for i in indices], vectors[selection])

        # Documents dropped from the corpus must not linger in the index.
        removed = [doc_id for doc_id in known if doc_id not in by_doc]
        for doc_id in removed:
            store.delete_document(doc_id)

        # BM25 is always rebuilt from the full chunk list: it costs milliseconds, and
        # rebuilding is what guarantees the two indexes describe the same corpus.
        sparse = SparseIndex.build(chunks)
        sparse.save(settings.paths.resolved(settings.paths.bm25_path))

        manifest = IndexManifest(
            config_fingerprint=settings.fingerprint(),
            embedding_model=settings.embedding.model_name,
            dimension=embedder.dimension,
            chunk_count=len(chunks),
            documents=[
                IndexedDocument(
                    doc_id=doc_id,
                    filename=chunks[indices[0]].filename,
                    sha256=digests.get(doc_id, ""),
                    chunks=len(indices),
                )
                for doc_id, indices in by_doc.items()
            ],
        )
        manifest_path.write_text(manifest.model_dump_json(indent=2), encoding="utf-8")

        dense_count = store.count()
        details.update(
            {
                "documents": len(by_doc),
                "documents_reindexed": len(changed),
                "documents_removed": len(removed),
                "dense_vectors": dense_count,
                "sparse_documents": len(sparse),
                "embeddings_recomputed": recomputed,
            }
        )

        if not (dense_count == len(sparse) == len(chunks)):
            raise RuntimeError(
                "index parity check failed: "
                f"dense={dense_count}, sparse={len(sparse)}, chunks={len(chunks)}"
            )

    store.close()
    return manifest
