"""Persistent vector store (features 19, 20, 21, 33).

ChromaDB in persistent mode: the index is built once and reloaded from disk on start, so a
user never waits for re-embedding.

Cosine space is configured explicitly rather than left to defaults. That is not pedantry -
the relevance floor that makes refusal real (feature 29) is a threshold on a score, and a
threshold is meaningless if the score's scale is whatever the library happened to pick.

HNSW parameters are tuned for recall on a corpus of a few thousand chunks. Library defaults
optimise for a scale this project does not have.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import chromadb
import numpy as np
from chromadb.api.client import SharedSystemClient  # type: ignore[attr-defined]

from citara.chunking.models import Chunk
from citara.config import Settings, get_settings
from citara.log import get_logger

log = get_logger("indexing.vector")

# Chroma accepts large batches, but smaller ones keep memory flat on a constrained host.
_BATCH = 512
# Sentinel for a document with no publication year: Chroma metadata cannot hold None.
_NO_YEAR = 0


@dataclass(frozen=True)
class DenseHit:
    chunk_id: str
    score: float
    content: str
    metadata: dict[str, Any]


def chunk_metadata(chunk: Chunk) -> dict[str, Any]:
    """Flatten a chunk into Chroma-compatible scalar metadata.

    The rendered citation is stored rather than recomputed downstream: provenance is fixed at
    ingestion and carried forward, never reassembled at output time.
    """
    return {
        "doc_id": chunk.doc_id,
        "title": chunk.title,
        "filename": chunk.filename,
        "page_start": chunk.page_start,
        "page_end": chunk.page_end,
        "total_pages": chunk.total_pages,
        "year": chunk.year if chunk.year is not None else _NO_YEAR,
        "kind": chunk.kind,
        "strategy": chunk.strategy,
        "ordinal": chunk.ordinal,
        "citation": chunk.citation,
        "source_record_ids": ",".join(chunk.source_record_ids),
        "duplicate_ids": ",".join(chunk.duplicate_ids),
    }


class VectorStore:
    """Thin wrapper over one persistent Chroma collection."""

    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()
        config = self.settings.embedding
        path = self.settings.paths.resolved(self.settings.paths.chroma_dir)
        path.mkdir(parents=True, exist_ok=True)
        self.client = chromadb.PersistentClient(path=str(path))
        self.collection = self.client.get_or_create_collection(
            name=config.collection_name,
            configuration={
                "hnsw": {
                    "space": config.hnsw_space,
                    "ef_construction": config.hnsw_construction_ef,
                    "ef_search": config.hnsw_search_ef,
                    "max_neighbors": config.hnsw_m,
                }
            },
        )

    def count(self) -> int:
        return int(self.collection.count())

    def upsert(self, chunks: list[Chunk], vectors: np.ndarray) -> None:
        """Add or replace chunks together with their precomputed embeddings."""
        if len(chunks) != vectors.shape[0]:
            raise ValueError(
                f"chunk/vector mismatch: {len(chunks)} chunks, {vectors.shape[0]} vectors"
            )
        for start in range(0, len(chunks), _BATCH):
            batch = chunks[start : start + _BATCH]
            self.collection.upsert(
                ids=[c.chunk_id for c in batch],
                embeddings=vectors[start : start + _BATCH].tolist(),
                documents=[c.content for c in batch],
                metadatas=[chunk_metadata(c) for c in batch],
            )
        log.info("vectors upserted", extra={"chunks": len(chunks), "total": self.count()})

    def delete_document(self, doc_id: str) -> None:
        """Remove every chunk of one document, for incremental rebuilds (feature 22)."""
        self.collection.delete(where={"doc_id": doc_id})

    def search(
        self, query_vector: np.ndarray, k: int = 12, where: dict[str, Any] | None = None
    ) -> list[DenseHit]:
        """Nearest chunks, best first.

        Chroma returns cosine *distance*; it is converted to similarity here so every score
        in the pipeline means the same thing: higher is better, 1.0 is identical.
        """
        result = self.collection.query(
            query_embeddings=[query_vector.tolist()],
            n_results=k,
            where=where or None,
            include=["documents", "metadatas", "distances"],
        )
        ids = result.get("ids") or [[]]
        if not ids or not ids[0]:
            return []
        documents = (result.get("documents") or [[]])[0]
        metadatas = (result.get("metadatas") or [[]])[0]
        distances = (result.get("distances") or [[]])[0]
        return [
            DenseHit(
                chunk_id=str(chunk_id),
                score=1.0 - float(distance),
                content=str(document),
                metadata=dict(metadata or {}),
            )
            for chunk_id, document, metadata, distance in zip(
                ids[0], documents, metadatas, distances, strict=True
            )
        ]

    def reset(self) -> None:
        """Drop the collection entirely, for a full rebuild."""
        self.client.delete_collection(self.settings.embedding.collection_name)
        self.collection = self.client.get_or_create_collection(
            name=self.settings.embedding.collection_name,
            configuration={
                "hnsw": {
                    "space": self.settings.embedding.hnsw_space,
                    "ef_construction": self.settings.embedding.hnsw_construction_ef,
                    "ef_search": self.settings.embedding.hnsw_search_ef,
                    "max_neighbors": self.settings.embedding.hnsw_m,
                }
            },
        )

    def close(self) -> None:
        """Release Chroma's file handles.

        Chroma exposes no close(); on Windows the open handles block deleting the directory,
        which breaks test teardown and any rebuild that removes the store.
        """
        SharedSystemClient.clear_system_cache()
