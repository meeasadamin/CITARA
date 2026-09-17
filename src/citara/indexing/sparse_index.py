"""BM25 sparse index (features 23, 25).

Exists because embeddings are structurally bad at exactly what a disaster officer types:
section numbers, district names, dates, acronyms. A vector compresses meaning and discards
the token; BM25 does the inverse. Building both from the identical chunk list in one pass is
what keeps them from drifting apart.

Persisted as JSON, not a pickle: rebuilding BM25 over a few thousand documents takes
milliseconds, and an index that executes arbitrary code when loaded has no place in a public
repository.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path

from rank_bm25 import BM25Okapi

from citara.chunking.models import Chunk
from citara.log import get_logger

log = get_logger("indexing.sparse")

# Keeps dotted and hyphenated identifiers whole: "4.2", "14.9", "covid-19", "ndma/2026".
_TOKEN = re.compile(r"[a-z0-9]+(?:[./-][a-z0-9]+)*")

_FORMAT_VERSION = 1


def tokenise(text: str) -> list[str]:
    """Lowercase tokens, preserving the identifiers dense retrieval loses."""
    return _TOKEN.findall(text.lower())


@dataclass(frozen=True)
class SparseHit:
    chunk_id: str
    score: float


class SparseIndex:
    """BM25 over the chunk corpus."""

    def __init__(self, chunk_ids: list[str], tokens: list[list[str]]) -> None:
        if len(chunk_ids) != len(tokens):
            raise ValueError("chunk_ids and tokens must be the same length")
        self.chunk_ids = chunk_ids
        self.tokens = tokens
        self._bm25 = BM25Okapi(tokens) if tokens else None

    def __len__(self) -> int:
        return len(self.chunk_ids)

    @classmethod
    def build(cls, chunks: list[Chunk]) -> SparseIndex:
        """Build from the same chunk list the vector store receives."""
        return cls([c.chunk_id for c in chunks], [tokenise(c.content) for c in chunks])

    def search(self, query: str, k: int = 12) -> list[SparseHit]:
        """Top-k chunks by BM25 score, best first. Zero-scoring chunks are not returned."""
        if self._bm25 is None or not query.strip():
            return []
        scores = self._bm25.get_scores(tokenise(query))
        ranked = sorted(enumerate(scores), key=lambda pair: pair[1], reverse=True)[:k]
        return [
            SparseHit(chunk_id=self.chunk_ids[index], score=float(score))
            for index, score in ranked
            if score > 0
        ]

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "version": _FORMAT_VERSION,
            "chunk_ids": self.chunk_ids,
            "tokens": self.tokens,
        }
        path.write_text(json.dumps(payload), encoding="utf-8")
        log.info("sparse index saved", extra={"path": str(path), "documents": len(self)})

    @classmethod
    def load(cls, path: Path) -> SparseIndex:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("version") != _FORMAT_VERSION:
            raise ValueError(f"unsupported sparse index version: {payload.get('version')}")
        return cls(payload["chunk_ids"], payload["tokens"])
