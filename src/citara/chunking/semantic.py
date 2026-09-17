"""Semantic chunking with a deterministic fallback (features 12, 13, 16).

Text is split where the topic actually shifts - at large distances between the embeddings of
neighbouring sentences - rather than at arbitrary character counts. The failure this prevents
is concrete for this corpus: a five-step evacuation SOP cut in half returns half a life-safety
procedure, which is worse than returning nothing.

Implemented here rather than taken from ``langchain-experimental``, which is sunset and
unmaintained, and which cannot enforce a maximum chunk size or report which strategy produced
a chunk. The algorithm is the standard one: embed sentences, measure consecutive cosine
distance, cut where the distance exceeds a threshold derived from the document's own
distribution.
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

import numpy as np
from langchain_text_splitters import RecursiveCharacterTextSplitter

from citara.config import ChunkingSettings
from citara.log import get_logger

if TYPE_CHECKING:  # pragma: no cover
    from citara.embeddings import Embedder

log = get_logger("chunking.semantic")

# Sentence boundary: terminator followed by whitespace, or a blank line between paragraphs.
_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+|\n{2,}")


def split_sentences(text: str) -> list[str]:
    """Split text into sentence-sized units, preserving order and content."""
    parts = [part.strip() for part in _SENTENCE_SPLIT.split(text) if part and part.strip()]
    return parts


def breakpoint_threshold(distances: np.ndarray, settings: ChunkingSettings) -> float:
    """Distance above which a sentence boundary becomes a chunk boundary.

    Derived from the document's own distance distribution, so a densely written plan and a
    loosely written advisory are each split where *they* change topic.
    """
    if distances.size == 0:
        return float("inf")
    amount = settings.breakpoint_threshold_amount
    if settings.breakpoint_threshold_type == "percentile":
        return float(np.percentile(distances, amount))
    if settings.breakpoint_threshold_type == "standard_deviation":
        return float(distances.mean() + amount * distances.std())
    q1, q3 = np.percentile(distances, [25, 75])
    return float(q3 + amount * (q3 - q1))


def recursive_split(text: str, settings: ChunkingSettings) -> list[str]:
    """Deterministic character splitting: the boring backup that always works (feature 13)."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.fallback_chunk_chars,
        chunk_overlap=settings.chunk_overlap_chars,
        separators=["\n\n", "\n", ". ", " ", ""],
        length_function=len,
    )
    return [part for part in splitter.split_text(text) if part.strip()]


def enforce_max_size(pieces: list[str], settings: ChunkingSettings) -> tuple[list[str], int]:
    """Re-split anything above the hard ceiling, returning the pieces and how many were split."""
    result: list[str] = []
    resplit = 0
    for piece in pieces:
        if len(piece) <= settings.max_chunk_chars:
            result.append(piece)
            continue
        resplit += 1
        result.extend(recursive_split(piece, settings))
    return result, resplit


def semantic_split(
    text: str, embedder: Embedder, settings: ChunkingSettings
) -> tuple[list[str], str]:
    """Split *text* at topic shifts.

    Returns the pieces and the strategy actually used, so the report can show how often the
    fallback engaged rather than assuming the primary path always worked.
    """
    if len(text) <= settings.fallback_chunk_chars:
        return ([text] if text.strip() else []), "semantic"

    sentences = split_sentences(text)
    if len(sentences) < 3:
        return recursive_split(text, settings), "recursive"

    try:
        vectors = embedder.embed_documents(sentences)
        # Vectors are unit-normalised, so 1 - dot product is cosine distance.
        distances = 1.0 - np.sum(vectors[:-1] * vectors[1:], axis=1)
    except Exception:
        log.exception("semantic split failed; using recursive fallback")
        return recursive_split(text, settings), "recursive"

    threshold = breakpoint_threshold(distances, settings)
    pieces: list[str] = []
    current: list[str] = []
    for index, sentence in enumerate(sentences):
        current.append(sentence)
        at_break = index < len(distances) and distances[index] > threshold
        long_enough = sum(len(s) + 1 for s in current) >= settings.min_chunk_chars
        if at_break and long_enough:
            pieces.append(" ".join(current))
            current = []
    if current:
        pieces.append(" ".join(current))

    return pieces, "semantic"
