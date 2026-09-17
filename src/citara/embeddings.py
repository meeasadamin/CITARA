"""Local embedding model, shared by chunking, indexing and retrieval (feature 18).

Loaded once per process and cached: the model costs seconds to load and ~130 MB of RAM, and
Streamlit re-runs the script on every interaction.

Runs on CPU with no API key, so document text never leaves the machine during indexing - an
economic choice, and a data-sovereignty argument that can be made directly to a government
reviewer.
"""

from __future__ import annotations

from functools import lru_cache

import numpy as np
from sentence_transformers import SentenceTransformer

from citara.config import EmbeddingSettings, get_settings
from citara.log import get_logger

log = get_logger("embeddings")


@lru_cache(maxsize=2)
def load_model(model_name: str, device: str) -> SentenceTransformer:
    """Load and cache a sentence-transformer model."""
    log.info("loading embedding model", extra={"model": model_name, "device": device})
    return SentenceTransformer(model_name, device=device)


class Embedder:
    """Thin wrapper applying this project's embedding conventions.

    BGE is an asymmetric retrieval model: queries carry an instruction prefix, documents do
    not. Getting that backwards quietly degrades every search, so the distinction is encoded
    here once rather than left to each call site.
    """

    def __init__(self, settings: EmbeddingSettings | None = None) -> None:
        self.settings = settings or get_settings().embedding
        self.model = load_model(self.settings.model_name, self.settings.device)

    @property
    def dimension(self) -> int:
        return int(self.model.get_sentence_embedding_dimension() or 0)

    def embed_documents(self, texts: list[str]) -> np.ndarray:
        """Embed passages. No prefix: documents are the indexed side of the pair."""
        if not texts:
            return np.zeros((0, self.dimension), dtype=np.float32)
        vectors = self.model.encode(
            texts,
            batch_size=self.settings.batch_size,
            normalize_embeddings=self.settings.normalise,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        return np.asarray(vectors, dtype=np.float32)

    def embed_query(self, text: str) -> np.ndarray:
        """Embed a question, with the BGE retrieval instruction prefix."""
        vector: np.ndarray = self.embed_documents([self.settings.query_prefix + text])[0]
        return vector
