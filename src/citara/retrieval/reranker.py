"""Cross-encoder reranking (features 27, 28).

A bi-encoder embeds the question and the chunk separately and compares two vectors that never
met. A cross-encoder reads both together in one forward pass, so it can judge whether *this*
passage answers *this* question rather than whether they occupy similar regions of a vector
space. It is far more accurate and far more expensive, which is exactly why it runs on a
shortlist of roughly two dozen candidates instead of on the corpus.

That cost asymmetry is the whole argument for the two-stage funnel: cheap retrieval casts the
net wide, expensive reranking decides what the model actually sees.
"""

from __future__ import annotations

from functools import lru_cache

from sentence_transformers import CrossEncoder

from citara.config import RetrievalSettings
from citara.log import get_logger
from citara.retrieval.models import RetrievedChunk

log = get_logger("retrieval.rerank")

# The cross-encoder truncates beyond this many tokens; chunk sizes are set so that the
# evidence a chunk carries survives truncation rather than being silently cut.
_MAX_LENGTH = 512


@lru_cache(maxsize=1)
def load_reranker(model_name: str) -> CrossEncoder:
    """Load and cache the cross-encoder.

    Roughly a gigabyte of resident memory, so it is loaded once per process, and only on first
    use or an explicit ``warm_up()`` - never at import or construction, which would make every
    script and test that touches retrieval pay for a model it may not run.
    """
    log.info("loading reranker", extra={"model": model_name})
    model: CrossEncoder = CrossEncoder(model_name, max_length=_MAX_LENGTH)
    return model


class CrossEncoderReranker:
    """Re-scores candidates by reading question and passage together."""

    def __init__(self, settings: RetrievalSettings) -> None:
        self.settings = settings

    def warm_up(self) -> None:
        """Load the model and run one pass, so the first real question pays for neither."""
        model = load_reranker(self.settings.reranker_model)
        model.predict([("warm up", "warm up")], show_progress_bar=False)

    def rerank(self, query: str, candidates: list[RetrievedChunk]) -> list[RetrievedChunk]:
        """Attach ``rerank_score`` to each candidate and return them best first."""
        if not candidates:
            return []

        model = load_reranker(self.settings.reranker_model)
        pairs = [(query, candidate.chunk.content) for candidate in candidates]
        scores = model.predict(
            pairs,
            batch_size=self.settings.reranker_batch_size,
            show_progress_bar=False,
        )
        for candidate, score in zip(candidates, scores, strict=True):
            candidate.rerank_score = float(score)

        return sorted(candidates, key=lambda item: (-(item.rerank_score or 0.0), item.chunk_id))
