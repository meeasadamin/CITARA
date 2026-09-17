"""Typed, centralised configuration (feature 72).

Every tunable in the pipeline lives here, in one validated object. Nothing else in the
codebase reads ``os.environ``. That is what makes the Phase 8 ablation study reproducible:
a run is fully described by this object, and :meth:`Settings.fingerprint` records which
configuration produced which numbers.

Overrides come from the environment or ``.env``, using the nested delimiter::

    CITARA_RETRIEVAL__DENSE_K=16
    CITARA_CHUNKING__MIN_CHUNK_CHARS=200

Provider keys keep their conventional unprefixed names (``GOOGLE_API_KEY``, ``GROQ_API_KEY``)
so Streamlit Community Cloud secrets work without translation.
"""

from __future__ import annotations

import hashlib
from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import (
    AliasChoices,
    BaseModel,
    Field,
    SecretStr,
    computed_field,
    model_validator,
)
from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parents[2]


class Paths(BaseModel):
    """Filesystem layout. Relative paths resolve against the project root."""

    model_config = {"frozen": True}

    docs_dir: Path = Path("docs")
    data_dir: Path = Path("data")

    @computed_field  # type: ignore[prop-decorator]
    @property
    def chroma_dir(self) -> Path:
        """Persistent vector store (feature 20)."""
        return self.data_dir / "chroma"

    @computed_field  # type: ignore[prop-decorator]
    @property
    def bm25_path(self) -> Path:
        """Serialised sparse index, built in the same pass as the dense one (feature 23)."""
        return self.data_dir / "bm25" / "index.pkl"

    @computed_field  # type: ignore[prop-decorator]
    @property
    def manifest_path(self) -> Path:
        """Corpus manifest produced by ingestion (feature 11)."""
        return self.data_dir / "corpus_manifest.json"

    @computed_field  # type: ignore[prop-decorator]
    @property
    def cache_dir(self) -> Path:
        """Response cache (feature 49)."""
        return self.data_dir / "cache"

    def resolved(self, path: Path) -> Path:
        """Absolute form of *path*, anchored at the project root when relative."""
        return path if path.is_absolute() else PROJECT_ROOT / path


class IngestionSettings(BaseModel):
    """PDF extraction and normalisation (features 1-11)."""

    model_config = {"frozen": True}

    sparse_page_chars: int = Field(
        default=100,
        ge=0,
        description="Pages below this many characters are skipped as blank/cover/image-only.",
    )
    scanned_mean_chars: int = Field(
        default=50,
        ge=0,
        description="Mean chars/page below which a document is reported as needing OCR.",
    )
    normalise_ligatures: bool = Field(
        default=True,
        description=(
            "Apply NFKC plus the explicit U+019F->'ti' map. NDMA typesetting extracts as "
            "'NaƟonal'/'eﬀorts'; without this, queries for 'National' cannot match (Phase 0)."
        ),
    )
    dehyphenate: bool = Field(
        default=True, description="Rejoin words broken across line ends (feature 3)."
    )
    extract_tables: bool = Field(default=True, description="Detect and serialise tables (7, 8).")
    collapse_duplicate_table_columns: bool = Field(
        default=True,
        description=(
            "Merged header cells make PyMuPDF emit repeated columns (Col1|Col2|Col3). "
            "Collapse them rather than indexing the noise (Phase 0 finding)."
        ),
    )
    min_table_cells: int = Field(
        default=4,
        ge=1,
        description="Below this cell count a detected table is treated as a false positive.",
    )


class ChunkingSettings(BaseModel):
    """Semantic chunking with a deterministic fallback (features 12-17)."""

    model_config = {"frozen": True}

    strategy: Literal["semantic", "recursive"] = "semantic"
    breakpoint_threshold_type: Literal["percentile", "standard_deviation", "interquartile"] = (
        "percentile"
    )
    breakpoint_threshold_amount: float = Field(default=90.0, gt=0)
    fallback_chunk_chars: int = Field(default=1200, gt=0, description="Recursive splitter size.")
    chunk_overlap_chars: int = Field(default=150, ge=0, description="Overlap window (feature 14).")
    min_chunk_chars: int = Field(
        default=120,
        ge=0,
        description="Shorter fragments are retrieval poison: high similarity, no information.",
    )
    max_chunk_chars: int = Field(
        default=2400, gt=0, description="Hard ceiling; oversized chunks are re-split."
    )
    near_duplicate_threshold: float = Field(
        default=0.97,
        ge=0.0,
        le=1.0,
        description=(
            "Cosine similarity above which a chunk is a near-duplicate. Yearly advisories repeat "
            "text almost verbatim and would otherwise fill the whole evidence set (feature 17)."
        ),
    )

    @model_validator(mode="after")
    def _check_sizes(self) -> ChunkingSettings:
        if self.chunk_overlap_chars >= self.fallback_chunk_chars:
            raise ValueError("chunk_overlap_chars must be smaller than fallback_chunk_chars")
        if self.max_chunk_chars < self.fallback_chunk_chars:
            raise ValueError("max_chunk_chars must be at least fallback_chunk_chars")
        return self


class EmbeddingSettings(BaseModel):
    """Local CPU embedding model and ANN index parameters (features 18-21)."""

    model_config = {"frozen": True}

    model_name: str = "BAAI/bge-small-en-v1.5"
    device: Literal["cpu", "cuda"] = "cpu"
    normalise: bool = Field(
        default=True,
        description="Unit vectors so cosine scores are interpretable, and thus thresholdable.",
    )
    batch_size: int = Field(default=32, gt=0)
    query_prefix: str = Field(
        default="Represent this sentence for searching relevant passages: ",
        description="BGE asymmetric retrieval prefix; applied to queries only, never to documents.",
    )
    collection_name: str = "ndma_corpus"
    hnsw_space: Literal["cosine", "l2", "ip"] = "cosine"
    hnsw_m: int = Field(
        default=32, gt=0, description="Graph connectivity; defaults favour billion-scale, we don't."
    )
    hnsw_construction_ef: int = Field(default=200, gt=0, description="Build-time recall effort.")
    hnsw_search_ef: int = Field(default=128, gt=0, description="Query-time recall effort.")


class RetrievalSettings(BaseModel):
    """Hybrid retrieval, fusion, reranking and the refusal gate (features 24-33)."""

    model_config = {"frozen": True}

    mode: Literal["dense", "sparse", "hybrid", "hybrid_rerank"] = Field(
        default="hybrid_rerank",
        description="The four configurations compared in the ablation table (feature 70).",
    )
    dense_k: int = Field(default=12, gt=0, description="Candidates from vector search.")
    sparse_k: int = Field(default=12, gt=0, description="Candidates from BM25.")
    rrf_k: int = Field(
        default=60, gt=0, description="Reciprocal rank fusion constant: score = w / (rrf_k + rank)."
    )
    dense_weight: float = Field(default=0.5, ge=0.0)
    sparse_weight: float = Field(default=0.5, ge=0.0)
    rerank_candidates: int = Field(
        default=24, gt=0, description="Shortlist size handed to the cross-encoder."
    )
    top_k: int = Field(default=5, gt=0, description="Evidence chunks passed to the LLM.")
    reranker_model: str = "BAAI/bge-reranker-base"
    reranker_batch_size: int = Field(default=16, gt=0)
    relevance_floor: float = Field(
        default=0.10,
        description=(
            "Reranker score below which evidence is discarded; if nothing clears it, the system "
            "refuses (feature 29). PROVISIONAL - Phase 6 must calibrate this on the gold set. "
            "A correctly-retrieved paraphrase measured 0.18, so a naive 0.5 cutoff would refuse "
            "valid questions."
        ),
    )
    use_logit_scores: bool = Field(
        default=False,
        description="Threshold raw cross-encoder logits instead of sigmoid scores.",
    )
    min_evidence_chunks: int = Field(
        default=1, ge=1, description="Chunks that must clear the floor before generation runs."
    )
    history_turns: int = Field(
        default=3,
        ge=0,
        description="Turns fed to query rewriting. Capped to keep it cheap and avoid topic drift.",
    )
    expand_acronyms: bool = Field(
        default=True, description="NDMA, PDMA, NEOC, GLOF... (feature 32)"
    )

    @model_validator(mode="after")
    def _check_funnel(self) -> RetrievalSettings:
        if self.dense_weight + self.sparse_weight <= 0:
            raise ValueError("at least one of dense_weight / sparse_weight must be positive")
        if self.top_k > self.rerank_candidates:
            raise ValueError("top_k cannot exceed rerank_candidates")
        if self.rerank_candidates > self.dense_k + self.sparse_k:
            raise ValueError("rerank_candidates cannot exceed the pooled candidate count")
        return self


class GenerationSettings(BaseModel):
    """Grounded generation and provider failover (features 34-41)."""

    model_config = {"frozen": True}

    primary_model: str = Field(
        default="gemini-3.5-flash-lite",
        description="Gemini 2.0 Flash (original spec choice) was shut down 2026-06-01.",
    )
    fallback_model: str = Field(
        default="openai/gpt-oss-120b",
        description="Groq's replacement for llama-3.3-70b-versatile, retired 2026-08-16.",
    )
    temperature: float = Field(
        default=0.1,
        ge=0.0,
        le=2.0,
        description="A policy assistant must not be creative; same question, same answer.",
    )
    max_output_tokens: int = Field(default=1024, gt=0, description="Token cap (feature 40).")
    stream: bool = Field(default=True, description="Token-by-token rendering (feature 41).")
    request_timeout_s: float = Field(default=60.0, gt=0)
    refusal_message: str = Field(
        default=(
            "I could not find this in the indexed NDMA corpus. I answer only from the "
            "documents listed in the sidebar, and will not answer without supporting evidence."
        )
    )


class GuardrailSettings(BaseModel):
    """Prompt-injection defense and scope control (features 42-47)."""

    model_config = {"frozen": True}

    screen_user_input: bool = True
    screen_retrieved_text: bool = Field(
        default=True,
        description="A corpus document could contain adversarial text; delimit it as data.",
    )
    max_query_chars: int = Field(default=1000, gt=0)
    log_queries: bool = Field(default=True, description="Non-identifying query logging (46).")
    prototype_disclaimer: str = Field(
        default=(
            "Independent decision-support prototype built on publicly available documents. "
            "Not an official NDMA system."
        )
    )


class ResilienceSettings(BaseModel):
    """Quota management and degraded operation (features 48-52)."""

    model_config = {"frozen": True}

    max_retries: int = Field(default=3, ge=0)
    backoff_base_s: float = Field(default=1.0, gt=0, description="Exponential backoff base.")
    backoff_max_s: float = Field(default=20.0, gt=0)
    enable_failover: bool = True
    enable_cache: bool = True
    cache_max_entries: int = Field(default=256, gt=0)
    cache_ttl_s: int = Field(default=86_400, gt=0)
    session_query_cap: int = Field(
        default=40, gt=0, description="Abuse protection on a public URL (feature 52)."
    )
    daily_request_budget: int = Field(
        default=1000, gt=0, description="Tracked against free-tier quota (feature 48)."
    )


class Settings(BaseSettings):
    """The single source of truth for runtime configuration."""

    model_config = SettingsConfigDict(
        env_prefix="CITARA_",
        env_file=".env",
        env_file_encoding="utf-8",
        env_nested_delimiter="__",
        extra="ignore",
        frozen=True,
    )

    app_name: str = "CITARA"
    environment: Literal["local", "cloud"] = "local"
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"
    log_json: bool = Field(
        default=False, description="JSON lines for machine parsing; plain text for humans."
    )

    google_api_key: SecretStr | None = Field(
        default=None,
        validation_alias=AliasChoices("GOOGLE_API_KEY", "CITARA_GOOGLE_API_KEY"),
    )
    groq_api_key: SecretStr | None = Field(
        default=None,
        validation_alias=AliasChoices("GROQ_API_KEY", "CITARA_GROQ_API_KEY"),
    )

    paths: Paths = Paths()
    ingestion: IngestionSettings = IngestionSettings()
    chunking: ChunkingSettings = ChunkingSettings()
    embedding: EmbeddingSettings = EmbeddingSettings()
    retrieval: RetrievalSettings = RetrievalSettings()
    generation: GenerationSettings = GenerationSettings()
    guardrails: GuardrailSettings = GuardrailSettings()
    resilience: ResilienceSettings = ResilienceSettings()

    @property
    def has_primary_provider(self) -> bool:
        """True when the primary LLM can be called at all."""
        return self.google_api_key is not None

    @property
    def has_any_provider(self) -> bool:
        """False means generation is impossible and only degraded mode can serve (feature 51)."""
        return self.google_api_key is not None or self.groq_api_key is not None

    def ensure_directories(self) -> None:
        """Create the writable directories the pipeline expects."""
        for path in (
            self.paths.data_dir,
            self.paths.chroma_dir,
            self.paths.bm25_path.parent,
            self.paths.cache_dir,
        ):
            self.paths.resolved(path).mkdir(parents=True, exist_ok=True)

    def public_dump(self) -> dict[str, object]:
        """Configuration without secrets - safe to log and to store in an evaluation run."""
        return self.model_dump(mode="json", exclude={"google_api_key", "groq_api_key"})

    def fingerprint(self) -> str:
        """Short stable hash of the non-secret configuration.

        Stamped on evaluation runs so an ablation row can never be separated from the
        configuration that produced it (feature 70, and 'documenting experiments').
        """
        payload = repr(sorted(self.public_dump().items())).encode("utf-8")
        return hashlib.sha256(payload).hexdigest()[:12]


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Process-wide settings singleton. Call ``get_settings.cache_clear()`` in tests."""
    return Settings()
