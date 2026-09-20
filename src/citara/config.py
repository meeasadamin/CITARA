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
    field_validator,
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
        """Sparse index, built in the same pass as the dense one (feature 23).

        JSON rather than a pickle: rebuilding BM25 over a few thousand documents costs
        milliseconds, and an index file that executes code on load is not something to ship
        in a public repository.
        """
        return self.data_dir / "bm25" / "index.json"

    @computed_field  # type: ignore[prop-decorator]
    @property
    def index_manifest_path(self) -> Path:
        """What is currently indexed, enabling incremental rebuilds (feature 22)."""
        return self.data_dir / "index_manifest.json"

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

    @computed_field  # type: ignore[prop-decorator]
    @property
    def usage_path(self) -> Path:
        """Daily request counts, persisted so a restart does not reset the budget."""
        return self.data_dir / "usage.json"

    @computed_field  # type: ignore[prop-decorator]
    @property
    def query_log_path(self) -> Path:
        """Non-identifying query log kept for evaluation (feature 46)."""
        return self.data_dir / "queries.jsonl"

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
    dense_weight: float = Field(
        default=1.0,
        ge=0.0,
        description=(
            "Fusion weight for vector search. Set from the ablation, not from the spec: on "
            "the current gold set dense retrieval alone reaches Hit@5 0.458 while an even "
            "hybrid reaches 0.375, so BM25 is weighted out of the ranking by default."
        ),
    )
    sparse_weight: float = Field(
        default=0.0,
        ge=0.0,
        description=(
            "Fusion weight for BM25. Zero by default because the measurement does not yet "
            "support it - not because sparse retrieval is useless. BM25 still runs and its "
            "candidates stay in the pool, and it remains the only retriever that can match an "
            "exact section number or phone number. Raise this and re-run the ablation once "
            "the gold set is larger than 24 questions, where a two-question swing is noise."
        ),
    )
    rerank_candidates: int = Field(
        default=24, gt=0, description="Shortlist size handed to the cross-encoder."
    )
    top_k: int = Field(default=5, gt=0, description="Evidence chunks passed to the LLM.")
    reranker_model: str = "BAAI/bge-reranker-base"
    reranker_batch_size: int = Field(default=16, gt=0)
    relevance_floor: float = Field(
        default=0.3508,
        description=(
            "Reranker score below which evidence is discarded; if nothing clears it, the "
            "system refuses (feature 29). CALIBRATED on the gold set rather than chosen, by "
            "ordering the errors: first admit no unanswerable question, then refuse as few "
            "answerable ones as possible. It answers 16 of 24 answerable questions and none of "
            "the 6 unanswerable; 3 refusals are real losses (the right page was retrieved but "
            "scored low). The value is a midpoint between observed scores, not a round number, "
            "because admission uses >= and a threshold on an observed score admits it. Re-run "
            "scripts/calibrate_floor.py after any change to chunking, retrieval or the "
            "reranker; see eval/runs/floor_calibration.json and eval/RESULTS.md."
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
        description=(
            "A policy assistant must not be creative; same question, same answer. NOTE: "
            "gemini-3.5-flash-lite reports that it uses fixed sampling defaults and ignores "
            "this value, so determinism there rests on the model's own defaults rather than "
            "on configuration. The setting still applies to the Groq failover and to any "
            "primary model that honours it."
        ),
    )
    max_output_tokens: int = Field(default=1024, gt=0, description="Token cap (feature 40).")
    stream: bool = Field(default=True, description="Token-by-token rendering (feature 41).")
    request_timeout_s: float = Field(
        default=30.0,
        gt=0,
        description=(
            "Per request. A capped answer takes seconds; a request still open at 30 s belongs "
            "to a struggling provider, and the question is better served by failover."
        ),
    )
    rewrite_timeout_s: float = Field(
        default=10.0,
        gt=0,
        description="Follow-up rewriting is a short request; past this the heuristic is used.",
    )
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
    max_query_chars: int = Field(
        default=1000, gt=0, description="Enforced in preflight and by the chat input box."
    )
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
    retry_budget_s: float = Field(
        default=30.0,
        gt=0,
        description="Total time one call may spend retrying before failover takes over.",
    )
    quota_cooldown_s: int = Field(
        default=3600,
        gt=0,
        description=(
            "How long a provider that reported its daily quota spent is skipped before being "
            "tried again, so every later question does not pay for rediscovering it."
        ),
    )
    enable_failover: bool = True
    enable_cache: bool = True
    cache_max_entries: int = Field(default=256, gt=0)
    cache_ttl_s: int = Field(default=86_400, gt=0)
    session_query_cap: int = Field(
        default=40, gt=0, description="Abuse protection on a public URL (feature 52)."
    )
    daily_request_budget: int = Field(
        default=1000,
        gt=0,
        description=(
            "The deployment's own daily spend cap across all providers (feature 48). Each "
            "provider's real quota is learned from its rate-limit responses instead."
        ),
    )


class UiSettings(BaseModel):
    """Interface presentation (features 53-65)."""

    model_config = {"frozen": True}

    page_title: str = "CITARA — NDMA Disaster Doctrine Assistant"
    evidence_strength_high: float = Field(
        default=0.80,
        ge=0.0,
        description=(
            "Reranker score at or above which evidence reads as High (feature 56). Calibrated on "
            "the gold set: 8 of the 16 admitted answerable questions score 0.80 or more."
        ),
    )
    evidence_strength_moderate: float = Field(
        default=0.50,
        ge=0.0,
        description=(
            "At or above this, Moderate (6 of the 16 admitted); below it, Low - evidence that "
            "cleared the relevance floor only narrowly, where the sources deserve a check (the "
            "other 2). A band, never a percentage."
        ),
    )
    show_latency: bool = Field(default=True, description="Per-answer timings (feature 62).")
    max_history_messages: int = Field(default=50, gt=0)
    transcript_prefix: str = Field(default="citara-transcript", description="Export filename (63).")

    @model_validator(mode="after")
    def _check_bands(self) -> UiSettings:
        if self.evidence_strength_high <= self.evidence_strength_moderate:
            raise ValueError("evidence_strength_high must exceed evidence_strength_moderate")
        return self


class EvaluationSettings(BaseModel):
    """Offline evaluation harness (features 66-71)."""

    model_config = {"frozen": True}

    gold_set_path: Path = Path("eval/gold_questions.json")
    runs_dir: Path = Path("eval/runs")
    hit_rate_k: tuple[int, ...] = Field(
        default=(1, 3, 5), description="k values reported for Hit Rate@k (feature 67)."
    )
    ablation_modes: tuple[str, ...] = Field(
        default=("dense", "sparse", "hybrid", "hybrid_rerank"),
        description="The four rows of the ablation table (feature 70).",
    )
    latency_percentiles: tuple[float, ...] = Field(
        default=(50.0, 95.0), description="Reported separately for retrieval and generation (71)."
    )
    judge_model: str = Field(
        default="gemini-3.5-flash-lite",
        description="Scores faithfulness and answer relevance (features 68, 69).",
    )
    judge_temperature: float = Field(default=0.0, ge=0.0, le=1.0)

    @model_validator(mode="after")
    def _check_k(self) -> EvaluationSettings:
        if not self.hit_rate_k or min(self.hit_rate_k) < 1:
            raise ValueError("hit_rate_k must contain positive values")
        return self


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

    index_url: str | None = Field(
        default=(
            "https://github.com/meeasadamin/CITARA/releases/download/index-v1/citara-index.tar.gz"
        ),
        description=(
            "The published index, fetched on first start when the data directory has none "
            "(the repository ships no index and no PDFs to rebuild one from). Set to an "
            "empty value to disable fetching and rely on a locally built index."
        ),
    )
    index_sha256: str | None = Field(
        default="a66ff311b59d1189c389f4893766ff462932cf478ad401038613aa85a793c1db",
        description=(
            "Checksum the downloaded archive must match. Without it a corrupted or "
            "substituted asset would be indexed and answered from. Printed by "
            "scripts/package_index.py; update both together."
        ),
    )

    google_api_key: SecretStr | None = Field(
        default=None,
        validation_alias=AliasChoices("GOOGLE_API_KEY", "CITARA_GOOGLE_API_KEY"),
    )
    groq_api_key: SecretStr | None = Field(
        default=None,
        validation_alias=AliasChoices("GROQ_API_KEY", "CITARA_GROQ_API_KEY"),
    )

    @field_validator("index_url", "index_sha256", mode="before")
    @classmethod
    def _blank_is_absent(cls, value: object) -> object:
        """An empty override means "do not fetch", not an empty URL to request."""
        if isinstance(value, str) and not value.strip():
            return None
        return value

    @field_validator("google_api_key", "groq_api_key", mode="before")
    @classmethod
    def _blank_key_is_absent(cls, value: object) -> object:
        """Treat ``GOOGLE_API_KEY=`` as missing, not as a key.

        The env template ships the names with empty values, so without this an unconfigured
        install reports a provider as available and fails mid-request instead of showing the
        actionable 'missing API key' state (feature 65).
        """
        if isinstance(value, str) and not value.strip():
            return None
        return value

    paths: Paths = Paths()
    ingestion: IngestionSettings = IngestionSettings()
    chunking: ChunkingSettings = ChunkingSettings()
    embedding: EmbeddingSettings = EmbeddingSettings()
    retrieval: RetrievalSettings = RetrievalSettings()
    generation: GenerationSettings = GenerationSettings()
    guardrails: GuardrailSettings = GuardrailSettings()
    resilience: ResilienceSettings = ResilienceSettings()
    ui: UiSettings = UiSettings()
    evaluation: EvaluationSettings = EvaluationSettings()

    @model_validator(mode="after")
    def _bands_sit_above_the_floor(self) -> Settings:
        """Evidence below the floor is never shown, so a band below it is unreachable.

        The first defaults put Moderate at 0.25 under a floor of 0.3386: every served answer
        was therefore Moderate or High, and 'Low' could never appear.
        """
        if self.ui.evidence_strength_moderate <= self.retrieval.relevance_floor:
            raise ValueError(
                "ui.evidence_strength_moderate must exceed retrieval.relevance_floor, "
                "or the Low evidence band can never be shown"
            )
        return self

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
