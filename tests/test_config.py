"""Tests for the centralised settings object (feature 72)."""

from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from citara.config import Settings, get_settings


def make_settings(**overrides: object) -> Settings:
    """Settings built in isolation from the developer's own .env file."""
    return Settings(_env_file=None, **overrides)  # type: ignore[arg-type]


def test_defaults_match_the_specified_pipeline() -> None:
    s = make_settings()
    assert s.retrieval.dense_k == 12
    assert s.retrieval.sparse_k == 12
    assert s.retrieval.top_k == 5
    assert s.retrieval.rerank_candidates == 24
    assert s.generation.temperature == pytest.approx(0.1)
    assert s.embedding.hnsw_space == "cosine"
    assert s.embedding.normalise is True


def test_nested_env_override(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CITARA_RETRIEVAL__DENSE_K", "16")
    monkeypatch.setenv("CITARA_CHUNKING__MIN_CHUNK_CHARS", "200")
    monkeypatch.setenv("CITARA_LOG_LEVEL", "DEBUG")
    s = make_settings()
    assert s.retrieval.dense_k == 16
    assert s.chunking.min_chunk_chars == 200
    assert s.log_level == "DEBUG"


def test_provider_keys_read_unprefixed_names(monkeypatch: pytest.MonkeyPatch) -> None:
    """Streamlit Cloud exposes secrets under their plain names."""
    monkeypatch.setenv("GOOGLE_API_KEY", "test-google-key")
    s = make_settings()
    assert s.has_primary_provider is True
    assert s.has_any_provider is True
    assert s.google_api_key is not None
    assert s.google_api_key.get_secret_value() == "test-google-key"


def test_missing_keys_are_not_fatal() -> None:
    """Config must load without keys: degraded mode still serves cited sources."""
    s = make_settings()
    assert s.has_any_provider is False


@pytest.mark.parametrize("blank", ["", "   ", "\t"])
def test_blank_keys_count_as_missing(blank: str, monkeypatch: pytest.MonkeyPatch) -> None:
    """.env ships GOOGLE_API_KEY= with no value; that must not read as a configured provider."""
    monkeypatch.setenv("GOOGLE_API_KEY", blank)
    monkeypatch.setenv("GROQ_API_KEY", blank)
    s = make_settings()
    assert s.google_api_key is None
    assert s.groq_api_key is None
    assert s.has_primary_provider is False
    assert s.has_any_provider is False


def test_real_key_still_detected(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GROQ_API_KEY", "gsk_realish_value")
    s = make_settings()
    assert s.has_primary_provider is False
    assert s.has_any_provider is True


def test_evaluation_defaults_cover_the_ablation_table() -> None:
    s = make_settings()
    assert s.evaluation.ablation_modes == ("dense", "sparse", "hybrid", "hybrid_rerank")
    assert s.evaluation.hit_rate_k == (1, 3, 5)
    assert s.evaluation.latency_percentiles == (50.0, 95.0)
    assert s.evaluation.judge_temperature == 0.0


def test_evidence_bands_must_be_ordered() -> None:
    from citara.config import UiSettings

    with pytest.raises(ValidationError):
        UiSettings(evidence_strength_high=0.2, evidence_strength_moderate=0.5)


def test_secrets_never_appear_in_dump_or_repr(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GROQ_API_KEY", "super-secret-value")
    s = make_settings()
    assert "super-secret-value" not in repr(s)
    assert "super-secret-value" not in str(s.public_dump())
    assert "groq_api_key" not in s.public_dump()


def test_settings_are_frozen() -> None:
    s = make_settings()
    with pytest.raises(ValidationError):
        s.retrieval.top_k = 9  # type: ignore[misc]


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("top_k", 100),  # cannot exceed the rerank shortlist
        ("rerank_candidates", 500),  # cannot exceed pooled candidates
        ("dense_k", 0),  # must be positive
    ],
)
def test_impossible_retrieval_funnels_are_rejected(field: str, value: int) -> None:
    from citara.config import RetrievalSettings

    with pytest.raises(ValidationError):
        RetrievalSettings(**{field: value})  # type: ignore[arg-type]


def test_zero_weights_rejected() -> None:
    from citara.config import RetrievalSettings

    with pytest.raises(ValidationError):
        RetrievalSettings(dense_weight=0.0, sparse_weight=0.0)


def test_overlap_must_be_smaller_than_chunk() -> None:
    from citara.config import ChunkingSettings

    with pytest.raises(ValidationError):
        ChunkingSettings(fallback_chunk_chars=500, chunk_overlap_chars=500)


def test_ablation_modes_are_all_valid() -> None:
    """The four rows of the ablation table must be expressible as configuration."""
    from citara.config import RetrievalSettings

    for mode in ("dense", "sparse", "hybrid", "hybrid_rerank"):
        assert RetrievalSettings(mode=mode).mode == mode  # type: ignore[arg-type]


def test_fingerprint_is_stable_and_sensitive() -> None:
    a = make_settings()
    b = make_settings()
    assert a.fingerprint() == b.fingerprint()

    changed = a.model_copy(update={"retrieval": a.retrieval.model_copy(update={"dense_k": 20})})
    assert changed.fingerprint() != a.fingerprint()


def test_fingerprint_ignores_secrets(monkeypatch: pytest.MonkeyPatch) -> None:
    baseline = make_settings().fingerprint()
    monkeypatch.setenv("GOOGLE_API_KEY", "some-key")
    assert make_settings().fingerprint() == baseline


def test_derived_paths_sit_under_data_dir() -> None:
    from citara.config import Paths

    # Explicit defaults: the test fixture redirects the data directory for isolation, and
    # this is asserting how paths derive from it, not what the environment happens to say.
    s = make_settings(paths=Paths())
    assert s.paths.chroma_dir == Path("data/chroma")
    # JSON, not a pickle: an index file that executes code on load does not belong in a
    # public repository, and rebuilding BM25 costs milliseconds.
    assert s.paths.bm25_path == Path("data/bm25/index.json")
    assert s.paths.manifest_path == Path("data/corpus_manifest.json")
    assert s.paths.index_manifest_path == Path("data/index_manifest.json")


def test_ensure_directories_creates_them(tmp_path: Path) -> None:
    from citara.config import Paths

    s = make_settings(paths=Paths(docs_dir=tmp_path / "docs", data_dir=tmp_path / "data"))
    s.ensure_directories()
    assert s.paths.chroma_dir.is_dir()
    assert s.paths.bm25_path.parent.is_dir()
    assert s.paths.cache_dir.is_dir()


def test_get_settings_is_cached() -> None:
    get_settings.cache_clear()
    assert get_settings() is get_settings()
    get_settings.cache_clear()
