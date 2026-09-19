"""Startup checks that turn missing pieces into specific, actionable messages (feature 65).

What a panel sees when something goes wrong says more about the engineering than what it sees
when things go right. Each problem the app can start into - no index, an empty index, no model
key, no failover key - gets a message naming the cause and the fix, instead of a stack trace
from somewhere deep in ChromaDB.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Literal

from citara.config import Settings

Severity = Literal["error", "warning", "info"]


@dataclass(frozen=True)
class Problem:
    """One thing wrong with the deployment, and what to do about it."""

    severity: Severity
    title: str
    detail: str

    @property
    def blocking(self) -> bool:
        """Errors stop the app: without an index there is nothing to answer from."""
        return self.severity == "error"


_BUILD_STEPS = (
    "Build it with `uv run python -m citara.ingestion`, then `uv run python -m citara.chunking` "
    "and `uv run python -m citara.indexing`. A deployment must ship the built index with the app."
)


def check(settings: Settings) -> list[Problem]:
    """Everything that would stop the app answering, or quietly change how it answers."""
    paths = settings.paths
    problems: list[Problem] = []

    required = {
        "chunk store": paths.resolved(paths.data_dir) / "chunks.jsonl",
        "vector index": paths.resolved(paths.chroma_dir),
        "keyword index": paths.resolved(paths.bm25_path),
        "index manifest": paths.resolved(paths.index_manifest_path),
    }
    missing = [name for name, path in required.items() if not path.exists()]
    if missing:
        problems.append(
            Problem(
                "error",
                "The document index is missing",
                f"Not found: {', '.join(missing)}. {_BUILD_STEPS}",
            )
        )
        return problems

    try:
        manifest = json.loads(required["index manifest"].read_text(encoding="utf-8"))
        chunk_count = int(manifest.get("chunk_count", 0))
        documents = len(manifest.get("documents", []))
    except (OSError, ValueError, AttributeError):
        problems.append(
            Problem(
                "error",
                "The document index is unreadable",
                "The index manifest could not be parsed; the index is likely damaged. "
                "Rebuild it with `uv run python -m citara.indexing --rebuild`.",
            )
        )
        return problems

    if chunk_count == 0 or documents == 0:
        problems.append(
            Problem(
                "error",
                "The document index is empty",
                "No documents were indexed, so there is nothing to answer from. Put the NDMA "
                f"PDFs in `docs/` and rebuild. {_BUILD_STEPS}",
            )
        )
        return problems

    if not settings.has_any_provider:
        problems.append(
            Problem(
                "warning",
                "No language-model key is configured",
                "Answers will show the most relevant source passages with their citations, but "
                "no written summary. Add GOOGLE_API_KEY (and GROQ_API_KEY for failover) to `.env` "
                "locally, or to the app's Secrets when deployed, then restart.",
            )
        )
    elif settings.groq_api_key is None or not settings.resilience.enable_failover:
        problems.append(
            Problem(
                "info",
                "Failover is not available",
                "Only the primary model is configured. If it is rate-limited, answers fall back "
                "to cited source passages. Add GROQ_API_KEY to enable the second provider.",
            )
        )
    return problems
