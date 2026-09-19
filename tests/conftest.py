"""Shared test fixtures.

Two kinds of isolation, in one fixture because their order matters. Splitting them let the
environment-stripping fixture delete the data-directory override the other had just set,
depending on how pytest resolved them - and the symptom was subtle: tests passed individually
and failed together, because a cached answer from one test satisfied another that was
asserting every provider had failed.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

_PROVIDER_VARS = ("GOOGLE_API_KEY", "GROQ_API_KEY", "CITARA_GOOGLE_API_KEY", "CITARA_GROQ_API_KEY")


@pytest.fixture(autouse=True)
def isolated_environment(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Give each test a clean environment and its own data directory.

    Tests must not depend on whatever the developer happens to have exported: a real
    GOOGLE_API_KEY in the shell would change provider-availability assertions. They must also
    never touch the working cache, usage counter or query log, which are real state the
    project depends on between runs.
    """
    for name in _PROVIDER_VARS:
        monkeypatch.delenv(name, raising=False)
    for name in list(os.environ):
        if name.startswith("CITARA_"):
            monkeypatch.delenv(name, raising=False)

    # Set last: the loop above would otherwise remove it.
    monkeypatch.setenv("CITARA_PATHS__DATA_DIR", str(tmp_path / "data"))
