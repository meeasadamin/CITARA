"""Shared test fixtures.

Tests must not depend on whatever the developer happens to have exported: a real
GOOGLE_API_KEY in the shell would silently change provider-availability assertions.
"""

from __future__ import annotations

import os

import pytest

_PROVIDER_VARS = ("GOOGLE_API_KEY", "GROQ_API_KEY", "CITARA_GOOGLE_API_KEY", "CITARA_GROQ_API_KEY")


@pytest.fixture(autouse=True)
def isolated_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    """Remove provider keys and any CITARA_* overrides for the duration of each test."""
    for name in _PROVIDER_VARS:
        monkeypatch.delenv(name, raising=False)
    for name in list(os.environ):
        if name.startswith("CITARA_"):
            monkeypatch.delenv(name, raising=False)
