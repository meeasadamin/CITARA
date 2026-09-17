"""Tests for text normalisation (feature 3).

The font-repair cases are taken verbatim from the corpus: these exact broken words appear in
the NDRP, the National Adaptation Plan and the PDNA.
"""

from __future__ import annotations

import pytest

from citara.ingestion.normalise import (
    dehyphenate,
    has_font_corruption,
    normalise_text,
    repair_font_encoding,
)


@pytest.mark.parametrize(
    ("broken", "expected"),
    [
        ("NaƟonal Disaster Management Authority", "National Disaster Management Authority"),
        ("collaboraƟve eﬀorts", "collaborative efforts"),
        ("ðoods and ðash ðooding", "floods and flash flooding"),
        ("Paciïc, ïrst, signiïcantly", "Pacific, first, significantly"),
        ("maƩers in QueƩa", "matters in Quetta"),
        ("aŌer the event, oŌen", "after the event, often"),
    ],
)
def test_broken_font_words_are_repaired(broken: str, expected: str) -> None:
    assert normalise_text(broken, repair_font=True) == expected


def test_private_use_ligatures_always_repaired() -> None:
    """Private-use glyphs are unambiguous, so they need no per-document decision."""
    assert normalise_text("Prole of condence") == "Profile of confidence"
    assert normalise_text("Conict and oods") == "Conflict and floods"


def test_real_ligature_characters_folded_by_nfkc() -> None:
    assert normalise_text("oﬃcials ﬁnal ﬂood") == "officials final flood"


def test_corruption_detection_requires_two_signatures() -> None:
    assert has_font_corruption("NaƟonal plan, ïrst response") is True
    assert has_font_corruption("The naïve approach") is False
    assert has_font_corruption("El Niño and ordinary text") is False


def test_legitimate_diacritics_survive_when_not_repairing() -> None:
    """ï and ð are real characters; unconditional repair would corrupt them."""
    text = "A naïve reading of El Niño"
    assert normalise_text(text, repair_font=False) == text
    assert "naïve" not in repair_font_encoding(text)  # proves the repair would damage it


def test_dehyphenation_joins_lowercase_continuations() -> None:
    assert dehyphenate("inunda-\ntion") == "inundation"
    assert dehyphenate("prepared-\n   ness") == "preparedness"


def test_dehyphenation_keeps_compound_terms() -> None:
    """An uppercase continuation is a real compound, not a split word."""
    assert dehyphenate("multi-\nHazard") == "multi-\nHazard"
    assert dehyphenate("NDMA-\nPDMA coordination") == "NDMA-\nPDMA coordination"


def test_paragraph_structure_preserved() -> None:
    text = "First line\nsecond line of same paragraph\n\nNew paragraph"
    assert normalise_text(text) == "First line second line of same paragraph\n\nNew paragraph"


def test_whitespace_and_nbsp_collapsed() -> None:
    assert normalise_text("spaced out    words") == "spaced out words"  # noqa: RUF001


def test_empty_input() -> None:
    assert normalise_text("") == ""
