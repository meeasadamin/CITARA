"""Text normalisation for print-typeset government PDFs (feature 3).

NDMA documents are typeset for print, not parsing. Three defects appear in this corpus and
each one silently breaks retrieval if left alone:

1. **Broken font encodings.** Subset fonts map ligatures onto unrelated Unicode code points,
   so the extracted text reads ``NaƟonal``, ``ðoods``, ``maƩers``, ``QueƩa``. Neither a
   dense embedding nor a BM25 token for "flood" or "Quetta" can match those.
2. **Hyphenation across line ends**, an artefact of justified columns.
3. **Layout whitespace** - single newlines inside sentences, no-break spaces, runs of spaces.

The substitution table below was derived by scanning the actual corpus, not assumed: every
mapping is supported by words that appear in these documents.

The Latin-letter repairs are **conditional**. ``ï`` and ``ð`` are real characters
("naïve", "El Niño"), so repairing them unconditionally would corrupt correct text. They are
applied only to documents whose text carries the corruption signature.
"""

from __future__ import annotations

import re
import unicodedata

# Always safe: private-use ligature glyphs and bullets emitted by Symbol-style subset fonts.
_ALWAYS: dict[str, str] = {
    "": "fi",
    "": "fl",
    "": "ff",
    "": "•",  # Symbol-font bullet
    " ": " ",  # noqa: RUF001 - mapping the no-break space is the point
}

# Conditional: correct characters in other documents, corruption in broken-font ones.
# Evidence from this corpus: Paciïc/ïrst (fi), ðoods/ðow (fl), NaƟonal (ti),
# maƩers/QueƩa (tt), aŌer/oŌen (ft).
_FONT_REPAIRS: dict[str, str] = {
    "Ɵ": "ti",  # Ɵ
    "Ʃ": "tt",  # Ʃ
    "Ō": "ft",  # Ō
    "ï": "fi",  # ï
    "ð": "fl",  # ð
    "Ÿ": "•",  # Ÿ used as a bullet glyph
}

# Words that can only result from the broken encodings above. Two independent hits are
# required before repairs are applied, so an isolated "naïve" never triggers them.
_CORRUPTION_SIGNATURES = re.compile(
    r"NaƟonal|ƟonaI|Ɵon\b|ïrst|ïcant|Paciïc|ðood|ðow\b|aƩ|maƩer|QueƩa|aŌer|oŌen",
)

# Hyphen, non-breaking hyphen or Unicode hyphen at a line end, continued by a lowercase letter.
_HYPHEN_BREAK = re.compile(r"(\w)[-‐‑]\n\s*([a-z])")  # noqa: RUF001 - Unicode hyphens intended
_SINGLE_NEWLINE = re.compile(r"(?<!\n)\n(?!\n)")
_MULTI_SPACE = re.compile(r"[ \t]{2,}")
_MULTI_BLANK = re.compile(r"\n{3,}")


def has_font_corruption(text: str) -> bool:
    """True when *text* shows the broken-font signature of this corpus.

    Requires two distinct signature hits: a single match could be a genuine word in an
    otherwise clean document.
    """
    return len(set(_CORRUPTION_SIGNATURES.findall(text))) >= 2


def repair_font_encoding(text: str) -> str:
    """Apply the conditional Latin-letter repairs."""
    for bad, good in _FONT_REPAIRS.items():
        text = text.replace(bad, good)
    return text


def dehyphenate(text: str) -> str:
    """Rejoin words split across a line end ("inunda-\\ntion" -> "inundation").

    Only joins when the continuation starts lowercase: "multi-\\nHazard" keeps its hyphen,
    which matters for the compound terms this corpus is full of.
    """
    return _HYPHEN_BREAK.sub(r"\1\2", text)


def normalise_markdown(text: str, *, repair_font: bool = False) -> str:
    """Normalise a serialised Markdown table without destroying its structure.

    :func:`normalise_text` folds single newlines into spaces, which is right for prose and
    fatal for a table: the rows collapse into one line and the Markdown an LLM relies on is
    gone. Here only character-level repairs are applied, line by line.
    """
    if not text:
        return ""

    for bad, good in _ALWAYS.items():
        text = text.replace(bad, good)
    text = unicodedata.normalize("NFKC", text)
    if repair_font:
        text = repair_font_encoding(text)

    lines = [_MULTI_SPACE.sub(" ", line).strip() for line in text.splitlines()]
    return "\n".join(line for line in lines if line)


def normalise_text(text: str, *, repair_font: bool = False, dehyphenate_words: bool = True) -> str:
    """Normalise one page of extracted text.

    Args:
        text: raw text from the PDF.
        repair_font: apply the conditional broken-font repairs. Decided per document by
            :func:`has_font_corruption`, not per page, so a clean page in a broken document
            is still treated consistently.
        dehyphenate_words: rejoin words broken across line ends.

    Returns:
        Text with paragraph structure preserved: single newlines inside a paragraph become
        spaces, blank lines stay as paragraph separators.
    """
    if not text:
        return ""

    for bad, good in _ALWAYS.items():
        text = text.replace(bad, good)

    # NFKC folds the real ligatures (ﬁ ﬂ ﬀ ﬃ) and compatibility spaces.
    text = unicodedata.normalize("NFKC", text)

    if repair_font:
        text = repair_font_encoding(text)

    if dehyphenate_words:
        text = dehyphenate(text)

    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = _SINGLE_NEWLINE.sub(" ", text)
    text = _MULTI_SPACE.sub(" ", text)
    text = _MULTI_BLANK.sub("\n\n", text)
    return text.strip()
