"""The CITARA mark, drawn as inline SVG (feature 60).

Two page-corner arcs left open to read as a C: the corpus, and the gap where an answer is
only as good as the page behind it. Inline rather than a file so it inherits the colour it
sits on, stays sharp on any screen, and costs no request.

Nothing here implies an NDMA identity. The prototype is independent (feature 47), so the mark
is abstract: no crest, no flag, no institutional emblem.
"""

from __future__ import annotations

ARC = (
    '<path d="M24.5 8.6A11 11 0 1 0 24.5 23.4" fill="none" stroke="currentColor" '
    'stroke-width="3.1" stroke-linecap="round"/>'
)
# The folded corner that turns the arc from a plain letter into a page.
FOLD = '<path d="M20.6 3.4h8v8z" fill="currentColor" opacity="0.9"/>'


def mark(size: int = 30, title: str = "CITARA") -> str:
    """The symbol on its own, inheriting the current text colour."""
    return (
        f'<svg viewBox="0 0 32 32" width="{size}" height="{size}" role="img" '
        f'aria-label="{title}" focusable="false">{ARC}{FOLD}</svg>'
    )


def lockup(subtitle: str) -> str:
    """Mark, wordmark and subtitle, as the header bar shows them."""
    return (
        '<div class="brand">'
        f'<span class="brand-mark">{mark()}</span>'
        '<span class="brand-text">'
        '<span class="brand-name">CITARA</span>'
        f'<span class="brand-sub">{subtitle}</span>'
        "</span></div>"
    )
