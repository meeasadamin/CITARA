"""The CITARA mark, drawn as inline SVG (feature 60).

A document tile with a folded corner, holding an open C: the corpus, and the gap where an
answer is only as good as the page behind it. Solid shapes rather than thin strokes, because
the mark has to survive the browser tab at 16px as well as the header at 32.

Inline rather than a file so it inherits the colours it is given - a light tile on the dark
brand bar, a dark tile on the pale sidebar - stays sharp on any screen, and costs no request.

Nothing here implies an NDMA identity. The prototype is independent (feature 47), so the mark
is abstract: no crest, no flag, no institutional emblem.
"""

from __future__ import annotations

# A rounded tile whose top-right corner is cut away, like a turned page.
_TILE = "M9 1 H21 L31 11 V25 A6 6 0 0 1 25 31 H7 A6 6 0 0 1 1 25 V7 A6 6 0 0 1 7 1 Z"
_FOLD = "M21 1 L31 11 H23 A2 2 0 0 1 21 9 Z"
# The C, open to the right, centred on the tile.
_GLYPH = "M20.6 13.4 A6.5 6.5 0 1 0 20.6 22.6"


def mark(size: int = 32, tile: str = "#0E3B3E", glyph: str = "#FFFFFF") -> str:
    """The symbol on its own: a *tile* square carrying the *glyph* C."""
    return (
        f'<svg viewBox="0 0 32 32" width="{size}" height="{size}" role="img" '
        f'aria-label="CITARA" focusable="false">'
        f'<path d="{_TILE}" fill="{tile}"/>'
        f'<path d="{_FOLD}" fill="{glyph}" opacity="0.32"/>'
        f'<path d="{_GLYPH}" fill="none" stroke="{glyph}" stroke-width="2.7" '
        f'stroke-linecap="round"/>'
        "</svg>"
    )


def on_dark(size: int = 32) -> str:
    """For the brand bar: a pale tile with the deep glyph cut out of it."""
    return mark(size, tile="#FFFFFF", glyph="#0E3B3E")


def on_light(size: int = 24) -> str:
    """For the sidebar and anywhere else on a pale surface."""
    return mark(size, tile="#0E3B3E", glyph="#FFFFFF")


def wordmark(size: int = 24) -> str:
    """Mark and name together, for the top of the sidebar."""
    return (
        f'<span class="sidebar-brand">{on_light(size)}'
        '<span class="sidebar-brand-name">CITARA</span></span>'
    )
