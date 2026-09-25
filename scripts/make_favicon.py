"""Render the CITARA mark to assets/favicon.png.

The browser tab needs a raster image, while the header draws the mark as inline SVG. Both are
described by the same geometry, kept here next to the SVG in citara/ui/logo.py so the two
cannot drift: an arc left open on the right, and a folded page corner.

Run:  uv run python scripts/make_favicon.py
"""

from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageDraw

from citara.ui.styles import TEAL

SIZE = 256
SCALE = SIZE / 32  # the SVG is drawn on a 32x32 grid
OUT = Path(__file__).resolve().parents[1] / "assets" / "favicon.png"


def main() -> int:
    image = Image.new("RGBA", (SIZE, SIZE), (0, 0, 0, 0))
    draw = ImageDraw.Draw(image)

    def at(*points: tuple[float, float]) -> list[tuple[float, float]]:
        return [(x * SCALE, y * SCALE) for x, y in points]

    # The tile, with its top-right corner cut away like a turned page.
    draw.rounded_rectangle(
        (1 * SCALE, 1 * SCALE, 31 * SCALE, 31 * SCALE), radius=6 * SCALE, fill=TEAL
    )
    draw.polygon(at((21, 1), (31, 1), (31, 11)), fill=(0, 0, 0, 0))
    draw.polygon(at((21, 1), (31, 11), (21, 11)), fill=(255, 255, 255, 82))

    # The C: centre (16,18), radius 6.5, open between 45 and -45 degrees on the right.
    radius, cx, cy = 6.5 * SCALE, 16 * SCALE, 18 * SCALE
    draw.arc(
        (cx - radius, cy - radius, cx + radius, cy + radius),
        start=45,
        end=315,
        fill="#FFFFFF",
        width=round(2.7 * SCALE),
    )

    OUT.parent.mkdir(parents=True, exist_ok=True)
    image.save(OUT)
    print(f"wrote {OUT.relative_to(OUT.parents[1])} ({OUT.stat().st_size // 1024} KB, {SIZE}px)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
