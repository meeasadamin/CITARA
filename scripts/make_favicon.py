"""Render the CITARA mark to assets/favicon.png.

The browser tab needs a raster image, while the header draws the mark as inline SVG. Both are
described by the same geometry, kept here next to the SVG in citara/ui/logo.py so the two
cannot drift: an arc left open on the right, and a folded page corner.

Run:  uv run python scripts/make_favicon.py
"""

from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageDraw

from citara.ui.styles import NAVY

SIZE = 256
SCALE = SIZE / 32  # the SVG is drawn on a 32x32 grid
OUT = Path(__file__).resolve().parents[1] / "assets" / "favicon.png"


def main() -> int:
    image = Image.new("RGBA", (SIZE, SIZE), (0, 0, 0, 0))
    draw = ImageDraw.Draw(image)

    # The arc: centre (16,16), radius 11, open between 41 and -41 degrees on the right.
    radius = 11 * SCALE
    centre = 16 * SCALE
    box = (centre - radius, centre - radius, centre + radius, centre + radius)
    draw.arc(box, start=41, end=319, fill=NAVY, width=round(3.1 * SCALE))

    # The folded corner, top right.
    draw.polygon(
        [(20.6 * SCALE, 3.4 * SCALE), (28.6 * SCALE, 3.4 * SCALE), (28.6 * SCALE, 11.4 * SCALE)],
        fill=NAVY,
    )

    OUT.parent.mkdir(parents=True, exist_ok=True)
    image.save(OUT)
    print(f"wrote {OUT.relative_to(OUT.parents[1])} ({OUT.stat().st_size // 1024} KB, {SIZE}px)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
