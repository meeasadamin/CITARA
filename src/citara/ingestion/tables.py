"""Table detection and Markdown serialisation (features 7, 8, 9).

The highest-value content in this corpus is inside tables: district casualty counts,
sectoral loss figures, provincial breakdowns. Phase 0 confirmed the PDNA's headline damage
totals appear *only* in tables, never as extractable prose.

Plain text extraction flattens a table into a run of concatenated numbers. Markdown, by
contrast, is parsed reliably by language models - so tables are extracted structurally and
re-serialised rather than read as text.

Phase 0 also found how PyMuPDF degrades here: merged header cells produce repeated columns
(``Col1|Col2|Col3`` all holding the same value) and headers split across several rows. Those
artefacts are collapsed below; tables too small or too empty to trust are flagged instead of
being indexed as noise.
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING, Any

from citara.config import IngestionSettings

if TYPE_CHECKING:  # pragma: no cover - import only for type checking
    import pymupdf

# A placeholder header PyMuPDF invents when a column has no readable name.
_PLACEHOLDER_PREFIX = "col"
# Compound split across lines inside a cell: "Medium-\nterm".
_CELL_HYPHEN_BREAK = re.compile(r"(\w)[-‐‑][ \t]*\n[ \t]*([a-z])")  # noqa: RUF001


def _clean_cell(value: Any) -> str:
    """Normalise one cell to a single-line string.

    Repairs line-broken compounds first. A cell holding "Medium-\\nterm" would otherwise
    collapse to "Medium- term": cells are flattened to one line here, not by the prose
    normaliser, which is why 540 such breaks survived inside tables after the prose path was
    fixed.
    """
    if value is None:
        return ""
    text = _CELL_HYPHEN_BREAK.sub(r"\1-\2", str(value))
    return " ".join(text.split())


def _collapse_duplicate_columns(rows: list[list[str]]) -> list[list[str]]:
    """Drop columns that merely repeat their left-hand neighbour.

    A cell spanning three columns is emitted by PyMuPDF as the same value three times. Kept
    as-is, one figure is indexed three times and the row reads as nonsense.
    """
    if not rows:
        return rows
    width = max(len(r) for r in rows)
    padded = [r + [""] * (width - len(r)) for r in rows]

    keep: list[int] = []
    for col in range(width):
        column = [row[col] for row in padded]
        if keep:
            previous = [row[keep[-1]] for row in padded]
            if column == previous:
                continue  # exact repeat of the previous column: a merged-cell artefact
        if any(cell for cell in column):
            keep.append(col)
    return [[row[c] for c in keep] for row in padded]


def _merge_complementary_columns(rows: list[list[str]]) -> list[list[str]]:
    """Merge adjacent columns that are never both filled in the same row.

    PyMuPDF splits one logical column across two physical ones when cells are offset: in the
    PDNA damage tables the header label lands in one column and its figures in the next, so
    "Province/Region" and "Balochistan" end up as separate columns that are never populated
    together. Two columns that never co-occur in any row are one column.
    """
    if not rows:
        return rows
    width = max(len(r) for r in rows)
    padded = [r + [""] * (width - len(r)) for r in rows]

    merged: list[list[str]] = [[row[0] for row in padded]] if width else []
    for col in range(1, width):
        column = [row[col] for row in padded]
        previous = merged[-1]
        both_filled = any(a and b for a, b in zip(previous, column, strict=True))
        if not both_filled:
            merged[-1] = [a or b for a, b in zip(previous, column, strict=True)]
        else:
            merged.append(column)
    return [list(row) for row in zip(*merged, strict=True)] if merged else []


def _merge_header_rows(rows: list[list[str]]) -> tuple[list[str], list[list[str]]]:
    """Fold multi-row headers into one header row.

    PyMuPDF splits a wrapped header across rows ("Value of" / "Damaged Assets" / "(PKR
    Million)"). Joining them keeps the unit attached to the column, which is the part that
    makes a damage figure meaningful.
    """
    if not rows:
        return [], []

    # A header may wrap onto following rows, but never consumes half the table: a text-only
    # table (common in responsibility matrices) has no digits to mark where data begins, and
    # an unbounded header swallowed every body row, leaving a table with no content at all.
    max_depth = min(3, max(1, len(rows) - 1))
    first_filled = sum(1 for c in rows[0] if c)

    header_depth = 1
    for index, row in enumerate(rows[:max_depth]):
        if index == 0:
            continue
        cells = [c for c in row if c]
        numeric = sum(1 for c in cells if any(ch.isdigit() for ch in c))
        if cells and numeric * 2 >= len(cells):
            break  # this row holds data, not header text
        if len(cells) >= first_filled:
            break  # as full as the header row: it is a data row, not a wrapped continuation
        header_depth = index + 1

    header_rows = rows[:header_depth]
    width = max((len(r) for r in rows), default=0)
    header: list[str] = []
    for col in range(width):
        parts = [r[col] for r in header_rows if col < len(r) and r[col]]
        # Discard PyMuPDF's invented "Col7" placeholders.
        parts = [p for p in parts if not p.lower().startswith(_PLACEHOLDER_PREFIX)]
        deduped: list[str] = []
        for part in parts:
            if part not in deduped:
                deduped.append(part)
        header.append(" ".join(deduped))
    return header, rows[header_depth:]


def to_markdown(rows: list[list[str]]) -> str:
    """Serialise cleaned rows as a Markdown table.

    Returns an empty string when there is a header but no body: a table with no data rows
    carries no information, and the caller flags it rather than indexing an empty shell.
    """
    header, body = _merge_header_rows(rows)
    if not header or not body:
        return ""
    width = len(header)
    lines = [
        "| " + " | ".join(header) + " |",
        "| " + " | ".join(["---"] * width) + " |",
    ]
    for row in body:
        cells = (row + [""] * width)[:width]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def extract_tables(
    page: pymupdf.Page, settings: IngestionSettings
) -> tuple[list[str], int, list[pymupdf.Rect]]:
    """Extract every table on *page* as Markdown.

    Returns:
        ``(markdown_tables, flagged_count, table_rects)``. ``flagged_count`` counts tables
        detected but rejected as unreliable - a known gap beats a silent corruption.
        ``table_rects`` lets the caller drop table regions from the page's prose so the same
        figures are not also indexed in flattened form.
    """
    if not settings.extract_tables:
        return [], 0, []

    try:
        finder = page.find_tables()
    except Exception:  # table detection is best-effort by design
        return [], 1, []

    markdown: list[str] = []
    flagged = 0
    rects: list[pymupdf.Rect] = []
    for table in finder.tables:
        rects.append(table.bbox)
        try:
            raw = table.extract()
        except Exception:  # a single malformed table must not stop the page
            flagged += 1
            continue

        rows = [[_clean_cell(cell) for cell in row] for row in raw]
        rows = [r for r in rows if any(c for c in r)]
        if settings.collapse_duplicate_table_columns:
            rows = _collapse_duplicate_columns(rows)
            rows = _merge_complementary_columns(rows)

        filled = sum(1 for row in rows for cell in row if cell)
        if len(rows) < 2 or filled < settings.min_table_cells:
            flagged += 1
            continue

        rendered = to_markdown(rows)
        if rendered:
            markdown.append(rendered)
        else:
            flagged += 1
    return markdown, flagged, rects
