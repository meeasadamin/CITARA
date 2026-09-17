"""Chunking orchestration (features 12-17).

Two decisions here are load-bearing.

**Prose is chunked across page boundaries, tables are not.** A page break is a typesetting
accident, not a semantic one: chunking strictly per page guarantees that any procedure
spanning a break is severed. Consecutive pages of a document are therefore joined into a
stream and split at topic shifts, with each chunk recording the page range it covers so the
citation stays exactly verifiable ("pp. 46-47"). A gap in page numbers - a skipped blank or
image-only page - ends the stream, because those pages are unknown content, not continuity.

**A Markdown table is atomic.** Split it and the rows lose their header, which is what made
the numbers meaningful. Oversized tables are divided by rows with the header repeated.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from itertools import pairwise

import numpy as np

from citara.chunking.models import Chunk, ChunkReport
from citara.chunking.semantic import enforce_max_size, recursive_split, semantic_split
from citara.config import Settings, get_settings
from citara.embeddings import Embedder
from citara.ingestion.models import PageRecord
from citara.log import get_logger, stage

log = get_logger("chunking")

_PAGE_SEPARATOR = "\n\n"


def _page_streams(records: Sequence[PageRecord]) -> list[list[PageRecord]]:
    """Group text records into runs of consecutive pages."""
    streams: list[list[PageRecord]] = []
    current: list[PageRecord] = []
    for record in sorted(records, key=lambda r: r.page_number):
        if current and record.page_number != current[-1].page_number + 1:
            streams.append(current)
            current = []
        current.append(record)
    if current:
        streams.append(current)
    return streams


def _collapse_with_map(text: str) -> tuple[str, list[int]]:
    """Whitespace-collapsed copy of *text*, plus a map back to original offsets.

    Chunks cannot be located in the page stream by plain string search: semantic splitting
    joins sentences with a single space, while the stream still holds the paragraph breaks
    the text was extracted with. Any chunk spanning a blank line then fails to match, the
    page span falls back to a guess, and the citation can name the wrong page - the one
    failure this system exists to prevent.
    """
    collapsed: list[str] = []
    index_map: list[int] = []
    previous_space = False
    for index, character in enumerate(text):
        if character.isspace():
            if previous_space or not collapsed:
                continue
            collapsed.append(" ")
            index_map.append(index)
            previous_space = True
        else:
            collapsed.append(character)
            index_map.append(index)
            previous_space = False
    return "".join(collapsed), index_map


def _page_for_offset(offsets: list[tuple[int, int, PageRecord]], position: int) -> PageRecord:
    """Which page a character offset in the joined stream belongs to."""
    for start, end, record in offsets:
        if start <= position < end:
            return record
    return offsets[-1][2]


def _split_table(content: str, settings: Settings) -> list[str]:
    """Divide an oversized table by rows, repeating the header on every part."""
    lines = content.splitlines()
    if len(lines) < 3 or len(content) <= settings.chunking.max_chunk_chars:
        return [content]

    header, separator, body = lines[0], lines[1], lines[2:]
    prefix_len = len(header) + len(separator) + 2
    parts: list[str] = []
    current: list[str] = []
    for row in body:
        if current and prefix_len + sum(len(r) + 1 for r in current) + len(row) > (
            settings.chunking.max_chunk_chars
        ):
            parts.append("\n".join([header, separator, *current]))
            current = []
        current.append(row)
    if current:
        parts.append("\n".join([header, separator, *current]))
    return parts


def _apply_overlap(pieces: list[str], overlap: int, ceiling: int | None = None) -> list[str]:
    """Give adjacent chunks a shared margin so meaning spanning a boundary survives.

    The margin is trimmed to keep the result under *ceiling*: overlap used to be added after
    the size limit was enforced, which pushed 25 chunks past it.
    """
    if overlap <= 0 or len(pieces) < 2:
        return pieces
    result = [pieces[0]]
    for previous, piece in pairwise(pieces):
        budget = overlap if ceiling is None else max(0, min(overlap, ceiling - len(piece) - 1))
        tail = previous[-budget:].lstrip() if budget else ""
        result.append(f"{tail} {piece}" if tail else piece)
    return result


def is_low_information(text: str) -> bool:
    """True for contents-page filler: dot leaders, rule lines, page-number columns.

    "Introduction ................ 5" embeds as plausible prose and matches almost any query
    while carrying no answer, so it is dropped rather than indexed (feature 4).
    """
    stripped = text.strip()
    if not stripped:
        return True
    informative = sum(1 for character in stripped if character.isalnum() or character.isspace())
    return informative / len(stripped) < 0.75


def chunk_records(
    records: Iterable[PageRecord],
    settings: Settings | None = None,
    embedder: Embedder | None = None,
) -> tuple[list[Chunk], ChunkReport]:
    """Turn page records into chunks."""
    settings = settings or get_settings()
    embedder = embedder or Embedder(settings.embedding)
    config = settings.chunking
    records = list(records)

    report = ChunkReport(config_fingerprint=settings.fingerprint(), records_in=len(records))
    chunks: list[Chunk] = []

    by_doc: dict[str, list[PageRecord]] = {}
    for record in records:
        by_doc.setdefault(record.doc_id, []).append(record)

    with stage(log, "chunk_records", documents=len(by_doc)) as details:
        for doc_id, doc_records in by_doc.items():
            ordinal = 0
            texts = [r for r in doc_records if r.kind == "text"]
            tables = [r for r in doc_records if r.kind == "table"]

            for stream in _page_streams(texts):
                joined_parts: list[str] = []
                offsets: list[tuple[int, int, PageRecord]] = []
                cursor = 0
                for record in stream:
                    offsets.append((cursor, cursor + len(record.content), record))
                    joined_parts.append(record.content)
                    cursor += len(record.content) + len(_PAGE_SEPARATOR)
                joined = _PAGE_SEPARATOR.join(joined_parts)

                if config.strategy == "recursive":
                    pieces, strategy = recursive_split(joined, config), "recursive"
                else:
                    pieces, strategy = semantic_split(joined, embedder, config)
                if strategy == "recursive" and stream[0].filename not in report.fallback_documents:
                    report.fallback_documents.append(stream[0].filename)

                pieces, resplit = enforce_max_size(pieces, config)
                report.oversized_resplit += resplit
                # Overlap is layered on afterwards, but every chunk is *located* by its own
                # text: re-splitting rewrites content, so an overlapped body no longer
                # matches the stream and its page span - the citation - becomes a guess.
                overlapped = _apply_overlap(
                    pieces, config.chunk_overlap_chars, config.max_chunk_chars
                )

                collapsed_joined, index_map = _collapse_with_map(joined)
                search_from = 0
                for core, piece in zip(pieces, overlapped, strict=True):
                    body = piece.strip()
                    anchor = core.strip()
                    if len(anchor) < config.min_chunk_chars:
                        report.dropped_short += 1
                        continue
                    if is_low_information(anchor):
                        report.dropped_noise += 1
                        continue

                    # Locate the piece in the stream to recover its true page span, matching
                    # on whitespace-collapsed text so a chunk spanning a paragraph break is
                    # still found exactly.
                    collapsed_body = " ".join(anchor.split())
                    probe = collapsed_body[:80]
                    found = collapsed_joined.find(probe, search_from)
                    if found < 0:
                        found = collapsed_joined.find(probe)
                    if found < 0:
                        report.unlocated_chunks += 1
                        found = min(search_from, max(len(index_map) - 1, 0))
                    last = min(found + len(collapsed_body) - 1, len(index_map) - 1)
                    search_from = max(found + 1, search_from)

                    position = index_map[found] if index_map else 0
                    end_position = index_map[last] if index_map else 0

                    start_record = _page_for_offset(offsets, position)
                    end_record = _page_for_offset(offsets, end_position)
                    sources = tuple(
                        r.record_id
                        for r in stream
                        if start_record.page_number <= r.page_number <= end_record.page_number
                    )
                    chunks.append(
                        Chunk(
                            chunk_id=Chunk.make_id(
                                doc_id, start_record.page_number, end_record.page_number, ordinal
                            ),
                            doc_id=doc_id,
                            title=start_record.title,
                            filename=start_record.filename,
                            page_start=start_record.page_number,
                            page_end=max(end_record.page_number, start_record.page_number),
                            total_pages=start_record.total_pages,
                            year=start_record.year,
                            kind="text",
                            strategy=strategy,  # type: ignore[arg-type]
                            ordinal=ordinal,
                            content=body,
                            source_record_ids=sources,
                        )
                    )
                    ordinal += 1
                    report.text_chunks += 1
                    if strategy == "semantic":
                        report.semantic_chunks += 1
                    else:
                        report.recursive_chunks += 1

            for record in sorted(tables, key=lambda r: (r.page_number, r.table_index or 0)):
                for part in _split_table(record.content, settings):
                    chunks.append(
                        Chunk(
                            chunk_id=Chunk.make_id(
                                doc_id, record.page_number, record.page_number, ordinal
                            ),
                            doc_id=doc_id,
                            title=record.title,
                            filename=record.filename,
                            page_start=record.page_number,
                            page_end=record.page_number,
                            total_pages=record.total_pages,
                            year=record.year,
                            kind="table",
                            strategy="table",
                            ordinal=ordinal,
                            content=part,
                            source_record_ids=(record.record_id,),
                        )
                    )
                    ordinal += 1
                    report.table_chunks += 1

        report.chunks_out = len(chunks)
        if chunks:
            lengths = np.array([c.char_count for c in chunks])
            report.mean_chars = float(lengths.mean())
            report.p95_chars = int(np.percentile(lengths, 95))
        details.update(report.summary())

    return chunks, report
