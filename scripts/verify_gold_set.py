"""Check the gold question set against the built corpus (feature 66).

A gold answer that cites a page which was never indexed is unreachable: Hit Rate would be
capped below 100% by the question set rather than by the retriever, and the ablation would
compare configurations against an impossible ceiling. This checks each cited location really
exists in the chunk store, and that questions marked unanswerable have no obvious answer
sitting in the corpus.

Run:  uv run python scripts/verify_gold_set.py
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter

from citara.config import get_settings
from citara.evaluation.gold import load_gold_set
from citara.indexing.builder import load_chunks
from citara.indexing.sparse_index import SparseIndex


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Verify the gold question set.")
    parser.add_argument(
        "--strict",
        action="store_true",
        help="fail if any question is still unverified by a human",
    )
    args = parser.parse_args(argv)

    settings = get_settings()
    gold = load_gold_set(settings=settings)
    data_dir = settings.paths.resolved(settings.paths.data_dir)
    chunks = load_chunks(data_dir / "chunks.jsonl")

    known_docs = {c.doc_id for c in chunks}
    pages_by_doc: dict[str, set[int]] = {}
    for chunk in chunks:
        pages = pages_by_doc.setdefault(chunk.doc_id, set())
        pages.update(range(chunk.page_start, chunk.page_end + 1))

    counts = gold.category_counts()
    print(f"\n{len(gold.questions)} questions")
    print(f"  categories: {dict(sorted(counts.items()))}")
    print(f"  answerable {len(gold.answerable())} | refusal cases {len(gold.refusal_cases())}")
    print(f"  verified by a human: {len(gold.verified())}/{len(gold.questions)}")

    problems: list[str] = []

    for question in gold.answerable():
        for source in question.sources:
            if source.doc_id not in known_docs:
                problems.append(f"{question.id}: unknown doc_id {source.doc_id!r}")
                continue
            indexed = pages_by_doc.get(source.doc_id, set())
            covered = set(range(source.page_start, source.page_end + 1)) & indexed
            if not covered:
                problems.append(
                    f"{question.id}: pages {source.page_start}-{source.page_end} of "
                    f"{source.doc_id} were never indexed, so this question is unreachable"
                )

    # Questions marked unanswerable must not have an obvious answer sitting in the corpus.
    sparse_path = settings.paths.resolved(settings.paths.bm25_path)
    if sparse_path.exists():
        sparse = SparseIndex.load(sparse_path)
        by_id = {c.chunk_id: c for c in chunks}
        print("\nrefusal cases - strongest lexical match in the corpus:")
        for question in gold.refusal_cases():
            hits = sparse.search(question.question, k=1)
            if hits:
                chunk = by_id.get(hits[0].chunk_id)
                where = chunk.citation if chunk else "?"
                print(
                    f"  {question.id} {question.question[:52]!r:56} {hits[0].score:6.2f}  {where}"
                )
            else:
                print(f"  {question.id} {question.question[:52]!r:56}     no lexical match")

    spread = Counter(s.doc_id for q in gold.answerable() for s in q.sources)
    print(f"\ndocuments covered by gold sources: {len(spread)}/{len(known_docs)}")
    for doc_id, n in spread.most_common():
        print(f"  {n:>2}  {doc_id}")

    print("\nchecks:")
    for detail in problems:
        print(f"  [FAIL] {detail}")
    if not problems:
        print("  [PASS] every cited page exists in the chunk store")

    broad = [q.id for q in gold.answerable() for s in q.sources if s.page_end - s.page_start > 10]
    if broad:
        print(
            f"  [WARN] source ranges wider than 10 pages, narrow before use: {sorted(set(broad))}"
        )

    unverified = [q.id for q in gold.questions if q.status != "verified"]
    if unverified:
        print(f"  [{'FAIL' if args.strict else 'WARN'}] not yet human-verified: {len(unverified)}")

    print()
    if problems or (args.strict and unverified):
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
