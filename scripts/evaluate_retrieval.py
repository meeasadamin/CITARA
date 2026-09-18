"""Measure retrieval against the gold set (feature 67, and the harness for feature 70).

Reports Hit Rate@k and MRR overall and per question category, because the aggregate hides
the thing worth knowing: hybrid retrieval is supposed to help identifier lookups
specifically, so a single averaged number cannot show whether it did.

The floor is disabled here. This measures whether retrieval *finds* the right page; whether
the system should then answer is the separate question the floor decides.

Run:  uv run python scripts/evaluate_retrieval.py [--mode hybrid_rerank]
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from datetime import UTC, datetime

from citara.config import get_settings
from citara.evaluation.gold import hit_rank, hit_rate_at_k, load_gold_set, mean_reciprocal_rank
from citara.log import configure_logging
from citara.retrieval.retriever import HybridRetriever


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Evaluate retrieval on the gold set.")
    parser.add_argument(
        "--mode",
        default=None,
        choices=["dense", "sparse", "hybrid", "hybrid_rerank"],
        help="retrieval configuration to measure (default: the configured mode)",
    )
    parser.add_argument("--out", default=None, help="write a JSON run record here")
    args = parser.parse_args(argv)

    settings = get_settings()
    configure_logging("WARNING")
    if args.mode:
        settings = settings.model_copy(
            update={
                "retrieval": settings.retrieval.model_copy(
                    # A floor of 0 keeps every candidate: this measures finding, not answering.
                    update={"mode": args.mode, "relevance_floor": 0.0}
                )
            }
        )
    else:
        settings = settings.model_copy(
            update={"retrieval": settings.retrieval.model_copy(update={"relevance_floor": 0.0})}
        )

    gold = load_gold_set(settings=settings)
    retriever = HybridRetriever(settings)
    mode = settings.retrieval.mode

    ranks: list[int | None] = []
    by_category: dict[str, list[int | None]] = defaultdict(list)
    rows: list[dict[str, object]] = []
    latencies: list[float] = []

    try:
        print(f"mode: {mode}  |  top_k={settings.retrieval.top_k}\n")
        for question in gold.answerable():
            outcome = retriever.retrieve(question.question, history=question.context or None)
            retrieved = [result.chunk for result in outcome.results]
            rank = hit_rank(question, retrieved)
            ranks.append(rank)
            by_category[question.category].append(rank)
            latencies.append(outcome.retrieval_ms + outcome.rerank_ms)

            found_by = {r.found_by for r in outcome.results[: rank or 0]} if rank else set()
            rows.append(
                {
                    "id": question.id,
                    "category": question.category,
                    "rank": rank,
                    "best_score": outcome.best_score,
                    "found_by": sorted(found_by),
                }
            )
            status = f"rank {rank}" if rank else "MISS"
            top = outcome.results[0] if outcome.results else None
            print(
                f"  {question.id:4} {question.category:12} {status:7} "
                f"score {outcome.best_score or 0:5.3f}  "
                f"via {top.found_by if top else '-':6}  {question.question[:44]}"
            )
    finally:
        retriever.close()

    print(
        f"\nHit Rate@1 {hit_rate_at_k(ranks, 1):.3f}   @3 {hit_rate_at_k(ranks, 3):.3f}   "
        f"@5 {hit_rate_at_k(ranks, 5):.3f}   MRR {mean_reciprocal_rank(ranks):.3f}"
    )

    print("\nby category:")
    for category, category_ranks in sorted(by_category.items()):
        print(
            f"  {category:12} n={len(category_ranks):2}  "
            f"hit@5 {hit_rate_at_k(category_ranks, 5):.2f}  "
            f"MRR {mean_reciprocal_rank(category_ranks):.3f}"
        )

    latencies.sort()
    if latencies:
        p50 = latencies[len(latencies) // 2]
        p95 = latencies[min(int(len(latencies) * 0.95), len(latencies) - 1)]
        print(f"\nlatency: p50 {p50:.0f} ms   p95 {p95:.0f} ms")

    record = {
        "generated_at": datetime.now(UTC).isoformat(),
        "mode": mode,
        "config_fingerprint": settings.fingerprint(),
        "hit_rate": {k: hit_rate_at_k(ranks, k) for k in settings.evaluation.hit_rate_k},
        "mrr": mean_reciprocal_rank(ranks),
        "questions": rows,
    }
    out = args.out or f"eval/runs/retrieval_{mode}.json"
    path = settings.paths.resolved(settings.paths.data_dir).parent / out
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(record, indent=2), encoding="utf-8")
    print(f"\nrun record: {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
