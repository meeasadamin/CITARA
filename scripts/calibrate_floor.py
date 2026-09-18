"""Calibrate the relevance floor on the gold set (feature 29).

The floor decides whether the system answers or refuses, so picking it by intuition would
make refusal arbitrary. This measures the two distributions it has to separate:

* the reranker score of the best chunk for questions the corpus *can* answer, and
* the reranker score of the best chunk for questions it cannot - which are not zero, because
  retrieval happily returns confident-looking passages for questions with no answer.

It then reports, for every candidate threshold, how many answerable questions would be
wrongly refused and how many unanswerable ones would be wrongly answered. In disaster
response those two errors are not symmetric: answering when the corpus is silent is the
failure that puts a wrong evacuation route in front of an officer.

Run:  uv run python scripts/calibrate_floor.py
"""

from __future__ import annotations

import argparse
import json
import sys
from itertools import pairwise

from citara.config import get_settings
from citara.evaluation.gold import hit_rank, load_gold_set
from citara.log import configure_logging
from citara.retrieval.retriever import HybridRetriever


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Calibrate the relevance floor.")
    parser.add_argument("--out", default="eval/runs/floor_calibration.json")
    args = parser.parse_args(argv)

    settings = get_settings()
    configure_logging("WARNING")
    gold = load_gold_set(settings=settings)
    retriever = HybridRetriever(settings)

    answerable: list[tuple[str, float]] = []
    refusable: list[tuple[str, float]] = []

    # Refusing a question whose evidence retrieval never found is the *correct* outcome, not
    # a cost: answering it would mean answering without support. Only a question whose gold
    # page was actually retrieved, and is then refused, is a real loss.
    had_evidence: dict[str, bool] = {}

    try:
        print(f"scoring {len(gold.questions)} questions through the full funnel...\n")
        for question in gold.questions:
            # The floor is being measured, so it must not filter the evidence here.
            outcome = retriever.retrieve(question.question, history=question.context or None)
            best = outcome.best_score if outcome.best_score is not None else 0.0
            row = (question.id, float(best))
            (answerable if question.answerable else refusable).append(row)
            had_evidence[question.id] = (
                hit_rank(question, [r.chunk for r in outcome.results]) is not None
            )
            marker = "ANSWERABLE " if question.answerable else "REFUSE     "
            evidence = (
                ""
                if not question.answerable
                else (
                    "  [gold retrieved]" if had_evidence[question.id] else "  [gold NOT retrieved]"
                )
            )
            print(f"  {marker} {question.id:4} {best:6.3f}  {question.question[:44]}{evidence}")
    finally:
        retriever.close()

    if not answerable or not refusable:
        print("\nNeed both answerable and unanswerable questions to calibrate.")
        return 1

    answerable_scores = sorted(score for _, score in answerable)
    refusable_scores = sorted(score for _, score in refusable)

    print("\nscore distributions")
    print(
        f"  answerable  n={len(answerable_scores):2}  min {answerable_scores[0]:.3f}  "
        f"median {answerable_scores[len(answerable_scores) // 2]:.3f}  "
        f"max {answerable_scores[-1]:.3f}"
    )
    print(
        f"  unanswerable n={len(refusable_scores):2}  "
        f"min {refusable_scores[0]:.3f}  median {refusable_scores[len(refusable_scores) // 2]:.3f}"
        f"  max {refusable_scores[-1]:.3f}"
    )

    # Thresholds sit *between* observed scores. Testing at an observed value is ambiguous,
    # because admission uses >=: a floor of exactly 0.302 still admits the chunk scoring
    # 0.302, which is the unanswerable question it was meant to exclude.
    observed = sorted({round(s, 4) for s in answerable_scores + refusable_scores})
    midpoints = [0.0] + [round((low + high) / 2, 4) for low, high in pairwise(observed)]
    midpoints.append(round(observed[-1] + 0.01, 4))

    print("\nthreshold   wrongly refused   wrongly answered   (lower is better for both)")
    best_threshold = None
    best_cost = None
    for threshold in midpoints:
        wrongly_refused = sum(1 for s in answerable_scores if s < threshold)
        wrongly_answered = sum(1 for s in refusable_scores if s >= threshold)
        # Answering when the corpus is silent is the more dangerous error: an officer acting
        # on an unsupported answer is a casualty event, while a refusal costs a lookup. Ties
        # break toward the higher - safer - threshold for the same reason.
        cost = wrongly_refused + 3 * wrongly_answered
        marker = ""
        if best_cost is None or cost <= best_cost:
            best_cost, best_threshold, marker = cost, threshold, "  <- best so far"
        print(f"  {threshold:7.4f}   {wrongly_refused:>13}   {wrongly_answered:>16}{marker}")

    if best_threshold is not None:
        genuine_losses = [
            qid for qid, score in answerable if score < best_threshold and had_evidence.get(qid)
        ]
        correct_refusals = [
            qid for qid, score in answerable if score < best_threshold and not had_evidence.get(qid)
        ]
        print(
            f"\nat that threshold: {len(genuine_losses)} answerable question(s) refused despite "
            f"having the right evidence {genuine_losses}"
        )
        print(
            f"                   {len(correct_refusals)} refused where retrieval had already "
            f"missed, which is the correct outcome {correct_refusals}"
        )

    print(f"\nrecommended relevance_floor: {best_threshold}")
    print(f"  set with: CITARA_RETRIEVAL__RELEVANCE_FLOOR={best_threshold}")
    print(f"  current setting: {settings.retrieval.relevance_floor}")

    payload = {
        "config_fingerprint": settings.fingerprint(),
        "reranker": settings.retrieval.reranker_model,
        "answerable": answerable,
        "unanswerable": refusable,
        "recommended_floor": best_threshold,
    }
    out_path = settings.paths.resolved(settings.paths.data_dir).parent / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"  written to {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
