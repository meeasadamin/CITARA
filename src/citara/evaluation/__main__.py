"""Run the ablation study: ``uv run python -m citara.evaluation`` (feature 70).

One command, one corpus, one gold set, every configuration - so the table describes a single
consistent experiment rather than five sessions stitched together.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import UTC, datetime

from citara.config import get_settings
from citara.evaluation.ablation import (
    CONFIGURATIONS,
    AblationRow,
    render_markdown,
    run_configuration,
)
from citara.evaluation.gold import load_gold_set
from citara.log import configure_logging, get_logger

log = get_logger("evaluation.cli")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run the retrieval ablation study.")
    parser.add_argument("--only", nargs="*", help="run just these configurations, by name")
    parser.add_argument("--out", default="eval/ABLATION.md", help="markdown artifact to write")
    parser.add_argument(
        "--render-only",
        action="store_true",
        help="rebuild the markdown from the last run instead of measuring again",
    )
    args = parser.parse_args(argv)

    settings = get_settings()
    configure_logging(settings.log_level)
    gold = load_gold_set(settings=settings)

    selected = [c for c in CONFIGURATIONS if not args.only or c.name in args.only]
    if not selected:
        names = ", ".join(c.name for c in CONFIGURATIONS)
        log.error("no matching configuration", extra={"available": names})
        return 2

    rows = []
    if args.render_only:
        # Re-rendering must never silently invent numbers: it reads the stored run only.
        stored = settings.paths.resolved(settings.evaluation.runs_dir) / "ablation.json"
        if not stored.exists():
            log.error("no stored run to render", extra={"path": str(stored)})
            return 2
        payload = json.loads(stored.read_text(encoding="utf-8"))
        rows = [
            AblationRow(
                name=row["name"],
                mode=row["mode"],
                dense_weight=row["dense_weight"],
                sparse_weight=row["sparse_weight"],
                rationale=row["rationale"],
                fingerprint=row.get("fingerprint", ""),
                hit_rate={int(k): v for k, v in row["hit_rate"].items()},
                mrr=row["mrr"],
                by_category=row["by_category"],
                retrieval_p50_ms=row["retrieval_p50_ms"],
                retrieval_p95_ms=row["retrieval_p95_ms"],
                unanswerable_admitted=row["unanswerable_admitted"],
                unanswerable_total=row["unanswerable_total"],
            )
            for row in payload["rows"]
        ]
    else:
        for index, configuration in enumerate(selected, start=1):
            print(f"[{index}/{len(selected)}] {configuration.name} ...", flush=True)
            rows.append(run_configuration(configuration, settings, gold))

    markdown = render_markdown(rows, gold)
    root = settings.paths.resolved(settings.paths.data_dir).parent
    markdown_path = root / args.out
    markdown_path.parent.mkdir(parents=True, exist_ok=True)
    markdown_path.write_text(markdown, encoding="utf-8")

    runs_dir = settings.paths.resolved(settings.evaluation.runs_dir)
    runs_dir.mkdir(parents=True, exist_ok=True)
    if args.render_only:
        print(markdown)
        print(f"written: {markdown_path} (from the stored run, nothing re-measured)")
        return 0
    record = {
        "generated_at": datetime.now(UTC).isoformat(),
        "gold_questions": len(gold.answerable()),
        "refusal_cases": len(gold.refusal_cases()),
        "relevance_floor": settings.retrieval.relevance_floor,
        "rows": [row.as_dict() for row in rows],
    }
    (runs_dir / "ablation.json").write_text(json.dumps(record, indent=2), encoding="utf-8")

    print()
    print(markdown)
    print(f"written: {markdown_path}")
    print(f"run record: {runs_dir / 'ablation.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
