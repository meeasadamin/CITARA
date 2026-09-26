"""Run an evaluation suite: ``uv run python -m citara.evaluation [suite]``.

Two suites, one command each.

- ``ablation`` (the default) measures retrieval across five configurations (feature 70). One
  command, one corpus, one gold set, every configuration - so the table describes a single
  consistent experiment rather than five sessions stitched together.
- ``generation`` answers every gold question once and scores what comes back for faithfulness,
  answer relevance and latency (features 68, 69, 71). It spends real model quota: roughly
  three requests per question.

Each suite writes a markdown artifact and the run record it was rendered from, so a number in
the artifact can always be traced back to the measurement that produced it.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import UTC, datetime
from pathlib import Path

from citara.config import Settings, get_settings
from citara.evaluation.ablation import (
    CONFIGURATIONS,
    AblationRow,
    render_markdown,
    run_configuration,
)
from citara.evaluation.generation import (
    GenerationReport,
    QuestionResult,
    evaluation_settings,
    run_generation,
)
from citara.evaluation.generation import render_markdown as render_generation
from citara.evaluation.gold import load_gold_set
from citara.evaluation.judge import JudgeUnavailable, build_judge
from citara.evaluation.models import GoldSet
from citara.log import configure_logging, get_logger

log = get_logger("evaluation.cli")

_DEFAULT_OUT = {"ablation": "eval/ABLATION.md", "generation": "eval/GENERATION.md"}
_RUN_RECORD = {"ablation": "ablation.json", "generation": "generation.json"}


def _artifact_path(settings: Settings, out: str) -> Path:
    root = settings.paths.resolved(settings.paths.data_dir).parent
    path = root / out
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def _runs_dir(settings: Settings) -> Path:
    runs = settings.paths.resolved(settings.evaluation.runs_dir)
    runs.mkdir(parents=True, exist_ok=True)
    return runs


def _ablation(args: argparse.Namespace, settings: Settings, gold: GoldSet) -> int:
    selected = [c for c in CONFIGURATIONS if not args.only or c.name in args.only]
    if not selected:
        names = ", ".join(c.name for c in CONFIGURATIONS)
        log.error("no matching configuration", extra={"available": names})
        return 2

    rows = []
    if args.render_only:
        # Re-rendering must never silently invent numbers: it reads the stored run only.
        stored = _runs_dir(settings) / _RUN_RECORD["ablation"]
        if not stored.exists():
            log.error("no stored run to render", extra={"path": str(stored)})
            return 2
        payload = json.loads(stored.read_text(encoding="utf-8"))
        measured_at = str(payload.get("generated_at", ""))
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
        measured_at = datetime.now(UTC).isoformat()
        for index, configuration in enumerate(selected, start=1):
            print(f"[{index}/{len(selected)}] {configuration.name} ...", flush=True)
            rows.append(run_configuration(configuration, settings, gold))

    markdown = render_markdown(rows, gold, measured_at)
    markdown_path = _artifact_path(settings, args.out or _DEFAULT_OUT["ablation"])
    markdown_path.write_text(markdown, encoding="utf-8")

    if args.render_only:
        print(markdown)
        print(f"written: {markdown_path} (from the stored run, nothing re-measured)")
        return 0

    record = {
        "generated_at": measured_at,
        "gold_questions": len(gold.answerable()),
        "refusal_cases": len(gold.refusal_cases()),
        "relevance_floor": settings.retrieval.relevance_floor,
        "rows": [row.as_dict() for row in rows],
    }
    stored = _runs_dir(settings) / _RUN_RECORD["ablation"]
    stored.write_text(json.dumps(record, indent=2), encoding="utf-8")

    print()
    print(markdown)
    print(f"written: {markdown_path}")
    print(f"run record: {stored}")
    return 0


def _report_from_record(payload: dict[str, object]) -> GenerationReport:
    """Rebuild a report from its stored run, so re-rendering measures nothing."""
    rows = payload.get("results", [])
    results = []
    for row in rows if isinstance(rows, list) else []:
        results.append(
            QuestionResult(
                id=row["id"],
                question=row["question"],
                category=row["category"],
                answerable=row["answerable"],
                mode=row["mode"],
                provider=row.get("provider", ""),
                model=row.get("model", ""),
                faithfulness=row.get("faithfulness"),
                relevance=row.get("relevance"),
                claims_checked=row.get("claims_checked", 0),
                unsupported=tuple(row.get("unsupported", ())),
                retrieval_ms=row.get("retrieval_ms", 0.0),
                generation_ms=row.get("generation_ms", 0.0),
                cited_sources=row.get("cited_sources", 0),
                fully_cited=row.get("fully_cited", False),
                error=row.get("error", ""),
            )
        )
    return GenerationReport(
        judge=str(payload.get("judge", "")),
        judge_model=str(payload.get("judge_model", "")),
        generator_model=str(payload.get("generator_model", "")),
        fingerprint=str(payload.get("fingerprint", "")),
        generated_at=str(payload.get("generated_at", "")),
        results=results,
    )


def _generation(args: argparse.Namespace, settings: Settings, gold: GoldSet) -> int:
    stored = _runs_dir(settings) / _RUN_RECORD["generation"]
    if args.render_only:
        if not stored.exists():
            log.error("no stored run to render", extra={"path": str(stored)})
            return 2
        report = _report_from_record(json.loads(stored.read_text(encoding="utf-8")))
    else:
        from citara.generation.answerer import Answerer

        measured = evaluation_settings(settings)
        name = args.judge or settings.evaluation.judge_provider
        try:
            judge = build_judge(measured, name)
        except JudgeUnavailable as error:
            log.error("judge unavailable", extra={"judge": name, "error": str(error)})
            return 2

        total = min(args.limit or len(gold.questions), len(gold.questions))
        print(
            f"scoring {total} question(s) with {judge.model} on {judge.name}; "
            "the answer cache is off, so every one spends model quota",
            flush=True,
        )

        def progress(index: int, count: int, question: object) -> None:
            label = getattr(question, "id", "")
            print(f"[{index}/{count}] {label} ...", flush=True)

        answerer = Answerer(measured)
        try:
            answerer.warm_up()
            report = run_generation(
                answerer, judge, gold, measured, limit=args.limit, on_progress=progress
            )
        finally:
            answerer.close()
        stored.write_text(json.dumps(report.as_dict(), indent=2), encoding="utf-8")

    markdown = render_generation(report, gold)
    markdown_path = _artifact_path(settings, args.out or _DEFAULT_OUT["generation"])
    markdown_path.write_text(markdown, encoding="utf-8")

    print()
    print(markdown)
    if args.render_only:
        print(f"written: {markdown_path} (from the stored run, nothing re-measured)")
    else:
        print(f"written: {markdown_path}")
        print(f"run record: {stored}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run an evaluation suite.")
    parser.add_argument(
        "suite",
        nargs="?",
        default="ablation",
        choices=("ablation", "generation"),
        help="retrieval ablation (default), or end-to-end generation quality and latency",
    )
    parser.add_argument("--only", nargs="*", help="ablation: run just these configurations")
    parser.add_argument("--out", default=None, help="markdown artifact to write")
    parser.add_argument(
        "--render-only",
        action="store_true",
        help="rebuild the markdown from the last run instead of measuring again",
    )
    parser.add_argument(
        "--judge",
        default=None,
        choices=("gemini", "groq"),
        help="generation: which provider scores the answers (default: evaluation.judge_provider)",
    )
    parser.add_argument(
        "--limit", type=int, default=None, help="generation: stop after this many questions"
    )
    args = parser.parse_args(argv)

    settings = get_settings()
    configure_logging(settings.log_level)
    gold = load_gold_set(settings=settings)

    if args.suite == "generation":
        return _generation(args, settings, gold)
    return _ablation(args, settings, gold)


if __name__ == "__main__":
    sys.exit(main())
