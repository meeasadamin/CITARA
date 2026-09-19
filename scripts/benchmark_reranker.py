"""Can the cross-encoder gate run faster without changing what it admits?

Reranking is nearly all of a warm query's latency - about 3.4 s of it on a laptop CPU - so it
is the obvious thing to speed up. But the reranker is the admissibility gate: its score is what
the calibrated floor thresholds. A faster gate that admits a different set of questions is not
an optimisation, it is a different safety system that nobody calibrated.

So each variant is judged on two things, on the exact (question, passage) pairs the live gate
scores for every gold question:

* latency, timed interleaved so machine noise lands on every variant alike, and
* whether it reaches the same admit/refuse decision as the current model, per question and
  per chunk, at the calibrated floor.

Variants: the current PyTorch model at its default and at every logical core, the ONNX export
published in the model's own repository, and a dynamically quantised int8 version of it.

Run:  uv run --with onnx python scripts/benchmark_reranker.py
(``onnx`` is needed only to quantise; ``onnxruntime`` already ships with ChromaDB.)
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time
from collections.abc import Callable
from pathlib import Path

import numpy as np

from citara.config import get_settings
from citara.evaluation.gold import load_gold_set
from citara.log import configure_logging
from citara.retrieval.models import RetrievedChunk
from citara.retrieval.reranker import load_reranker
from citara.retrieval.retriever import HybridRetriever

Scorer = Callable[[str, list[str]], list[float]]


def capture_pairs(settings) -> dict[str, dict]:  # type: ignore[no-untyped-def]
    """The query and passages the gate actually scores, for every gold question."""
    gold = load_gold_set(settings=settings)
    retriever = HybridRetriever(settings)
    captured: dict[str, dict] = {}
    original = retriever.reranker.rerank
    current = {"id": ""}

    def spy(query: str, candidates: list[RetrievedChunk]) -> list[RetrievedChunk]:
        captured[current["id"]] = {
            "query": query,
            "passages": [c.chunk.content for c in candidates],
        }
        return original(query, candidates)

    retriever.reranker.rerank = spy  # type: ignore[method-assign]
    try:
        for question in gold.questions:
            current["id"] = question.id
            retriever.retrieve(question.question, history=question.context or None)
            if question.id in captured:
                captured[question.id]["answerable"] = question.answerable
    finally:
        retriever.close()
    return captured


def onnx_scorers(model_name: str, work_dir: Path) -> dict[str, Scorer]:
    """ONNX Runtime scorers for the published fp32 export and an int8 quantisation of it."""
    import onnxruntime as ort
    from huggingface_hub import hf_hub_download
    from onnxruntime.quantization import QuantType, quantize_dynamic
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    fp32 = hf_hub_download(model_name, "onnx/model.onnx")
    int8 = work_dir / "reranker_int8.onnx"
    if not int8.exists():
        work_dir.mkdir(parents=True, exist_ok=True)
        quantize_dynamic(fp32, str(int8), weight_type=QuantType.QInt8)

    def scorer(path: str) -> Scorer:
        options = ort.SessionOptions()
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        session = ort.InferenceSession(path, options, providers=["CPUExecutionProvider"])
        names = [i.name for i in session.get_inputs()]

        def score(query: str, passages: list[str]) -> list[float]:
            encoded = tokenizer(
                [query] * len(passages),
                passages,
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors="np",
            )
            logits = session.run(None, {n: encoded[n].astype(np.int64) for n in names})[0]
            # The same sigmoid CrossEncoder applies, so scores share the floor's scale.
            return [float(x) for x in 1.0 / (1.0 + np.exp(-logits.reshape(-1)))]

        return score

    return {"onnx_fp32": scorer(fp32), "onnx_int8": scorer(str(int8))}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Benchmark reranker backends against the gate.")
    parser.add_argument("--out", default="eval/runs/reranker_backends.json")
    parser.add_argument("--repeats", type=int, default=2)
    args = parser.parse_args(argv)

    import torch

    settings = get_settings()
    configure_logging("WARNING")
    config = settings.retrieval
    floor = config.relevance_floor

    print("capturing the pairs the gate scores for each gold question...")
    cases = capture_pairs(settings)

    model = load_reranker(config.reranker_model)
    default_threads = torch.get_num_threads()
    logical = os.cpu_count() or default_threads

    def torch_scorer(threads: int) -> Scorer:
        def score(query: str, passages: list[str]) -> list[float]:
            torch.set_num_threads(threads)
            pairs = [(query, p) for p in passages]
            out = model.predict(
                pairs, batch_size=config.reranker_batch_size, show_progress_bar=False
            )
            return [float(x) for x in out]

        return score

    variants: dict[str, Scorer] = {"torch": torch_scorer(default_threads)}
    if logical != default_threads:
        variants[f"torch_{logical}_threads"] = torch_scorer(logical)
    models_dir = settings.paths.resolved(settings.paths.data_dir) / "models"
    try:
        variants.update(onnx_scorers(config.reranker_model, models_dir))
    except ImportError as error:
        print(f"skipping ONNX variants ({error}); add: --with onnx", file=sys.stderr)

    for score in variants.values():
        score("warm up", ["warm up", "again"])

    scores: dict[str, dict[str, list[float]]] = {name: {} for name in variants}
    timings: dict[str, list[float]] = {name: [] for name in variants}
    for _ in range(args.repeats):
        for qid, case in cases.items():
            for name, score in variants.items():
                started = time.perf_counter()
                scores[name][qid] = score(case["query"], case["passages"])
                timings[name].append((time.perf_counter() - started) * 1000)
    torch.set_num_threads(default_threads)

    def admitted(values: list[float]) -> bool:
        return sum(v >= floor for v in values) >= config.min_evidence_chunks

    reference = scores["torch"]
    report: dict[str, dict] = {}
    for name in variants:
        flips = []
        chunk_same = order_same = 0
        diffs = []
        for qid, case in cases.items():
            a, b = reference[qid], scores[name][qid]
            diffs.append(max(abs(x - y) for x, y in zip(a, b, strict=True)))
            chunk_same += [x >= floor for x in a] == [y >= floor for y in b]
            order_same += list(np.argsort(-np.array(a))) == list(np.argsort(-np.array(b)))
            if admitted(a) != admitted(b):
                flips.append(
                    {
                        "id": qid,
                        "answerable": case["answerable"],
                        "reference_best": round(max(a), 4),
                        "variant_best": round(max(b), 4),
                        "now": "admitted" if admitted(b) else "refused",
                    }
                )
        report[name] = {
            "median_ms": round(statistics.median(timings[name])),
            "p95_ms": round(float(np.percentile(timings[name], 95))),
            "max_abs_score_diff": round(max(diffs), 5),
            "gate_decisions_same": len(cases) - len(flips),
            "chunk_admissions_same": chunk_same,
            "rankings_same": order_same,
            "flipped": flips,
        }

    n = len(cases)
    print(f"\n{'variant':22} {'median':>8} {'p95':>8} {'gate same':>10} {'chunks same':>12}  flips")
    for name, row in report.items():
        flipped = ", ".join(f"{f['id']}->{f['now']}" for f in row["flipped"]) or "-"
        gate = f"{row['gate_decisions_same']}/{n}"
        chunks = f"{row['chunk_admissions_same']}/{n}"
        print(
            f"{name:22} {row['median_ms']:>6} ms {row['p95_ms']:>5} ms "
            f"{gate:>10} {chunks:>12}  {flipped}"
        )

    payload = {
        "config_fingerprint": settings.fingerprint(),
        "reranker": config.reranker_model,
        "relevance_floor": floor,
        "questions": n,
        "repeats": args.repeats,
        "cpu_count": logical,
        "torch_default_threads": default_threads,
        "variants": report,
    }
    out_path = settings.paths.resolved(settings.paths.data_dir).parent / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"\nrun record: {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
