# Retrieval evaluation

Measured on the gold question set (24 answerable questions, 6 refusal cases) against the
indexed corpus of 3,141 chunks from 19 NDMA and UNDRR documents.

Reproduce with:

```bash
uv run python scripts/evaluate_retrieval.py --mode dense      # or sparse | hybrid | hybrid_rerank
uv run python scripts/calibrate_floor.py
```

## Ablation

Hit Rate counts a question as found when a retrieved chunk covers a gold page. The floor is
disabled for these runs: this measures whether retrieval *finds* the evidence, which is a
separate question from whether the system should answer.

| Configuration | Hit@1 | Hit@3 | Hit@5 | MRR | Latency p50 |
|---|---|---|---|---|---|
| Dense only | 0.125 | 0.333 | **0.458** | 0.238 | 68 ms |
| Sparse only (BM25) | 0.083 | 0.250 | 0.333 | 0.181 | 32 ms |
| Hybrid (even RRF) | 0.167 | 0.250 | 0.375 | 0.228 | 43 ms |
| Hybrid + cross-encoder | 0.125 | 0.292 | 0.375 | 0.206 | 4,251 ms |
| **Dense + cross-encoder gate** | **0.167** | 0.292 | **0.458** | **0.255** | 4,175 ms |

### What this says

**Hybrid fusion did not beat dense retrieval here.** The specification called hybrid "the
single most defensible engineering decision in the project"; on this gold set it is not.
Even fusion costs eight percentage points of Hit@5 against dense alone. BM25 still runs and
its candidates remain in the pool, but it is weighted out of the ranking by default, because
shipping a configuration the measurement contradicts would be decoration rather than
engineering.

**The cross-encoder is a gate, not a ranker.** Sweeping how much to trust it against fusion
order moved Hit@5 between 0.375 and 0.417 and MRR between 0.249 and 0.274 - noise at this
sample size. Its scores do separate answerable from unanswerable questions cleanly, so it
now scores only the chunks about to be served and decides admissibility. That cut its cost
from 19.6 s to 4.2 s per query.

**Absolute quality is modest and the reason is known.** A funnel diagnostic showed 16 of 24
gold pages reaching the candidate pool, so the loss is in ranking rather than recall. The
per-category breakdown says where: follow-up questions score 0.00 across every configuration,
including with model-based query rewriting active, so that failure is in retrieval rather
than in rewriting.

**The sample is small.** Twenty-four questions means a two-question swing moves Hit@5 by
0.083. These differences are suggestive, not settled, and the ablation should be re-run when
the gold set grows.

## Relevance floor

Calibrated rather than chosen, by scoring every gold question through the full funnel and
comparing the two distributions the floor has to separate.

| | n | min | median | max |
|---|---|---|---|---|
| Answerable | 24 | 0.082 | 0.920 | 0.999 |
| Unanswerable | 6 | 0.001 | 0.077 | 0.302 |

At the calibrated floor of **0.3386**:

- **0** unanswerable questions admitted, including "What is the capital of France?", which
  BM25 still scores at 14.41 - retrieval returns confident-looking evidence for questions the
  corpus cannot answer, which is exactly what the floor exists to catch.
- **0** answerable questions refused despite their evidence being retrieved.
- **4** refused where retrieval had already missed the evidence. Answering those would mean
  answering without support, so refusing them is the correct outcome rather than a cost.

Thresholds are tested at midpoints between observed scores, because admission uses `>=` and a
threshold sitting exactly on an observed score admits it - an earlier calibration let through
the unanswerable question scoring 0.302 for precisely that reason.

Re-run `scripts/calibrate_floor.py` after any change to chunking, retrieval or the reranker:
the floor is a property of the whole pipeline, not of the model alone.
