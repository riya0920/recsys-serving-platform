# Two-Stage Recommender Serving Platform

ANN candidate retrieval → learned ranker, behind an HTTP service whose model path
is designed to be killed. Offline evaluation uses a global time-based split; the
serving tier degrades to a popularity fallback instead of returning 5xx.

> **Status: ~40% built.** Retrieval, evaluation, the ANN benchmark harness and the
> degradable serving tier are implemented and tested. The ranker, shadow
> deployment, load tests and the measured latency/throughput numbers are not —
> see [Roadmap](#roadmap). Nothing below is reported as a measured result until
> it is. There are no A/B results in this repo, only offline metrics.

## What is here

```
src/recsys/config.py     every tunable knob, one place
src/recsys/data.py       event generator (zipf popularity, latent tastes) + time split
src/recsys/model.py      two-tower, in-batch negatives, sampling-bias (logQ) correction
src/recsys/index.py      FAISS behind one interface: flat | IVF | HNSW
src/recsys/bench_ann.py  recall-vs-latency curve -> the operating point in config.py
src/recsys/eval.py       recall@k / NDCG@k, one protocol, popularity baseline
src/recsys/train.py      end-to-end: train -> index -> offline eval -> artifacts/
src/service/app.py       FastAPI two-stage serving with a killable model path
docs/                    split justification, ANN operating point, omitted scope
```

## Run it

```bash
pip install -r requirements.txt
make test                                    # unit tests
PYTHONPATH=src python -m recsys.train --epochs 3
PYTHONPATH=src python -m recsys.bench_ann --n-items 50000 --queries 1000
make serve                                   # http://localhost:8000/docs
```

Point it at real data instead of the generator:

```bash
PYTHONPATH=src python -m recsys.train --ratings-csv /path/to/ml-25m/ratings.csv
```

## The three decisions worth reading

**1. Time-based split, global boundaries.** A random split leaks the future
through three separate channels, one of which survives even a per-user time
split. Written up in [docs/SPLIT_JUSTIFICATION.md](docs/SPLIT_JUSTIFICATION.md),
including what the choice costs (cold entities, a single seasonal window) and
what would change it.

**2. In-batch negatives with a logQ correction.** In-batch negatives are drawn
from the interaction distribution, so popular items appear as negatives far more
often than uniform sampling implies and the model over-penalises them. Logits are
corrected by `-log Q(item)` (Yi et al., RecSys'19). `--no-logq` runs the ablation.
`test_logq_correction_demotes_popular_items` asserts the correction moves logits
in the intended direction rather than trusting that it does.

**3. The model is a dependency, not the service.** `/recommend` catches every
failure in the model path and serves popular items with `degraded: true` and a
200. `/readyz` deliberately does *not* gate on the model — gating readiness on a
model takes the whole fleet out during a bad rollout. `POST /admin/kill-model`
makes this demonstrable rather than claimed.

## Benchmark harness

`bench_ann.py` sweeps flat / IVF (nprobe 4, 16, 64) / HNSW (ef 32, 64, 128) and
reports recall against the exact top-k with per-query p50/p95/p99. Two details it
gets right that most benchmark tables do not:

* **Clustered vectors, not uniform noise.** Uniform random vectors in high
  dimensions make every index look identical and perfect.
* **Per-query latency, not batched.** At request time there is exactly one user
  vector. Quoting batched throughput as serving latency is the standard error.

The measured result is written up in
[docs/ANN_OPERATING_POINT.md](docs/ANN_OPERATING_POINT.md), and it is not the
answer the tutorials give: **at 50K items exact search beats HNSW on both recall
and latency** (1.000 vs 0.854 recall@100, 0.57 ms vs 0.97 ms p50), so
`IndexConfig.kind` defaults to `flat`. The doc states the crossover where that
stops being true (~1-5M items) and labels the extrapolated rows as extrapolation.
Shipping an approximate index at this corpus size would have been cargo-culting.

## Roadmap (the remaining ~60%)

| Milestone | Status |
|---|---|
| Two-tower retrieval + logQ correction | done |
| Time-split protocol + cold-start reporting | done |
| ANN benchmark harness (recall vs latency) | done |
| Degradable serving tier + chaos hook | done |
| **Learned ranker over the top-500 candidates** | not started |
| **Load test (zipf mix), max RPS at p99 < 100ms on named hardware** | not started |
| **Chaos drill artifact: kill model mid-load-test, 0 errors** | hook exists, drill not scripted |
| **Shadow deployment + gated promotion + automated rollback** | not started |
| **MLflow experiment lineage (10+ runs)** | not started |
| **Feature-staleness experiment (0/1/7-day) to justify a TTL** | not started |
| **`docs/SCALING.md`: what breaks first at 25M interactions/hour** | not started |

## Honesty notes

* All metrics are **offline**. No online experiment has been run.
* The default dataset is **synthetic** with a zipf popularity prior and latent
  per-user tastes. It exists so the pipeline is runnable end to end without a
  25M-row download; `--ratings-csv` switches to real MovieLens with an identical
  schema. Numbers from the generator are labelled as such and are not evidence
  about real user behaviour.
* The only latency numbers here are **index search latency**, single-threaded and
  unloaded (see the caveats in the ANN doc). No end-to-end serving latency and no
  throughput number is quoted, because the load test has not been run yet.
