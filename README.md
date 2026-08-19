# Two-Stage Recommender Serving Platform

ANN candidate retrieval → learned ranker, behind an HTTP service whose model path
is designed to be killed. Offline evaluation uses a global time-based split; the
serving tier degrades to a popularity fallback instead of returning 5xx.

> **Status: ~70% built.** Retrieval, the **learned stage-2 ranker**, evaluation,
> the ANN benchmark, **MLflow experiment lineage**, the degradable serving tier
> and a **chaos drill run under live load** are implemented and measured. Shadow
> deployment, automated rollback and the throughput benchmark are not — see
> [Roadmap](#roadmap). Every metric here is **offline**; there are no A/B results
> in this repo.

## Stage 2: the learned ranker

Retrieval optimises recall over 50K items with one dot product. Ranking optimises
precision over ~200 candidates and can afford features that would be impossible
at corpus scale. Measured on the same users, the same candidate slates, so the
only difference is the ordering:

| | retrieval only | + ranker | lift |
|---|---|---|---|
| NDCG@10 | 0.0923 | **0.1079** | **+16.9%** |
| recall@10 | 0.1211 | **0.1388** | +14.6% |

**The training-data decision matters more than the model choice.** Negatives are
sampled from the *retriever's own top-K*, not uniformly from the catalogue. A
ranker trained on uniform negatives learns to separate "plausible" from "absurd"
— which retrieval already did — so at serving time, where it only ever sees
plausible items, its training and serving distributions disagree and the offline
lift evaporates. `test_training_negatives_come_from_the_retrieved_slate` pins it.

**The ranker is trained on the validation window and scored on test.** Training
it on the window it is evaluated on leaks, and the leak is invisible in the
metric — it just looks like a very good ranker.

**Features are deliberately cheap**: six values that are all lookups or
arithmetic on things already in hand at request time. No joins, no second model,
no feature-store round trip. A ranker whose features cannot be computed inside
the latency budget is a research artifact.

**GBDT, not a neural ranker**, because with six tabular features trees win on
quality per unit of effort and are far cheaper to serve. The documented trigger
for revisiting is high-cardinality or sequential features.

## Experiment lineage

Every ranker run is tracked in MLflow (SQLite backend), so comparing two
configurations is a query rather than a memory:

| candidates | negatives/pos | lr | depth | NDCG lift |
|---|---|---|---|---|
| 200 | 4 | 0.05 | 4 | **+20.9%** |
| 200 | 4 | 0.05 | 8 | +19.9% |
| 200 | 4 | 0.12 | 4 | +19.1% |
| 200 | 4 | 0.12 | 8 | +16.2% |
| 100 | 6 | 0.08 | 6 | +16.2% |
| 300 | 2 | 0.08 | 6 | +11.0% |
| 100 | 2 | 0.08 | 6 | +9.7% |

The clearest signal in the sweep is **negatives per positive**, not the tree
hyperparameters: dropping from 4 to 2 costs more lift than any depth or
learning-rate change tested. That is consistent with the hard-negative argument
above and is the kind of thing a tracked sweep tells you and a remembered one
does not.

## The chaos drill, run under live load

The differentiator the spec asks for — model dies mid-load-test, fallback
engages, zero errors — as a **re-runnable script** rather than a GIF, because a
script can be verified and a GIF cannot:

```
$ make chaos

total_requests   7,243
total_errors         0

  healthy        req= 1570  errors=0  degraded=0.0000  p50= 94.2ms  p99=1259.2ms
  model_killed   req= 3553  errors=0  degraded=0.9963  p50= 49.5ms  p99= 151.7ms
  recovered      req= 2120  errors=0  degraded=0.0047  p50= 85.6ms  p99= 147.7ms

PASSED: True
```

Zero 5xx and zero connection errors across all three phases. During the outage
**99.6% of requests were served from the fallback and none were empty**; after
revival the service returns to the model path on its own.

Note the fallback is *faster* than the model path (49.5 ms vs 94.2 ms p50),
which is exactly what you want from a degradation path — it must not itself
become the bottleneck under the conditions that triggered it.

**A bug the drill found in its own harness:** the first run reported 32 errors
that looked like service failures. They were connection refusals during the
42-second startup while uvicorn imports torch and reads the FAISS index. The
drill now waits for `/healthz` before it starts, because counting startup
artifacts as errors makes a passing drill look like a failing one.

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
| Learned stage-2 ranker with hard-negative sampling | done |
| Two-stage offline evaluation on identical candidate slates | done |
| MLflow experiment lineage across a tracked sweep | done |
| Chaos drill under live load: 7,243 requests, 0 errors | done |
| **Load test: max RPS at p99 < 100ms on named hardware** | not measured |
| **Shadow deployment + gated promotion + automated rollback** | not started |
| **Feature-staleness experiment (0/1/7-day) to justify a TTL** | not started |
| **`docs/SCALING.md`: what breaks first at 25M interactions/hour** | not started |

## Honesty notes

* All metrics are **offline**. No online experiment has been run.
* The default dataset is **synthetic** with a zipf popularity prior and latent
  per-user tastes. It exists so the pipeline is runnable end to end without a
  25M-row download; `--ratings-csv` switches to real MovieLens with an identical
  schema. Numbers from the generator are labelled as such and are not evidence
  about real user behaviour.
* **The chaos drill's latency figures are from a 32-client run on a laptop that
  was also generating the load.** They demonstrate that degradation works, not
  what this service can sustain. **No max-RPS number is claimed**, because the
  throughput benchmark has not been run.
* The ANN table's latencies remain single-threaded and unloaded index search.
* The ranker lift is **offline**, on a synthetic corpus, against the same
  candidate slates. It is not evidence about live user behaviour, and no A/B
  test has been run.
