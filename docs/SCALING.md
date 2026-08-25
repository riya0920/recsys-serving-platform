# What breaks first at 25M interactions/hour

MovieLens-25M is 25 million interactions. A large consumer recommender does that
**per hour**. This is the arithmetic for what fails first, in the order it fails,
with the trigger for each.

Everything here is capacity *arithmetic* against measured single-node numbers - none of it is a measurement of a system at that scale, and it is labelled as such
throughout.

## The load

| | value |
|---|---|
| interactions | 25 M/hour = **6,944/sec** |
| serving requests | ~10x interactions (impressions ≫ clicks) = **~70K RPS** |
| catalogue | 10 M items |
| active users | 50 M |

## 1. Single-node FAISS memory - the first hard wall

Measured today: exact search over 50K × 64-d beats HNSW on both recall and
latency ([ANN_OPERATING_POINT.md](ANN_OPERATING_POINT.md)). That result does not
survive this scale.

```
10 M items x 64 dims x 4 bytes (fp32)   = 2.56 GB   raw vectors
HNSW graph overhead (M=32, ~8 bytes/edge, 2x M edges)
                     10 M x 64 x 8 B     = 5.12 GB
                                          --------
                                            7.7 GB per replica
```

7.7 GB **fits** on one machine - but every serving replica needs its own copy,
because the index is memory-resident. At 70K RPS you need ~20-40 replicas, so
this is 7.7 GB × N of pure duplication, and every index rebuild has to be shipped
to all of them.

**What breaks:** not capacity, **deploy time and cost**. Rebuilding and
distributing a 7.7 GB artifact to 40 replicas is a ~300 GB transfer per refresh,
which collides directly with the staleness result below.

**Trigger:** index > ~4 GB, or replica count > ~10.

**Fix, in order of preference:**
1. **Product quantisation** - PQ at 8× compression takes 2.56 GB → 320 MB of
   vectors at a measurable recall cost. The recall-vs-compression curve is the
   same shape as the existing exact-vs-HNSW table and is measured the same way.
2. **Sharded index by item partition**, scatter-gather across shards. Adds a
   network hop to every retrieval, which is why it comes second.
3. **Dedicated retrieval service** so the index is replicated on its own axis,
   independent of the stateless app tier.

## 2. Feature freshness vs rebuild cost - the tightest constraint

The [staleness experiment](../README.md#feature-staleness) measures this
directly. Under 4%/day taste drift, a frozen index loses **7.0% recall@10 by day
3 and 10.8% by day 7**, and under a stationary world it loses nothing.

That gives a real TTL rather than a guessed one: **refresh every 1-3 days** for a
population drifting at that rate, and *measure your own drift rate* before
adopting the number.

**The conflict:** §1 says the artifact is 7.7 GB and expensive to distribute;
this says refresh it every 1-3 days. At 40 replicas that is ~300 GB of transfer
every couple of days, and it is the single tightest coupling in the design.

**Fix:** incremental index updates (add/remove without a full rebuild), or
accept the recall cost and refresh weekly - a decision that now has a **measured
price tag** attached instead of an opinion.

## 3. Redis as a single point of failure

Measured today: the limiter is the dominant serving cost, and the SQLite backend
produces a 4310 ms p99 under 64 concurrent clients because `BEGIN IMMEDIATE`
takes a database-wide write lock
([LOADTEST.md](../../url-shortener-100x/docs/LOADTEST.md) documents the same
pattern in the sibling project).

At 70K RPS, a single Redis handles the *throughput* (Redis does ~100K ops/sec)
but becomes a hard availability dependency for the entire serving fleet.

**What breaks:** availability, not capacity.

**Fix:** Redis Cluster with hash-slot sharding by user id, plus the local-bucket
fallback described in the fail-open section. Approximate limits during a partial
outage beat no service.

**Trigger:** > 50K ops/sec, or when Redis availability becomes the binding
constraint on the service SLO.

## 4. Retraining cadence and cost

25 M interactions/hour = 600 M/day. The current two-tower trains on 250 K events
in ~2 minutes on a laptop CPU. Linearly extrapolated (and it is *labelled*
extrapolation - training does not scale linearly, and the real curve bends
against you):

```
600 M / 250 K x 2 min  =  ~80 hours of laptop-CPU-equivalent per day of data
```

Which is the actual argument for GPUs: not "GPUs are faster" but "the CPU
training time exceeds the data arrival rate, so the model can never catch up."

**Fix:** GPU training with the distributed setup measured in the sibling
[profiling study](../../profiling-training-study/docs/SCALING.md) - noting that
its efficiency numbers are CPU/gloo and do **not** transfer, only the method
does. Plus incremental/warm-start training rather than from-scratch every cycle.

**Trigger:** training wall-clock > the refresh interval from §2. That is the
condition, and it is checkable rather than a feeling.

## 5. The ranker becomes the latency budget

The ranker currently scores 200 candidates with 6 features via GBDT. At 70K RPS
that is **14 M candidate-scorings/sec**.

GBDT inference is ~1 µs per row single-threaded, so 14 M/sec needs ~14 cores of
pure ranking, before feature construction. Feature construction is the real cost:
today every feature is an in-memory lookup, which is exactly why it fits.

**What breaks:** the moment a feature requires a network call - a feature store
lookup, a real-time counter - the p99 budget is gone. One 5 ms feature-store
round trip inside a 100 ms budget is survivable; five sequential ones are not.

**Fix:** batch feature fetches, co-locate the feature store, and hold the line on
"every ranking feature must be computable from data already in hand." That
constraint is written into `ranker.py` deliberately.

## The order things break, and why that order

1. **Retraining cadence** (§4) - fails first, because it fails *silently*: the
   model just gets progressively staler and the recall loss looks like a product
   problem rather than an infrastructure one.
2. **Feature freshness vs rebuild cost** (§2) - the tightest *coupling*, and the
   one with a measured price.
3. **FAISS memory** (§1) - the first hard *wall*, but it announces itself loudly
   (OOM, slow deploys) rather than degrading quietly.
4. **Redis SPOF** (§3) - an availability cliff rather than a gradient.
5. **Ranker latency** (§5) - last, and only if someone adds a networked feature.

**The general rule this ordering reflects:** the failures that degrade *quietly*
are more dangerous than the ones that crash, because a crash gets fixed in an
hour and a quiet 10% recall loss gets attributed to the product for a quarter.

## What this document is not

It is arithmetic, not measurement. Every number here is derived from a measured
single-node figure and an extrapolation that is stated as one. The extrapolations
that would bend worst under reality are the training-time linearity in §4 and the
GBDT per-row cost in §5, and both are flagged in place rather than in a footnote.
