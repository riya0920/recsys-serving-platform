# Choosing the ANN index: the measurement, then the decision

Produced by `python -m recsys.bench_ann --n-items 50000 --queries 500 --k 100`.
Hardware: Windows 11, single-threaded FAISS CPU search, 64-d unit vectors, 64
Gaussian clusters (σ=0.35). Recall is measured against the **exact** top-100 from
the same vectors. Latency is **per query** (batch size 1), because at request
time there is exactly one user vector - batched throughput is not serving
latency.

| index | recall@100 | p50 (ms) | p95 (ms) | p99 (ms) | build (s) |
|---|---|---|---|---|---|
| flat (exact) | 1.0000 | 0.569 | 1.550 | 2.192 | 0.2 |
| ivf nprobe=4 | 0.1461 | 0.069 | 0.113 | 0.289 | 0.9 |
| ivf nprobe=16 | 0.3207 | 0.136 | 0.210 | 0.442 | 0.9 |
| ivf nprobe=64 | 0.6163 | 0.302 | 0.549 | 1.187 | 1.0 |
| hnsw ef=32 | 0.4907 | 0.396 | 0.533 | 0.648 | 6.2 |
| hnsw ef=64 | 0.6764 | 0.544 | 0.882 | 1.614 | 6.8 |
| hnsw ef=128 | 0.8543 | 0.968 | 1.565 | 2.463 | 6.5 |

## The decision: exact search, at this corpus size

**HNSW at ef=128 is strictly dominated here.** It gives up 15% of recall and is
*slower* at p50 (0.968 ms vs 0.569 ms) and at p99 (2.46 ms vs 2.19 ms) than
brute force, and costs 30x the build time. Shipping an approximate index at 50K
items would be cargo-culting: it buys nothing and costs recall.

`IndexConfig.kind` therefore defaults to `flat`.

## Why exact wins at this size, and where it stops winning

Exact search is 50,000 × 64 = 3.2M multiply-adds per query - one dense matmul
over ~13 MB of contiguous float32. That is a bandwidth-bound kernel that streams
at several GB/s and lands well inside a 100 ms budget. Graph traversal (HNSW)
instead does hundreds of dependent random reads; it wins on *asymptotics*, not on
constants, and 50K items is far below the crossover.

Rough crossover arithmetic: exact search scales linearly with catalogue size, so
the same hardware gives roughly

| items | exact p50 (extrapolated) | verdict |
|---|---|---|
| 50 K | 0.57 ms | exact, measured |
| 500 K | ~5.7 ms | exact still comfortable |
| 5 M | ~57 ms | exact eats over half the p99 budget → switch |
| 50 M | ~570 ms | approximate is mandatory; also no longer fits one node |

**The trigger to revisit is ~1-5M items**, and the extrapolated rows above are
labelled as extrapolation, not measurement - the rerun is one command.

## Why IVF looks so bad here

IVF's recall is poor at every nprobe tested because the vectors sit in 64 tight
clusters while `nlist=512`, so the true neighbours of a query are scattered
across many small cells and a low nprobe misses most of them. That is a
*tuning* result about this configuration, not a claim that IVF is a bad index.
Tuning `nlist` to the cluster structure would move the curve; it was not done
because the decision was already settled by the flat row.

## Caveats I would state before anyone quoted this table

* Synthetic clustered vectors, not learned item embeddings. Real embeddings have
  different intrinsic dimensionality, which is the main driver of ANN difficulty.
  The correct version of this table is regenerated from `artifacts/item_vectors.npy`
  after training, and that rerun is a roadmap item.
* Single-threaded, single-process, no concurrent load. Under real concurrency the
  flat index competes for memory bandwidth with every other request, and it will
  degrade faster than the graph index. That is precisely the measurement the
  load-test milestone exists to make.
* recall@100 measured against exact top-100 is *index* recall, not product
  recall@100. They are different numbers and the README does not mix them.
