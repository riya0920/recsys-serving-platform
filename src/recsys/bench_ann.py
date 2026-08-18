"""ANN benchmark: exact vs IVF vs HNSW, recall@k against the exact result set.

This is a design-review artifact, not a demo. It answers the only question that
matters when you pick an index: how much recall am I buying back per millisecond,
and where do I want to sit on that curve.

    python -m recsys.bench_ann --n-items 50000 --dim 64 --queries 1000

Writes artifacts/ann_bench.json and prints a markdown table you can paste into
docs/. The chosen operating point is recorded in config.IndexConfig.
"""
from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np

from .config import IndexConfig
from .index import VectorIndex

ART = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "artifacts"))


def _normalise(x: np.ndarray) -> np.ndarray:
    return (x / np.linalg.norm(x, axis=1, keepdims=True)).astype("float32")


def exact_topk(vectors: np.ndarray, queries: np.ndarray, k: int) -> np.ndarray:
    """Ground truth for recall. Chunked so it stays honest on 50K+ items."""
    out = np.empty((queries.shape[0], k), dtype=np.int64)
    for s in range(0, queries.shape[0], 256):
        block = queries[s : s + 256] @ vectors.T
        out[s : s + 256] = np.argpartition(-block, k, axis=1)[:, :k]
        rows = np.arange(out[s : s + 256].shape[0])[:, None]
        order = np.argsort(-block[rows, out[s : s + 256]], axis=1)
        out[s : s + 256] = out[s : s + 256][rows, order]
    return out


def recall_vs_exact(approx: np.ndarray, exact: np.ndarray) -> float:
    hits = [len(set(a.tolist()) & set(e.tolist())) for a, e in zip(approx, exact)]
    return float(np.mean(hits) / exact.shape[1])


def timed_search(index: VectorIndex, queries: np.ndarray, k: int, repeats: int = 3):
    """Per-query latency percentiles, measured one query at a time.

    Batch search flatters an index in a way that serving never does: at request
    time we have exactly one user vector. Measuring batched throughput here and
    quoting it as serving latency is the most common lie in this kind of table.
    """
    lat = []
    for _ in range(repeats):
        for q in queries:
            t0 = time.perf_counter()
            index.search(q.reshape(1, -1), k)
            lat.append((time.perf_counter() - t0) * 1000.0)
    lat = np.array(lat)
    return {
        "p50_ms": float(np.percentile(lat, 50)),
        "p95_ms": float(np.percentile(lat, 95)),
        "p99_ms": float(np.percentile(lat, 99)),
    }


def run(n_items: int, dim: int, n_queries: int, k: int, seed: int = 7) -> dict:
    rng = np.random.default_rng(seed)
    # Clustered vectors, not uniform noise: uniform random vectors in high
    # dimensions make every ANN index look identical (and perfect), which is why
    # so many published benchmarks are meaningless.
    centres = _normalise(rng.normal(size=(64, dim)))
    assign = rng.integers(0, 64, size=n_items)
    vectors = _normalise(centres[assign] + 0.35 * rng.normal(size=(n_items, dim)))
    queries = _normalise(centres[rng.integers(0, 64, size=n_queries)] + 0.35 * rng.normal(size=(n_queries, dim)))

    truth = exact_topk(vectors, queries, k)

    grid = [
        ("flat", IndexConfig(kind="flat")),
        ("ivf nprobe=4", IndexConfig(kind="ivf", ivf_nprobe=4)),
        ("ivf nprobe=16", IndexConfig(kind="ivf", ivf_nprobe=16)),
        ("ivf nprobe=64", IndexConfig(kind="ivf", ivf_nprobe=64)),
        ("hnsw ef=32", IndexConfig(kind="hnsw", hnsw_ef_search=32)),
        ("hnsw ef=64", IndexConfig(kind="hnsw", hnsw_ef_search=64)),
        ("hnsw ef=128", IndexConfig(kind="hnsw", hnsw_ef_search=128)),
    ]

    rows = []
    for label, cfg in grid:
        t0 = time.perf_counter()
        idx = VectorIndex(cfg, dim).build(vectors)
        build_s = time.perf_counter() - t0
        _, ids = idx.search(queries, k)
        row = {"index": label, "recall@%d" % k: recall_vs_exact(ids, truth), "build_s": build_s}
        row.update(timed_search(idx, queries[: min(200, n_queries)], k))
        rows.append(row)
        print("%-16s recall=%.4f  p50=%.3fms  p99=%.3fms  build=%.1fs"
              % (label, row["recall@%d" % k], row["p50_ms"], row["p99_ms"], build_s))

    return {"params": {"n_items": n_items, "dim": dim, "n_queries": n_queries, "k": k}, "rows": rows}


def to_markdown(result: dict) -> str:
    k = result["params"]["k"]
    head = "| index | recall@%d | p50 (ms) | p95 (ms) | p99 (ms) | build (s) |" % k
    sep = "|---|---|---|---|---|---|"
    lines = [head, sep]
    for r in result["rows"]:
        lines.append("| %s | %.4f | %.3f | %.3f | %.3f | %.1f |"
                     % (r["index"], r["recall@%d" % k], r["p50_ms"], r["p95_ms"], r["p99_ms"], r["build_s"]))
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-items", type=int, default=50_000)
    ap.add_argument("--dim", type=int, default=64)
    ap.add_argument("--queries", type=int, default=1000)
    ap.add_argument("--k", type=int, default=100)
    args = ap.parse_args()

    result = run(args.n_items, args.dim, args.queries, args.k)
    os.makedirs(ART, exist_ok=True)
    with open(os.path.join(ART, "ann_bench.json"), "w") as fh:
        json.dump(result, fh, indent=2)
    table = to_markdown(result)
    with open(os.path.join(ART, "ann_bench.md"), "w") as fh:
        fh.write(table + "\n")
    print()
    print(table)


if __name__ == "__main__":
    main()
