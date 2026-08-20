"""Concurrency sweep: find max sustained RPS at p99 < 100 ms.

    python -m service.loadtest --url http://localhost:8000 --sweep 8 16 32 64 128

The spec asks for "max RPS at p99 < 100ms ON NAMED HARDWARE". A single
concurrency level cannot answer that — it gives one point on a curve. The sweep
walks concurrency upward and reports the last level that still met the budget,
which is what "max RPS at p99 < X" actually means.

Traffic is **zipf over user ids**, not uniform: a real recommender serves a small
set of active users far more often than the long tail, and uniform sampling
defeats every cache in the system by construction.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import platform
import time

import numpy as np

try:
    import aiohttp
except ImportError:  # pragma: no cover
    aiohttp = None


def zipf_users(n_users: int, n: int, a: float = 1.2, seed: int = 0):
    rng = np.random.default_rng(seed)
    ranks = np.arange(1, n_users + 1)
    w = 1.0 / np.power(ranks, a)
    w /= w.sum()
    return rng.choice(n_users, size=n, p=w)


async def _worker(session, base, users, stop_at, out, k):
    i = 0
    while time.perf_counter() < stop_at:
        user = int(users[i % len(users)])
        i += 1
        t0 = time.perf_counter()
        try:
            async with session.get(f"{base}/recommend", params={"user_id": user, "k": k}) as r:
                await r.read()
                out.append(((time.perf_counter() - t0) * 1000.0, r.status))
        except Exception:
            out.append((0.0, -1))


async def _wait_ready(session, base, timeout_s: float = 120.0) -> float:
    """The service imports torch and reads a FAISS index at startup; firing before
    it is up produces connection refusals that look like failures but are not."""
    t0 = time.perf_counter()
    deadline = t0 + timeout_s
    while time.perf_counter() < deadline:
        try:
            async with session.get(f"{base}/healthz") as r:
                if r.status == 200:
                    await r.read()
                    return time.perf_counter() - t0
        except Exception:
            pass
        await asyncio.sleep(0.25)
    raise SystemExit("service never became ready")


async def run_level(session, base, concurrency, duration, warmup, users, k):
    samples = []
    t_start = time.perf_counter()
    stop_at = t_start + warmup + duration
    tasks = [asyncio.create_task(_worker(session, base, users, stop_at, samples, k))
             for _ in range(concurrency)]
    await asyncio.sleep(warmup)
    cutoff = time.perf_counter()
    marker = len(samples)
    await asyncio.gather(*tasks)

    measured = samples[marker:]
    if not measured:
        return None
    lat = np.array([s[0] for s in measured])
    errors = sum(1 for s in measured if s[1] < 0 or s[1] >= 500)
    return {
        "concurrency": concurrency,
        "requests": len(measured),
        "rps": len(measured) / duration,
        "p50_ms": float(np.percentile(lat, 50)),
        "p95_ms": float(np.percentile(lat, 95)),
        "p99_ms": float(np.percentile(lat, 99)),
        "errors": errors,
    }


async def sweep(base, levels, duration, warmup, n_users, k, budget_ms):
    if aiohttp is None:
        raise SystemExit("pip install aiohttp")
    users = zipf_users(n_users, 20000)
    rows = []
    async with aiohttp.ClientSession(
        connector=aiohttp.TCPConnector(limit=max(levels) + 16),
        timeout=aiohttp.ClientTimeout(total=30),
    ) as session:
        startup = await _wait_ready(session, base)
        for c in levels:
            row = await run_level(session, base, c, duration, warmup, users, k)
            if row:
                row["meets_budget"] = row["p99_ms"] < budget_ms and row["errors"] == 0
                rows.append(row)
                print("  c=%-4d rps=%7.1f p50=%6.1f p99=%7.1f errors=%d %s"
                      % (c, row["rps"], row["p50_ms"], row["p99_ms"], row["errors"],
                         "OK" if row["meets_budget"] else "over budget"))

    passing = [r for r in rows if r["meets_budget"]]
    best = max(passing, key=lambda r: r["rps"]) if passing else None
    return {
        "hardware": {"platform": platform.platform(),
                     "processor": platform.processor() or platform.machine(),
                     "cpu_count": os.cpu_count()},
        "budget_p99_ms": budget_ms,
        "startup_wait_s": round(startup, 1),
        "levels": rows,
        "max_rps_within_budget": best,
        "caveat": ("closed-loop client sharing the machine with the server, so the tail is "
                   "understated under saturation and the throughput is a floor."),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://localhost:8000")
    ap.add_argument("--sweep", type=int, nargs="+", default=[8, 16, 32, 64])
    ap.add_argument("--duration", type=float, default=10.0)
    ap.add_argument("--warmup", type=float, default=3.0)
    ap.add_argument("--users", type=int, default=5000)
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--budget-ms", type=float, default=100.0)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    result = asyncio.run(sweep(args.url, args.sweep, args.duration, args.warmup,
                               args.users, args.k, args.budget_ms))
    print()
    print(json.dumps({k: v for k, v in result.items() if k != "levels"}, indent=2))
    if args.out:
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        with open(args.out, "w") as fh:
            json.dump(result, fh, indent=2)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
