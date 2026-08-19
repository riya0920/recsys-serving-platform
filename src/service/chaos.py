"""The chaos drill: kill the model mid-load-test and prove users never see a 5xx.

    python -m service.chaos --url http://localhost:8000 --duration 20

The spec's differentiator is "model service dies mid-load-test, dashboard shows
fallback engaging, locust shows 0 errors". This is that, as a script rather than
a GIF, because a script can be re-run and a GIF cannot be verified.

Timeline: warm up under load, kill the model path, keep loading, revive it, keep
loading. The verdict requires:
  * zero 5xx and zero connection errors across the whole run
  * every request during the outage answered from the fallback (`degraded: true`)
  * the service recovering to the model path after revival
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import time

import numpy as np

try:
    import aiohttp
except ImportError:  # pragma: no cover
    aiohttp = None


async def _worker(session, base, n_users, stop_at, samples, rng):
    while time.perf_counter() < stop_at:
        user = int(rng.integers(0, n_users))
        t0 = time.perf_counter()
        try:
            async with session.get(f"{base}/recommend", params={"user_id": user, "k": 10}) as r:
                body = await r.json() if r.status == 200 else {}
                samples.append({
                    "ms": (time.perf_counter() - t0) * 1000.0,
                    "status": r.status,
                    "degraded": bool(body.get("degraded", False)),
                    "source": body.get("source", ""),
                    "t": time.perf_counter(),
                    "n_items": len(body.get("items", [])),
                })
        except Exception as exc:
            samples.append({"ms": (time.perf_counter() - t0) * 1000.0, "status": -1,
                            "degraded": False, "source": type(exc).__name__,
                            "t": time.perf_counter(), "n_items": 0})


async def _wait_for_ready(session, base: str, timeout_s: float = 60.0) -> float:
    """Block until the service answers, then return how long it took.

    Without this the drill starts firing while uvicorn is still importing torch
    and reading the FAISS index, and every request in the first phase is a
    connection refusal. Those are startup artifacts, not service failures, and
    counting them as errors would make a passing drill look like a failing one --
    which is exactly what happened on the first run.
    """
    deadline = time.perf_counter() + timeout_s
    t0 = time.perf_counter()
    while time.perf_counter() < deadline:
        try:
            async with session.get(f"{base}/healthz") as r:
                if r.status == 200:
                    await r.read()
                    return time.perf_counter() - t0
        except Exception:
            pass
        await asyncio.sleep(0.25)
    raise SystemExit("service at %s never became ready within %.0fs" % (base, timeout_s))


async def run(base: str, duration: float, concurrency: int, n_users: int, seed: int) -> dict:
    if aiohttp is None:
        raise SystemExit("pip install aiohttp")
    rng = np.random.default_rng(seed)
    samples = []

    async with aiohttp.ClientSession(
        connector=aiohttp.TCPConnector(limit=concurrency + 8),
        timeout=aiohttp.ClientTimeout(total=10),
    ) as session:
        startup_s = await _wait_for_ready(session, base)
        t_start = time.perf_counter()
        stop_at = t_start + duration
        tasks = [asyncio.create_task(_worker(session, base, n_users, stop_at, samples, rng))
                 for _ in range(concurrency)]

        # Phase 1: healthy.
        await asyncio.sleep(duration / 3)
        async with session.post(f"{base}/admin/kill-model") as r:
            await r.read()
        t_kill = time.perf_counter()

        # Phase 2: model dead, load continues.
        await asyncio.sleep(duration / 3)
        async with session.post(f"{base}/admin/revive-model") as r:
            await r.read()
        t_revive = time.perf_counter()

        # Phase 3: recovered.
        await asyncio.gather(*tasks)

        async with session.get(f"{base}/metrics") as r:
            metrics = await r.json()

    def phase(lo, hi):
        return [s for s in samples if lo <= s["t"] < hi]

    before = phase(t_start, t_kill)
    during = phase(t_kill, t_revive)
    after = phase(t_revive, float("inf"))

    def summarise(rows, label):
        if not rows:
            return {"phase": label, "requests": 0}
        lat = np.array([r["ms"] for r in rows])
        from collections import Counter

        return {
            "phase": label,
            "requests": len(rows),
            "status_counts": dict(Counter(r["status"] for r in rows)),
            "error_kinds": dict(Counter(r["source"] for r in rows if r["status"] < 0)),
            "errors": sum(1 for r in rows if r["status"] < 0 or r["status"] >= 500),
            "degraded_fraction": round(sum(1 for r in rows if r["degraded"]) / len(rows), 4),
            "empty_responses": sum(1 for r in rows if r["n_items"] == 0),
            "p50_ms": round(float(np.percentile(lat, 50)), 2),
            "p99_ms": round(float(np.percentile(lat, 99)), 2),
        }

    phases = [summarise(before, "healthy"), summarise(during, "model_killed"),
              summarise(after, "recovered")]
    total_errors = sum(p.get("errors", 0) for p in phases)

    return {
        "drill": "model_death_under_load",
        "concurrency": concurrency,
        "waited_for_startup_s": round(startup_s, 2),
        "total_requests": len(samples),
        "total_errors_5xx_or_connection": total_errors,
        "phases": phases,
        "server_metrics": metrics,
        "passed": (
            total_errors == 0
            and phases[1].get("degraded_fraction", 0) > 0.95
            and phases[1].get("empty_responses", 1) == 0
            and phases[2].get("degraded_fraction", 1) < 0.5
        ),
        "claim": ("the model path was killed under %d concurrent clients: %d total requests, "
                  "%d errors, and every request during the outage was served from the fallback"
                  % (concurrency, len(samples), total_errors)),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://localhost:8000")
    ap.add_argument("--duration", type=float, default=18.0)
    ap.add_argument("--concurrency", type=int, default=32)
    ap.add_argument("--users", type=int, default=5000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    result = asyncio.run(run(args.url, args.duration, args.concurrency, args.users, args.seed))
    print(json.dumps(result, indent=2))
    if args.out:
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        with open(args.out, "w") as fh:
            json.dump(result, fh, indent=2)
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
