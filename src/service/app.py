"""Serving tier: two-stage recommend with a killable model path.

The design point of this file: /recommend must return 200 with useful items even
when the model path is dead. The model is a *dependency*, not the service. You
can prove it in a live demo -- POST /admin/kill-model while a load test runs and
watch the error rate stay at zero and `degraded: true` appear in the payload.
"""
from __future__ import annotations

import os
import time
from collections import deque
from threading import Lock

import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

ART = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "artifacts"))


class LatencyRecorder:
    """Fixed-window latency reservoir. Cheap enough to sit on the hot path."""

    def __init__(self, window: int = 20_000):
        self._buf = deque(maxlen=window)
        self._lock = Lock()

    def record(self, ms: float):
        with self._lock:
            self._buf.append(ms)

    def snapshot(self) -> dict:
        with self._lock:
            data = np.array(self._buf) if self._buf else np.array([0.0])
        return {
            "count": int(len(self._buf)),
            "p50_ms": float(np.percentile(data, 50)),
            "p95_ms": float(np.percentile(data, 95)),
            "p99_ms": float(np.percentile(data, 99)),
        }


class Recommender:
    """Holds the model path. Every method here is allowed to fail."""

    def __init__(self):
        self.enabled = True
        self.model = None
        self.index = None
        self.popular: list[int] = []
        self.loaded = False

    def load(self, art_dir: str = ART):
        """Best-effort load. A missing artifact degrades the service, not kills it."""
        import json

        try:
            with open(os.path.join(art_dir, "train_report.json")) as fh:
                report = json.load(fh)
            self.popular = report.get("popular_items", [])
        except Exception:
            self.popular = []

        try:
            import torch

            from recsys.config import Config
            from recsys.index import VectorIndex
            from recsys.model import TwoTower

            cfg = Config()
            ckpt = torch.load(os.path.join(art_dir, "model.pt"), map_location="cpu", weights_only=False)
            model = TwoTower(ckpt["n_users"], ckpt["n_items"], cfg.model)
            model.load_state_dict(ckpt["state_dict"])
            model.eval()
            self.model = model
            self.index = VectorIndex.load(os.path.join(art_dir, "items.faiss"), cfg.index, cfg.model.dim)
            if not self.popular:
                self.popular = list(range(min(1000, ckpt["n_items"])))
            self.loaded = True
        except Exception:
            self.loaded = False

    def healthy(self) -> bool:
        return self.enabled and self.loaded

    def retrieve(self, user_id: int, k: int) -> list[int]:
        import torch

        with torch.no_grad():
            uv = self.model.user_vec(torch.tensor([user_id])).numpy()
        _, ids = self.index.search(uv.astype("float32"), k)
        return [int(x) for x in ids[0] if x >= 0]


REC = Recommender()
LAT = LatencyRecorder()
COUNTERS = {"requests": 0, "degraded": 0, "errors": 0}

app = FastAPI(title="recsys-serving", version="0.4.0")


class RecommendResponse(BaseModel):
    user_id: int
    items: list[int]
    source: str = Field(description="two_stage | popularity_fallback")
    degraded: bool
    latency_ms: float


@app.on_event("startup")
def _startup():
    REC.load()


@app.get("/healthz")
def healthz():
    """Liveness. Always 200 while the process is up -- a degraded service is still up."""
    return {"status": "ok", "model_loaded": REC.loaded, "model_enabled": REC.enabled}


@app.get("/readyz")
def readyz():
    """Readiness. The fallback path is enough to serve traffic, so this is 200
    even without a model. Gating readiness on the model would take the whole
    service out during a bad model rollout -- exactly backwards."""
    return {"ready": True, "degraded": not REC.healthy()}


@app.get("/metrics")
def metrics():
    out = {"latency": LAT.snapshot(), "counters": dict(COUNTERS)}
    out["degraded_rate"] = COUNTERS["degraded"] / max(COUNTERS["requests"], 1)
    return out


@app.post("/admin/kill-model")
def kill_model():
    """Chaos hook. Used by the scripted degradation drill; not exposed publicly."""
    REC.enabled = False
    return {"model_enabled": REC.enabled}


@app.post("/admin/revive-model")
def revive_model():
    REC.enabled = True
    return {"model_enabled": REC.enabled}


@app.get("/recommend", response_model=RecommendResponse)
def recommend(user_id: int, k: int = 10):
    if k < 1 or k > 200:
        raise HTTPException(status_code=400, detail="k must be in [1, 200]")
    t0 = time.perf_counter()
    source, degraded = "two_stage", False
    try:
        if not REC.healthy():
            raise RuntimeError("model path unavailable")
        items = REC.retrieve(user_id, k)
        if not items:
            raise RuntimeError("empty candidate set")
    except Exception:
        # Every failure in the model path lands here. The user gets popular items
        # and a 200; the fact that we degraded is a metric, not an error.
        items = REC.popular[:k]
        source, degraded = "popularity_fallback", True
        COUNTERS["degraded"] += 1

    ms = (time.perf_counter() - t0) * 1000.0
    LAT.record(ms)
    COUNTERS["requests"] += 1
    return RecommendResponse(user_id=user_id, items=items, source=source, degraded=degraded, latency_ms=ms)
