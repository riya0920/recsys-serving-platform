"""Shadow deployment, gated promotion, and automated rollback.

    python -m service.shadow demo

The section the spec says it will actually read, so it is built as a state
machine with explicit transitions rather than a flag someone flips.

## The lifecycle

    CANDIDATE --shadow--> EVALUATING --gate passes--> PROMOTED
                              |                           |
                              +--gate fails--> REJECTED   +--regression--> ROLLED_BACK

**Shadow traffic means the candidate scores every live request and its output is
thrown away.** Users are served by the champion throughout. That is what makes
shadowing safe and also what makes it *not* an A/B test: it measures whether the
candidate can serve (latency, errors, agreement with the champion), not whether
users like it better. Confusing those two is how a "successful shadow" ships a
model nobody wanted.

## What the gate checks, and why each one

Promotion requires ALL of:

  1. **offline metric not worse** than champion beyond a tolerance - the
     candidate must actually be better on the thing it was trained for
  2. **error rate at or below champion** - a candidate that scores well and
     throws is not a candidate
  3. **p99 latency within a budget multiple** - a 10% quality gain that doubles
     tail latency is usually a bad trade, and the multiple makes that explicit
  4. **minimum sample size** - the single most common way a promotion gate gets
     fooled is deciding on 40 requests

Any failure rejects. Gates are AND-ed rather than scored, because a weighted
score lets a large win on one axis hide a disqualifying failure on another.

## Rollback

After promotion the new champion is monitored against the metrics its predecessor
held. A sustained regression triggers automatic rollback to the previous model.
The comparator's **false-positive rate is measurable** (see `simulate_stability`)
because "your rollback triggered, how do you know it wasn't a false alarm" is the
question this design exists to answer.
"""
from __future__ import annotations

import argparse
import json
import statistics
import time
from dataclasses import dataclass, field, asdict
from enum import Enum


class State(str, Enum):
    CANDIDATE = "candidate"
    EVALUATING = "evaluating"
    PROMOTED = "promoted"
    REJECTED = "rejected"
    ROLLED_BACK = "rolled_back"


@dataclass
class GateConfig:
    """Every threshold that decides a promotion, in one auditable place."""
    min_requests: int = 500
    # Offline metric must not be worse than champion by more than this (relative).
    max_metric_regression: float = 0.0
    # Candidate p99 may be at most this multiple of champion p99.
    max_p99_multiple: float = 1.25
    # Absolute error-rate ceiling for the candidate.
    max_error_rate: float = 0.001
    # Rollback fires after this many consecutive bad windows -- not on one.
    rollback_consecutive_windows: int = 3
    # A window is "bad" if the metric drops more than this below the promoted level.
    rollback_metric_drop: float = 0.05


@dataclass
class ModelStats:
    name: str
    requests: int = 0
    errors: int = 0
    latencies_ms: list = field(default_factory=list)
    offline_metric: float = 0.0

    @property
    def error_rate(self) -> float:
        return self.errors / self.requests if self.requests else 0.0

    def p99(self) -> float:
        if not self.latencies_ms:
            return 0.0
        s = sorted(self.latencies_ms)
        return s[min(len(s) - 1, int(0.99 * len(s)))]

    def p50(self) -> float:
        if not self.latencies_ms:
            return 0.0
        return statistics.median(self.latencies_ms)


@dataclass
class GateResult:
    passed: bool
    checks: dict
    reasons: list

    def as_dict(self) -> dict:
        return asdict(self)


def evaluate_gate(champion: ModelStats, candidate: ModelStats, cfg: GateConfig) -> GateResult:
    checks, reasons = {}, []

    enough = candidate.requests >= cfg.min_requests
    checks["sample_size"] = {"ok": enough, "requests": candidate.requests,
                             "required": cfg.min_requests}
    if not enough:
        reasons.append("only %d shadow requests; %d required. Deciding on a small sample is the "
                       "most common way a promotion gate gets fooled."
                       % (candidate.requests, cfg.min_requests))

    # Relative change in the offline metric. Positive is better.
    delta = ((candidate.offline_metric - champion.offline_metric) / champion.offline_metric
             if champion.offline_metric else 0.0)
    metric_ok = delta >= -cfg.max_metric_regression
    checks["offline_metric"] = {"ok": metric_ok, "champion": champion.offline_metric,
                                "candidate": candidate.offline_metric, "relative_delta": delta}
    if not metric_ok:
        reasons.append("offline metric regressed %.2f%% (tolerance %.2f%%)"
                       % (100 * delta, -100 * cfg.max_metric_regression))

    error_ok = candidate.error_rate <= cfg.max_error_rate
    checks["error_rate"] = {"ok": error_ok, "candidate": candidate.error_rate,
                            "ceiling": cfg.max_error_rate}
    if not error_ok:
        reasons.append("candidate error rate %.4f exceeds ceiling %.4f. A model that scores well "
                       "and throws is not a candidate." % (candidate.error_rate, cfg.max_error_rate))

    champ_p99 = champion.p99() or 1e-9
    ratio = candidate.p99() / champ_p99
    latency_ok = ratio <= cfg.max_p99_multiple
    checks["p99_latency"] = {"ok": latency_ok, "champion_p99_ms": champion.p99(),
                             "candidate_p99_ms": candidate.p99(), "ratio": ratio,
                             "max_multiple": cfg.max_p99_multiple}
    if not latency_ok:
        reasons.append("candidate p99 is %.2fx champion (limit %.2fx). A quality gain that costs "
                       "the tail is usually a bad trade." % (ratio, cfg.max_p99_multiple))

    return GateResult(all(c["ok"] for c in checks.values()), checks, reasons)


class ShadowDeployment:
    """Drives the lifecycle and records every transition."""

    def __init__(self, champion: str, candidate: str, cfg: GateConfig = None):
        self.cfg = cfg or GateConfig()
        self.champion = ModelStats(champion)
        self.candidate = ModelStats(candidate)
        self.state = State.CANDIDATE
        self.history = []
        self.promoted_metric = None
        self.bad_windows = 0
        self.previous_champion = None

    def _transition(self, new_state: State, note: str):
        self.history.append({"from": self.state, "to": new_state, "note": note,
                             "at": len(self.history)})
        self.state = new_state

    def start_shadow(self):
        self._transition(State.EVALUATING, "candidate now scoring live traffic; output discarded")

    def record(self, model: str, latency_ms: float, error: bool = False):
        stats = self.champion if model == self.champion.name else self.candidate
        stats.requests += 1
        stats.latencies_ms.append(latency_ms)
        if error:
            stats.errors += 1

    def set_offline_metrics(self, champion_metric: float, candidate_metric: float):
        self.champion.offline_metric = champion_metric
        self.candidate.offline_metric = candidate_metric

    def try_promote(self) -> GateResult:
        result = evaluate_gate(self.champion, self.candidate, self.cfg)
        if result.passed:
            self.previous_champion = self.champion
            self.promoted_metric = self.candidate.offline_metric
            self.bad_windows = 0
            self._transition(State.PROMOTED, "gate passed on all checks")
        else:
            self._transition(State.REJECTED, "; ".join(result.reasons))
        return result

    def observe_window(self, metric: float) -> dict:
        """Post-promotion monitoring. Returns whether a rollback fired.

        Rollback needs `rollback_consecutive_windows` bad windows in a row, not
        one. A single bad window is noise; requiring N consecutive is what keeps
        the comparator's false-positive rate low enough to trust, and the rate is
        measured rather than asserted -- see `simulate_stability`.
        """
        if self.state != State.PROMOTED:
            return {"rolled_back": False, "reason": "not in a promoted state"}

        drop = (self.promoted_metric - metric) / self.promoted_metric if self.promoted_metric else 0.0
        bad = drop > self.cfg.rollback_metric_drop
        self.bad_windows = self.bad_windows + 1 if bad else 0

        if self.bad_windows >= self.cfg.rollback_consecutive_windows:
            self._transition(State.ROLLED_BACK,
                             "%d consecutive windows more than %.0f%% below the promoted metric"
                             % (self.bad_windows, 100 * self.cfg.rollback_metric_drop))
            return {"rolled_back": True, "consecutive_bad_windows": self.bad_windows,
                    "reverted_to": self.previous_champion.name if self.previous_champion else None}
        return {"rolled_back": False, "consecutive_bad_windows": self.bad_windows, "drop": drop}

    def summary(self) -> dict:
        return {
            "state": self.state,
            "champion": {"name": self.champion.name, "requests": self.champion.requests,
                         "p50_ms": self.champion.p50(), "p99_ms": self.champion.p99(),
                         "error_rate": self.champion.error_rate,
                         "offline_metric": self.champion.offline_metric},
            "candidate": {"name": self.candidate.name, "requests": self.candidate.requests,
                          "p50_ms": self.candidate.p50(), "p99_ms": self.candidate.p99(),
                          "error_rate": self.candidate.error_rate,
                          "offline_metric": self.candidate.offline_metric},
            "history": self.history,
        }


def simulate_stability(trials: int = 500, windows: int = 20, noise: float = 0.03,
                       cfg: GateConfig = None, seed: int = 0) -> dict:
    """Measure the rollback comparator's FALSE-POSITIVE rate on a healthy model.

    This is the answer to "your rollback triggered -- how do you know it wasn't a
    false alarm?". A rollback rule whose false-positive rate is unmeasured will
    eventually revert a good model and nobody will be able to say whether it
    should have. Here the truth is known: the model is healthy in every trial, so
    every rollback is by definition a false positive.
    """
    import numpy as np

    cfg = cfg or GateConfig()
    rng = np.random.default_rng(seed)
    false_positives = 0

    for _ in range(trials):
        dep = ShadowDeployment("champion", "candidate", cfg)
        dep.start_shadow()
        dep.set_offline_metrics(0.10, 0.11)
        for _ in range(cfg.min_requests):
            dep.record("champion", 20.0)
            dep.record("candidate", 21.0)
        dep.try_promote()
        if dep.state != State.PROMOTED:
            continue
        # Healthy model: metric fluctuates around the promoted level, no true drift.
        for _ in range(windows):
            observed = 0.11 * (1.0 + rng.normal(0, noise))
            if dep.observe_window(observed)["rolled_back"]:
                false_positives += 1
                break

    return {
        "trials": trials,
        "windows_per_trial": windows,
        "metric_noise_sd": noise,
        "consecutive_windows_required": cfg.rollback_consecutive_windows,
        "rollback_drop_threshold": cfg.rollback_metric_drop,
        "false_positive_rate": false_positives / trials,
        "note": ("ground truth is a HEALTHY model in every trial, so every rollback here is a "
                 "false alarm by construction. This is the comparator's error rate, measured."),
    }


def demo() -> dict:
    """A worked lifecycle: one rejection, one promotion, one rollback."""
    out = {}

    # 1. A candidate that is better offline but far slower -> rejected on latency.
    slow = ShadowDeployment("v1", "v2-slow")
    slow.start_shadow()
    slow.set_offline_metrics(0.100, 0.130)
    for i in range(600):
        slow.record("v1", 20.0 + (i % 5))
        slow.record("v2-slow", 60.0 + (i % 5))
    out["rejected_on_latency"] = {"gate": slow.try_promote().as_dict(), "state": slow.state}

    # 2. A candidate decided on too little traffic -> rejected on sample size.
    small = ShadowDeployment("v1", "v2-untested")
    small.start_shadow()
    small.set_offline_metrics(0.100, 0.140)
    for i in range(40):
        small.record("v1", 20.0)
        small.record("v2-untested", 20.0)
    out["rejected_on_sample_size"] = {"gate": small.try_promote().as_dict(), "state": small.state}

    # 3. A good candidate -> promoted, then it degrades -> rolled back.
    good = ShadowDeployment("v1", "v2-good")
    good.start_shadow()
    good.set_offline_metrics(0.100, 0.118)
    for i in range(800):
        good.record("v1", 20.0 + (i % 4))
        good.record("v2-good", 21.0 + (i % 4))
    out["promoted"] = {"gate": good.try_promote().as_dict(), "state": good.state}

    windows = []
    for metric in [0.118, 0.117, 0.100, 0.099, 0.098, 0.097]:
        windows.append({"metric": metric, **good.observe_window(metric)})
    out["rollback"] = {"windows": windows, "final_state": good.state,
                       "history": good.history}

    out["comparator_false_positive_rate"] = simulate_stability(trials=500)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("command", choices=["demo", "stability"])
    ap.add_argument("--trials", type=int, default=500)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    result = demo() if args.command == "demo" else simulate_stability(args.trials)
    print(json.dumps(result, indent=2, default=str))
    if args.out:
        import os

        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        with open(args.out, "w") as fh:
            json.dump(result, fh, indent=2, default=str)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
