"""How much recall does stale data cost? The measurement that justifies a TTL.

    python -m recsys.staleness --ages 0 1 3 7 14

Every recommender caches something - user embeddings, item popularity, the ANN
index itself - and every cache needs a TTL. That TTL is almost always chosen by
vibes ("an hour feels right"), which means nobody can say what it costs or what
would justify changing it.

This measures it directly. The model and index are frozen at day T, then
evaluated against traffic from T+0, T+1, T+3, T+7, T+14. The retrieval quality
curve against age IS the TTL argument: the point where the curve bends is where
the refresh has to happen.

**What is held stale and what is not**, because "staleness" is ambiguous and the
answer differs per component:

  * **item embeddings + ANN index** - frozen. This is the expensive artifact,
    rebuilt on a cadence, and the thing a TTL actually governs.
  * **user history used to exclude already-seen items** - kept CURRENT. In
    production this comes from an online store, not the batch job, and freezing
    it would conflate two very different failures. Conflating them is how a
    staleness study concludes "we must retrain hourly" when the real fix was a
    fresher exclusion list.

So the number below is specifically **the cost of a stale index**, which is the
component whose refresh is expensive enough to need justifying.
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np
import torch

from .config import Config
from .data import synthesize, time_split
from .eval import PopularityBaseline, build_ground_truth, evaluate, seen_items
from .index import VectorIndex
from .model import TwoTower, empirical_log_q, in_batch_softmax_loss

ART = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "artifacts"))
RESULTS = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "results"))

SECONDS_PER_DAY = 86400.0


def train_frozen_model(train_df, n_users: int, n_items: int, cfg: Config):
    """Train on data available up to the freeze point. Nothing after it is seen."""
    torch.manual_seed(cfg.data.seed)
    model = TwoTower(n_users, n_items, cfg.model)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.model.lr,
                            weight_decay=cfg.model.weight_decay)
    log_q = torch.tensor(empirical_log_q(train_df["item_id"].to_numpy(), n_items),
                         dtype=torch.float32)
    u_all = torch.tensor(train_df["user_id"].to_numpy(), dtype=torch.long)
    i_all = torch.tensor(train_df["item_id"].to_numpy(), dtype=torch.long)

    for _ in range(cfg.model.epochs):
        model.train()
        perm = torch.randperm(u_all.numel())
        for s in range(0, perm.numel(), cfg.model.batch_size):
            idx = perm[s: s + cfg.model.batch_size]
            if idx.numel() < 8:
                continue
            u, i = u_all[idx], i_all[idx]
            loss = in_batch_softmax_loss(model.user_vec(u), model.item_vec(i),
                                         log_q[i] if cfg.model.logq_correction else None,
                                         cfg.model.temperature)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
    model.eval()
    return model


def run(ages_days, cfg: Config, eval_window_days: float = 1.0, max_users: int = 1200) -> dict:
    df = synthesize(cfg.data)
    n_users = int(df["user_id"].max()) + 1
    n_items = int(df["item_id"].max()) + 1

    # Freeze point: train on everything up to here, then look forward.
    t_freeze = df["ts"].quantile(cfg.data.train_end_q)
    train_df = df[df["ts"] <= t_freeze]

    model = train_frozen_model(train_df, n_users, n_items, cfg)
    item_vecs = model.all_item_vectors()
    index = VectorIndex(cfg.index, cfg.model.dim).build(item_vecs)

    # The exclusion list stays CURRENT -- see the module docstring for why.
    seen = seen_items(train_df)
    known_users = set(int(u) for u in train_df["user_id"].unique())

    @torch.no_grad()
    def model_rank(user_id: int, k: int):
        uv = model.user_vec(torch.tensor([int(user_id)])).numpy().astype("float32")
        blocked = seen.get(int(user_id), set())
        _, ids = index.search(uv, k + len(blocked) + 1)
        return [int(x) for x in ids[0] if x >= 0 and int(x) not in blocked][:k]

    rows = []
    for age in ages_days:
        lo = t_freeze + age * SECONDS_PER_DAY
        hi = lo + eval_window_days * SECONDS_PER_DAY
        window = df[(df["ts"] > lo) & (df["ts"] <= hi)]
        if window.empty:
            rows.append({"age_days": age, "skipped": True,
                         "reason": "no traffic in this window; the generated span is too short"})
            continue

        gt = build_ground_truth(window)
        users = [u for u in gt if u in known_users][:max_users]
        if not users:
            rows.append({"age_days": age, "skipped": True, "reason": "no evaluable users"})
            continue

        model_metrics = evaluate(model_rank, users, gt)
        baseline = PopularityBaseline(train_df, exclude_seen=seen)
        base_metrics = evaluate(baseline.rank, users, gt)

        rows.append({
            "age_days": age,
            "eval_rows": int(len(window)),
            "users_scored": int(model_metrics["n_users_scored"]),
            "recall@10": model_metrics["recall@10"],
            "ndcg@10": model_metrics["ndcg@10"],
            "recall@100": model_metrics["recall@100"],
            "baseline_recall@10": base_metrics["recall@10"],
        })

    measured = [r for r in rows if not r.get("skipped")]
    fresh = measured[0] if measured else None
    for r in measured:
        if fresh and fresh["recall@10"]:
            r["recall@10_vs_fresh_pct"] = 100.0 * (r["recall@10"] - fresh["recall@10"]) / fresh["recall@10"]

    return {
        "taste_drift_per_day": cfg.data.taste_drift_per_day,
        "freeze_quantile": cfg.data.train_end_q,
        "eval_window_days": eval_window_days,
        "frozen": ["item_embeddings", "ann_index", "user_tower_weights"],
        "kept_current": ["already-seen exclusion list"],
        "rows": rows,
        "note": ("this isolates the cost of a STALE INDEX. The seen-item exclusion list is kept "
                 "current because in production it comes from an online store, and freezing both "
                 "would conflate two failures with different fixes."),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ages", type=float, nargs="+", default=[0, 1, 3, 7, 14])
    ap.add_argument("--n-items", type=int, default=8000)
    ap.add_argument("--n-events", type=int, default=250_000)
    ap.add_argument("--days", type=int, default=180)
    ap.add_argument("--drift", type=float, default=0.0,
                    help="per-day probability a user's taste resamples; 0 = stationary")
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--eval-window-days", type=float, default=1.0)
    ap.add_argument("--out", default=os.path.join(RESULTS, "staleness.json"))
    args = ap.parse_args()

    cfg = Config()
    cfg.data.n_items = args.n_items
    cfg.data.n_events = args.n_events
    cfg.model.epochs = args.epochs
    cfg.data.days = args.days
    cfg.data.taste_drift_per_day = args.drift

    result = run(args.ages, cfg, args.eval_window_days)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as fh:
        json.dump(result, fh, indent=2)

    print("| index age (days) | recall@10 | ndcg@10 | recall@100 | vs fresh | users |")
    print("|---|---|---|---|---|---|")
    for r in result["rows"]:
        if r.get("skipped"):
            print("| %s | _skipped_ | | | | %s |" % (r["age_days"], r["reason"]))
            continue
        print("| %g | %.4f | %.4f | %.4f | %s | %d |"
              % (r["age_days"], r["recall@10"], r["ndcg@10"], r["recall@100"],
                 ("%+.1f%%" % r["recall@10_vs_fresh_pct"]) if "recall@10_vs_fresh_pct" in r else "-",
                 r["users_scored"]))
    print("\nwrote", args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
