"""Train the retrieval tower. One command, deterministic, writes artifacts/."""
from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np
import torch

from .config import Config
from .data import cold_start_report, load_movielens, synthesize, time_split
from .eval import PopularityBaseline, build_ground_truth, evaluate, lift, seen_items
from .index import VectorIndex
from .model import TwoTower, empirical_log_q, in_batch_softmax_loss

ART = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "artifacts"))


def train(cfg: Config, ratings_csv=None, out_dir: str = ART) -> dict:
    os.makedirs(out_dir, exist_ok=True)
    torch.manual_seed(cfg.data.seed)
    np.random.seed(cfg.data.seed)

    df = load_movielens(ratings_csv) if ratings_csv else synthesize(cfg.data)
    n_users = int(df["user_id"].max()) + 1
    n_items = int(df["item_id"].max()) + 1
    train_df, valid_df, test_df = time_split(df, cfg.data)
    cold = cold_start_report(train_df, test_df)

    model = TwoTower(n_users, n_items, cfg.model)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.model.lr, weight_decay=cfg.model.weight_decay)

    log_q_all = torch.tensor(
        empirical_log_q(train_df["item_id"].to_numpy(), n_items), dtype=torch.float32
    )
    u_all = torch.tensor(train_df["user_id"].to_numpy(), dtype=torch.long)
    i_all = torch.tensor(train_df["item_id"].to_numpy(), dtype=torch.long)

    history = []
    for epoch in range(cfg.model.epochs):
        model.train()
        perm = torch.randperm(u_all.numel())
        total, nb = 0.0, 0
        t0 = time.perf_counter()
        for s in range(0, perm.numel(), cfg.model.batch_size):
            idx = perm[s : s + cfg.model.batch_size]
            if idx.numel() < 8:
                continue
            u, i = u_all[idx], i_all[idx]
            loss = in_batch_softmax_loss(
                model.user_vec(u),
                model.item_vec(i),
                log_q_all[i] if cfg.model.logq_correction else None,
                cfg.model.temperature,
            )
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            total += float(loss)
            nb += 1
        rec = {"epoch": epoch, "loss": total / max(nb, 1), "sec": time.perf_counter() - t0}
        history.append(rec)
        print("epoch %d  loss=%.4f  %.1fs" % (rec["epoch"], rec["loss"], rec["sec"]))

    item_vecs = model.all_item_vectors()
    np.save(os.path.join(out_dir, "item_vectors.npy"), item_vecs)
    torch.save(
        {"state_dict": model.state_dict(), "n_users": n_users, "n_items": n_items},
        os.path.join(out_dir, "model.pt"),
    )

    index = VectorIndex(cfg.index, cfg.model.dim).build(item_vecs)
    index.save(os.path.join(out_dir, "items.faiss"))

    metrics = offline_eval(model, index, train_df, test_df)
    report = {
        "config": json.loads(cfg.to_json()),
        "cold_start": cold,
        "history": history,
        "metrics": metrics,
        "note": "OFFLINE evaluation only. No online A/B test was run.",
    }
    with open(os.path.join(out_dir, "train_report.json"), "w") as fh:
        json.dump(report, fh, indent=2)
    return report


def offline_eval(model, index, train_df, test_df, max_users: int = 2000) -> dict:
    seen = seen_items(train_df)
    gt = build_ground_truth(test_df)
    known_users = set(int(u) for u in train_df["user_id"].unique())
    users = [u for u in gt if u in known_users][:max_users]

    baseline = PopularityBaseline(train_df, exclude_seen=seen)
    base = evaluate(baseline.rank, users, gt)

    @torch.no_grad()
    def model_rank(user_id: int, k: int):
        uv = model.user_vec(torch.tensor([user_id])).numpy()
        blocked = seen.get(user_id, set())
        _, ids = index.search(uv, k + len(blocked) + 1)
        return [int(x) for x in ids[0] if x >= 0 and int(x) not in blocked][:k]

    mdl = evaluate(model_rank, users, gt)
    return {
        "popularity_baseline": base,
        "two_tower": mdl,
        "lift_pct": {m: lift(mdl, base, m) for m in ("recall@10", "ndcg@10", "recall@100")},
    }


def main():
    ap = argparse.ArgumentParser(description="Train the two-tower retrieval model.")
    ap.add_argument("--ratings-csv", default=None, help="path to a real MovieLens ratings.csv")
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--n-items", type=int, default=None)
    ap.add_argument("--n-events", type=int, default=None)
    ap.add_argument("--no-logq", action="store_true", help="ablation: drop the sampling-bias correction")
    args = ap.parse_args()

    cfg = Config()
    if args.epochs:
        cfg.model.epochs = args.epochs
    if args.n_items:
        cfg.data.n_items = args.n_items
    if args.n_events:
        cfg.data.n_events = args.n_events
    if args.no_logq:
        cfg.model.logq_correction = False

    report = train(cfg, args.ratings_csv)
    print(json.dumps(report["metrics"], indent=2))


if __name__ == "__main__":
    main()
