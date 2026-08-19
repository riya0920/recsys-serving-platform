"""Train the stage-2 ranker and measure what it buys, with MLflow lineage.

    python -m recsys.train_ranker --candidates 200

Every run is tracked: params, metrics and the artifact, so a comparison between
two configurations is a query rather than a memory.
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np
import torch

from .config import Config
from .data import synthesize, time_split
from .eval import seen_items
from .index import VectorIndex
from .model import TwoTower
from .ranker import FEATURE_NAMES, FeatureBuilder, GBDTRanker, build_training_data, evaluate_two_stage

ART = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "artifacts"))


def _load_retriever(cfg: Config):
    ckpt = torch.load(os.path.join(ART, "model.pt"), map_location="cpu", weights_only=False)
    model = TwoTower(ckpt["n_users"], ckpt["n_items"], cfg.model)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    index = VectorIndex.load(os.path.join(ART, "items.faiss"), cfg.index, cfg.model.dim)
    return model, index, ckpt["n_items"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--candidates", type=int, default=200)
    ap.add_argument("--negatives-per-pos", type=int, default=4)
    ap.add_argument("--max-iter", type=int, default=120)
    ap.add_argument("--learning-rate", type=float, default=0.08)
    ap.add_argument("--max-depth", type=int, default=6)
    ap.add_argument("--train-users", type=int, default=1500)
    ap.add_argument("--eval-users", type=int, default=800)
    ap.add_argument("--n-items", type=int, default=8000)
    ap.add_argument("--n-events", type=int, default=200_000)
    ap.add_argument("--no-mlflow", action="store_true")
    args = ap.parse_args()

    cfg = Config()
    cfg.data.n_items = args.n_items
    cfg.data.n_events = args.n_events

    df = synthesize(cfg.data)
    train_df, valid_df, test_df = time_split(df, cfg.data)
    model, index, n_items = _load_retriever(cfg)
    seen = seen_items(train_df)

    @torch.no_grad()
    def retrieve(user_id: int, k: int):
        uv = model.user_vec(torch.tensor([int(user_id)])).numpy().astype("float32")
        blocked = seen.get(int(user_id), set())
        scores, ids = index.search(uv, k + len(blocked) + 1)
        out_i, out_s = [], []
        for item, score in zip(ids[0], scores[0]):
            if item < 0 or int(item) in blocked:
                continue
            out_i.append(int(item))
            out_s.append(float(score))
            if len(out_i) >= k:
                break
        return np.array(out_i), np.array(out_s)

    fb = FeatureBuilder(train_df, n_items)

    # Train on VALIDATION interactions, evaluate on TEST. Training the ranker on
    # the same window it is scored on would leak, and the leak is invisible in
    # the metric -- it just looks like a very good ranker.
    X, y = build_training_data(retrieve, fb, train_df, valid_df,
                               n_users=args.train_users, candidates_k=args.candidates,
                               negatives_per_pos=args.negatives_per_pos)
    print("training rows: %d  positives: %d (%.1f%%)" % (len(y), int(y.sum()), 100 * y.mean() if len(y) else 0))
    if len(y) < 100:
        raise SystemExit("not enough training rows; increase --n-events or --train-users")

    ranker = GBDTRanker(max_iter=args.max_iter, learning_rate=args.learning_rate,
                        max_depth=args.max_depth).fit(X, y)

    truth_users = [int(u) for u in test_df["user_id"].unique()][: args.eval_users]
    metrics = evaluate_two_stage(retrieve, ranker, fb, test_df, truth_users,
                                 candidates_k=args.candidates)
    metrics.n_train_rows = int(len(y))

    os.makedirs(ART, exist_ok=True)
    ranker.save(os.path.join(ART, "ranker.pkl"))
    report = {"params": vars(args), "features": FEATURE_NAMES, "metrics": metrics.as_dict()}
    with open(os.path.join(ART, "ranker_report.json"), "w") as fh:
        json.dump(report, fh, indent=2)

    if not args.no_mlflow:
        try:
            import mlflow

            # SQLite rather than the file store: the file backend is deprecated
            # and its run directories break when the artifacts dir is recreated,
            # which is exactly what the training script does on every run.
            mlflow.set_tracking_uri("sqlite:///" + os.path.join(ART, "mlflow.db").replace("\\", "/"))
            mlflow.set_experiment("recsys-ranker")
            with mlflow.start_run():
                mlflow.log_params({k: v for k, v in vars(args).items() if k != "no_mlflow"})
                mlflow.log_metrics(metrics.as_dict())
                mlflow.log_artifact(os.path.join(ART, "ranker_report.json"))
        except Exception as exc:
            print("mlflow logging skipped: %s" % exc)

    print(json.dumps(metrics.as_dict(), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
