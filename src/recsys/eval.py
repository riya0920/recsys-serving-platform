"""Offline retrieval metrics. Same protocol for every model, including baselines.

Protocol (stated once, applied everywhere):
  * ground truth = the set of items a user interacted with in the eval window
  * candidates   = top-K from the retriever, with the user's TRAIN items removed
  * cold users/items are excluded and the exclusion is reported, never hidden

These are OFFLINE metrics. Nothing in this repo is an A/B result and the README
says so in the same words.
"""
from __future__ import annotations

from collections import defaultdict

import numpy as np
import pandas as pd


def build_ground_truth(eval_df: pd.DataFrame) -> dict:
    gt = defaultdict(set)
    for u, i in zip(eval_df["user_id"].to_numpy(), eval_df["item_id"].to_numpy()):
        gt[int(u)].add(int(i))
    return dict(gt)


def seen_items(train_df: pd.DataFrame) -> dict:
    seen = defaultdict(set)
    for u, i in zip(train_df["user_id"].to_numpy(), train_df["item_id"].to_numpy()):
        seen[int(u)].add(int(i))
    return dict(seen)


def recall_at_k(ranked, truth, k: int) -> float:
    if not truth:
        return float("nan")
    return len(set(ranked[:k]) & truth) / min(len(truth), k)


def ndcg_at_k(ranked, truth, k: int) -> float:
    if not truth:
        return float("nan")
    dcg = sum(1.0 / np.log2(rank + 2) for rank, item in enumerate(ranked[:k]) if item in truth)
    ideal = sum(1.0 / np.log2(r + 2) for r in range(min(len(truth), k)))
    return dcg / ideal if ideal else 0.0


def evaluate(rank_fn, users, ground_truth, ks=(10, 50, 100), max_k=None) -> dict:
    """rank_fn(user_id, k) -> list[item_id]. One function signature, every model."""
    max_k = max_k or max(ks)
    acc = {}
    for k in ks:
        acc["recall@%d" % k] = []
        acc["ndcg@%d" % k] = []
    for u in users:
        truth = ground_truth.get(u)
        if not truth:
            continue
        ranked = rank_fn(u, max_k)
        for k in ks:
            acc["recall@%d" % k].append(recall_at_k(ranked, truth, k))
            acc["ndcg@%d" % k].append(ndcg_at_k(ranked, truth, k))
    out = {m: (float(np.nanmean(v)) if v else float("nan")) for m, v in acc.items()}
    out["n_users_scored"] = float(len(acc["recall@%d" % ks[0]]))
    return out


class PopularityBaseline:
    """The bar every learned model must clear. Fitted on TRAIN only."""

    def __init__(self, train_df: pd.DataFrame, exclude_seen=None):
        counts = train_df["item_id"].value_counts()
        self.ranked_items = counts.index.to_numpy()
        self.exclude = exclude_seen or {}

    def rank(self, user_id: int, k: int):
        blocked = self.exclude.get(user_id, set())
        out = []
        for item in self.ranked_items:
            if int(item) in blocked:
                continue
            out.append(int(item))
            if len(out) >= k:
                break
        return out


def lift(candidate: dict, baseline: dict, metric: str) -> float:
    """Relative lift in percent. Both sides must come from the same protocol."""
    b = baseline.get(metric)
    if not b:
        return float("nan")
    return 100.0 * (candidate[metric] - b) / b
