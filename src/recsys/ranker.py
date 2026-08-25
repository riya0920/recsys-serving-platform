"""Stage 2: a learned ranker over the retrieved candidates.

Retrieval optimises recall over 50K items with one dot product. Ranking optimises
precision over ~500 candidates and can afford features that would be impossible
at corpus scale - that division of labour is the entire reason for two stages.

**Training data is generated the way serving works.** Negatives are sampled from
the *retriever's own top-K*, not uniformly from the catalogue. This matters more
than the model choice: a ranker trained against uniform random negatives learns
to separate "plausible" from "absurd", which retrieval already did. At serving
time it only ever sees plausible items, so its training distribution and its
serving distribution disagree and the measured offline lift evaporates. Sampling
hard negatives from the retriever closes that gap.

**Features are deliberately cheap.** Everything here is a lookup or arithmetic on
values already in hand at request time - no joins, no second model, no feature
store round trip. A ranker whose features cannot be computed inside the latency
budget is a research artifact.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass

import numpy as np


FEATURE_NAMES = [
    "retrieval_score",       # cosine from the two-tower; the retriever's own opinion
    "item_log_popularity",   # global prior, log-scaled to stop head items dominating
    "item_rank_in_slate",    # position within the candidate list
    "user_activity",         # log interactions in train; heavy users behave differently
    "user_item_topic_match", # cheap affinity proxy
    "item_recency",          # normalised last-seen time; freshness matters in feeds
]


@dataclass
class RankerMetrics:
    ndcg_at_10_retrieval: float
    ndcg_at_10_ranked: float
    recall_at_10_retrieval: float
    recall_at_10_ranked: float
    lift_ndcg_pct: float
    n_users: int
    n_train_rows: int

    def as_dict(self) -> dict:
        return self.__dict__.copy()


class FeatureBuilder:
    """Builds ranking features from artifacts already loaded for serving."""

    def __init__(self, train_df, n_items: int, item_topics=None):
        counts = np.bincount(train_df["item_id"].to_numpy(), minlength=n_items).astype("float64")
        self.item_log_pop = np.log1p(counts)
        self.item_log_pop /= max(self.item_log_pop.max(), 1e-9)

        user_counts = train_df.groupby("user_id").size()
        self.user_activity = {int(u): float(np.log1p(c)) for u, c in user_counts.items()}

        last_ts = train_df.groupby("item_id")["ts"].max()
        ts_min, ts_max = train_df["ts"].min(), train_df["ts"].max()
        span = max(ts_max - ts_min, 1e-9)
        self.item_recency = np.zeros(n_items)
        for item, ts in last_ts.items():
            self.item_recency[int(item)] = (ts - ts_min) / span

        # Cheap topic affinity: the user's historical item-popularity profile.
        self.item_topics = item_topics
        self.user_topic_pref = {}
        if item_topics is not None:
            for u, grp in train_df.groupby("user_id"):
                topics = item_topics[grp["item_id"].to_numpy()]
                pref = np.bincount(topics, minlength=int(item_topics.max()) + 1).astype("float64")
                total = pref.sum()
                self.user_topic_pref[int(u)] = pref / total if total else pref

    def build(self, user_id: int, candidates, scores) -> np.ndarray:
        n = len(candidates)
        feats = np.zeros((n, len(FEATURE_NAMES)), dtype="float32")
        activity = self.user_activity.get(int(user_id), 0.0)
        pref = self.user_topic_pref.get(int(user_id))

        for i, (item, score) in enumerate(zip(candidates, scores)):
            item = int(item)
            feats[i, 0] = score
            feats[i, 1] = self.item_log_pop[item]
            # Rank is normalised so the feature means the same thing regardless
            # of how many candidates retrieval returned.
            feats[i, 2] = i / max(n - 1, 1)
            feats[i, 3] = activity
            feats[i, 4] = (pref[self.item_topics[item]] if pref is not None and self.item_topics is not None else 0.0)
            feats[i, 5] = self.item_recency[item]
        return feats


class GBDTRanker:
    """Gradient-boosted trees over the candidate features.

    GBDT rather than a neural ranker, deliberately: with six tabular features and
    a few hundred thousand rows, trees win on quality per unit of effort and are
    far cheaper to serve. A neural ranker earns its place when features become
    high-cardinality or sequential, and that is a documented trigger rather than
    a default.
    """

    def __init__(self, max_iter: int = 120, learning_rate: float = 0.08, max_depth: int = 6,
                 seed: int = 0):
        from sklearn.ensemble import HistGradientBoostingClassifier

        self.model = HistGradientBoostingClassifier(
            max_iter=max_iter, learning_rate=learning_rate, max_depth=max_depth,
            random_state=seed, early_stopping=True, validation_fraction=0.1,
        )
        self.fitted = False

    def fit(self, X: np.ndarray, y: np.ndarray) -> "GBDTRanker":
        self.model.fit(X, y)
        self.fitted = True
        return self

    def score(self, X: np.ndarray) -> np.ndarray:
        if not self.fitted:
            raise RuntimeError("ranker is not fitted")
        return self.model.predict_proba(X)[:, 1]

    def save(self, path: str):
        import pickle

        with open(path, "wb") as fh:
            pickle.dump(self.model, fh)

    @classmethod
    def load(cls, path: str) -> "GBDTRanker":
        import pickle

        obj = cls.__new__(cls)
        with open(path, "rb") as fh:
            obj.model = pickle.load(fh)
        obj.fitted = True
        return obj


def build_training_data(retrieve_fn, feature_builder: FeatureBuilder, train_df, eval_df,
                        n_users: int = 1500, candidates_k: int = 200, negatives_per_pos: int = 4,
                        seed: int = 0):
    """Positives from held-out interactions; negatives from the retriever's own slate."""
    rng = np.random.default_rng(seed)
    truth = {}
    for u, i in zip(eval_df["user_id"].to_numpy(), eval_df["item_id"].to_numpy()):
        truth.setdefault(int(u), set()).add(int(i))

    users = [u for u in truth if u in feature_builder.user_activity][:n_users]
    X_parts, y_parts = [], []

    for user in users:
        cands, scores = retrieve_fn(user, candidates_k)
        if not len(cands):
            continue
        positives = [i for i, c in enumerate(cands) if int(c) in truth[user]]
        if not positives:
            continue
        # HARD negatives: items retrieval ranked highly that the user did not take.
        negative_pool = [i for i in range(len(cands)) if int(cands[i]) not in truth[user]]
        if not negative_pool:
            continue
        n_neg = min(len(negative_pool), len(positives) * negatives_per_pos)
        negatives = rng.choice(negative_pool, size=n_neg, replace=False)

        rows = list(positives) + list(negatives)
        feats = feature_builder.build(user, cands, scores)
        X_parts.append(feats[rows])
        y_parts.append(np.array([1] * len(positives) + [0] * len(negatives), dtype="int8"))

    if not X_parts:
        return np.zeros((0, len(FEATURE_NAMES)), dtype="float32"), np.zeros(0, dtype="int8")
    return np.vstack(X_parts), np.concatenate(y_parts)


def evaluate_two_stage(retrieve_fn, ranker: GBDTRanker, feature_builder: FeatureBuilder,
                       eval_df, users, candidates_k: int = 200, k: int = 10) -> RankerMetrics:
    """Compare retrieval-only against retrieval + ranking, same candidates, same users.

    The comparison is deliberately tight: identical candidate slates, so the only
    difference is the ordering. Any lift is attributable to the ranker and not to
    the retriever having seen more items.
    """
    from .eval import ndcg_at_k, recall_at_k

    truth = {}
    for u, i in zip(eval_df["user_id"].to_numpy(), eval_df["item_id"].to_numpy()):
        truth.setdefault(int(u), set()).add(int(i))

    r_ndcg, k_ndcg, r_rec, k_rec = [], [], [], []
    for user in users:
        gold = truth.get(int(user))
        if not gold:
            continue
        cands, scores = retrieve_fn(user, candidates_k)
        if not len(cands):
            continue
        retrieval_order = [int(c) for c in cands]

        feats = feature_builder.build(user, cands, scores)
        ranked_idx = np.argsort(-ranker.score(feats))
        ranked_order = [int(cands[i]) for i in ranked_idx]

        r_ndcg.append(ndcg_at_k(retrieval_order, gold, k))
        k_ndcg.append(ndcg_at_k(ranked_order, gold, k))
        r_rec.append(recall_at_k(retrieval_order, gold, k))
        k_rec.append(recall_at_k(ranked_order, gold, k))

    rn, kn = float(np.nanmean(r_ndcg)), float(np.nanmean(k_ndcg))
    return RankerMetrics(
        ndcg_at_10_retrieval=rn,
        ndcg_at_10_ranked=kn,
        recall_at_10_retrieval=float(np.nanmean(r_rec)),
        recall_at_10_ranked=float(np.nanmean(k_rec)),
        lift_ndcg_pct=100.0 * (kn - rn) / rn if rn else float("nan"),
        n_users=len(r_ndcg),
        n_train_rows=0,
    )
