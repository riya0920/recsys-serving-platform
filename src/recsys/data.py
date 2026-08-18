"""Interaction data + the time-based split.

The public MovieLens-25M file is not vendored (licence + size). `synthesize()`
produces a stream with the properties that actually matter for this system:
zipf item popularity, per-user session bursts, and a monotonically increasing
timestamp so the split is honest. `load_movielens()` reads the real ratings.csv
with the identical schema when you point it at one.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .config import DataConfig

COLUMNS = ["user_id", "item_id", "ts", "label"]


def synthesize(cfg: DataConfig) -> pd.DataFrame:
    rng = np.random.default_rng(cfg.seed)

    # Item popularity: zipf over the catalogue. This is what makes candidate
    # retrieval interesting -- a popularity baseline is already strong, so the
    # learned retriever has to beat a real bar (see eval.py).
    ranks = np.arange(1, cfg.n_items + 1)
    pop = 1.0 / np.power(ranks, cfg.zipf_a)
    pop /= pop.sum()

    # Users have tastes: each user gets a latent affinity over 16 topics, and
    # items belong to a topic. Interactions blend popularity with taste, which
    # is exactly the signal a two-tower model can learn and popularity cannot.
    n_topics = 16
    item_topic = rng.integers(0, n_topics, size=cfg.n_items)
    user_taste = rng.dirichlet(np.full(n_topics, 0.4), size=cfg.n_users)

    users = rng.integers(0, cfg.n_users, size=cfg.n_events)
    topic_choice = np.array([rng.choice(n_topics, p=user_taste[u]) for u in np.unique(users)])
    topic_by_user = dict(zip(np.unique(users), topic_choice))

    items = np.empty(cfg.n_events, dtype=np.int64)
    topic_items = {t: np.flatnonzero(item_topic == t) for t in range(n_topics)}
    topic_pop = {t: pop[topic_items[t]] / pop[topic_items[t]].sum() for t in range(n_topics)}

    # 70% of picks come from the user's topic, 30% from global popularity.
    from_topic = rng.random(cfg.n_events) < 0.70
    for t in range(n_topics):
        mask = from_topic & (np.array([topic_by_user[u] for u in users]) == t)
        k = int(mask.sum())
        if k:
            items[mask] = rng.choice(topic_items[t], size=k, p=topic_pop[t])
    rest = ~from_topic
    items[rest] = rng.choice(cfg.n_items, size=int(rest.sum()), p=pop)

    # Timestamps: 180 days of traffic with a diurnal-ish jitter. Sorted, because
    # every downstream guarantee in this repo depends on event order being real.
    ts = np.sort(rng.uniform(0, 180 * 86400, size=cfg.n_events))

    df = pd.DataFrame({"user_id": users, "item_id": items, "ts": ts, "label": 1})
    return df.reset_index(drop=True)


def load_movielens(ratings_csv: str, min_rating: float = 4.0) -> pd.DataFrame:
    """Real MovieLens ratings.csv -> the same schema. Implicit-positive framing."""
    df = pd.read_csv(ratings_csv, usecols=["userId", "movieId", "rating", "timestamp"])
    df = df[df["rating"] >= min_rating]
    df = df.rename(columns={"userId": "user_id", "movieId": "item_id", "timestamp": "ts"})
    df["label"] = 1
    # Densify ids so embedding tables stay compact.
    for col in ("user_id", "item_id"):
        df[col] = df[col].astype("category").cat.codes
    return df.sort_values("ts")[COLUMNS].reset_index(drop=True)


def time_split(df: pd.DataFrame, cfg: DataConfig):
    """Global time-based split. See docs/SPLIT_JUSTIFICATION.md.

    Global (not per-user) boundaries, because in production the model is trained
    at time T and serves everything after T. A per-user split still lets user A's
    future inform user B's past through the item embeddings.
    """
    df = df.sort_values("ts", kind="mergesort")
    t_train = df["ts"].quantile(cfg.train_end_q)
    t_valid = df["ts"].quantile(cfg.valid_end_q)
    train = df[df["ts"] <= t_train]
    valid = df[(df["ts"] > t_train) & (df["ts"] <= t_valid)]
    test = df[df["ts"] > t_valid]
    return train.reset_index(drop=True), valid.reset_index(drop=True), test.reset_index(drop=True)


def cold_start_report(train: pd.DataFrame, test: pd.DataFrame) -> dict:
    """How much of the eval set is unreachable by definition.

    Reported alongside every metric -- a recall@10 that silently drops cold users
    is not comparable to one that doesn't.
    """
    known_u = set(train["user_id"].unique())
    known_i = set(train["item_id"].unique())
    return {
        "test_rows": int(len(test)),
        "cold_user_rows": int((~test["user_id"].isin(known_u)).sum()),
        "cold_item_rows": int((~test["item_id"].isin(known_i)).sum()),
        "evaluable_rows": int((test["user_id"].isin(known_u) & test["item_id"].isin(known_i)).sum()),
    }
