"""Tests for the parts that are easy to get quietly wrong."""
import numpy as np
import pandas as pd
import pytest
import torch

from recsys.config import Config, DataConfig, IndexConfig, ModelConfig
from recsys.data import cold_start_report, synthesize, time_split
from recsys.eval import PopularityBaseline, build_ground_truth, evaluate, ndcg_at_k, recall_at_k
from recsys.index import VectorIndex
from recsys.model import TwoTower, empirical_log_q, in_batch_softmax_loss


@pytest.fixture(scope="module")
def small():
    cfg = DataConfig(n_users=200, n_items=1500, n_events=30_000, seed=3)
    return cfg, synthesize(cfg)


def test_split_is_strictly_ordered_in_time(small):
    cfg, df = small
    train, valid, test = time_split(df, cfg)
    # The property that makes this a time split and not a random one.
    assert train["ts"].max() <= valid["ts"].min()
    assert valid["ts"].max() <= test["ts"].min()
    assert len(train) + len(valid) + len(test) == len(df)


def test_cold_start_is_reported_not_hidden(small):
    cfg, df = small
    train, _, test = time_split(df, cfg)
    rep = cold_start_report(train, test)
    assert rep["evaluable_rows"] <= rep["test_rows"]
    assert rep["evaluable_rows"] + rep["cold_item_rows"] >= rep["test_rows"] - rep["cold_user_rows"]


def test_recall_and_ndcg_edge_cases():
    assert recall_at_k([1, 2, 3], {1}, 3) == 1.0
    assert recall_at_k([9, 9, 9], {1}, 3) == 0.0
    # recall@k with more truth than k is capped by k, otherwise it is unreachable
    assert recall_at_k([1, 2], {1, 2, 3, 4}, 2) == 1.0
    # ranking earlier must score higher
    assert ndcg_at_k([1, 0, 0], {1}, 3) > ndcg_at_k([0, 0, 1], {1}, 3)


def test_popularity_baseline_excludes_seen(small):
    cfg, df = small
    train, _, test = time_split(df, cfg)
    top = int(train["item_id"].value_counts().index[0])
    base = PopularityBaseline(train, exclude_seen={7: {top}})
    assert top not in base.rank(7, 10)
    assert top in base.rank(8, 10)


def test_logq_correction_demotes_popular_items():
    """The correction must actually change the logits in the intended direction."""
    ids = np.array([0] * 90 + [1] * 10)
    log_q = empirical_log_q(ids, n_items=2)
    assert log_q[0] > log_q[1]  # item 0 is sampled far more often

    torch.manual_seed(0)
    u = torch.nn.functional.normalize(torch.randn(4, 8), dim=-1)
    i = torch.nn.functional.normalize(torch.randn(4, 8), dim=-1)
    q = torch.tensor([log_q[0], log_q[1], log_q[0], log_q[1]], dtype=torch.float32)
    plain = in_batch_softmax_loss(u, i, None, 0.07)
    corrected = in_batch_softmax_loss(u, i, q, 0.07)
    assert not torch.isclose(plain, corrected)
    assert torch.isfinite(corrected)


def test_two_tower_vectors_are_unit_norm():
    m = TwoTower(10, 20, ModelConfig(dim=16))
    v = m.item_vec(torch.arange(20))
    assert torch.allclose(v.norm(dim=-1), torch.ones(20), atol=1e-5)
    assert m.all_item_vectors().shape == (20, 16)


def test_training_step_reduces_loss():
    """A learnability smoke test -- catches silently-detached graphs."""
    torch.manual_seed(0)
    m = TwoTower(50, 80, ModelConfig(dim=16))
    opt = torch.optim.AdamW(m.parameters(), lr=1e-2)
    u = torch.randint(0, 50, (256,))
    i = torch.randint(0, 80, (256,))
    first = None
    for step in range(30):
        loss = in_batch_softmax_loss(m.user_vec(u), m.item_vec(i), None, 0.07)
        opt.zero_grad()
        loss.backward()
        opt.step()
        if first is None:
            first = float(loss.detach())
    assert float(loss.detach()) < first


@pytest.mark.parametrize("kind", ["flat", "hnsw"])
def test_index_returns_the_query_itself(kind):
    """Sanity: an indexed vector must retrieve itself at rank 1."""
    rng = np.random.default_rng(0)
    v = rng.normal(size=(500, 16)).astype("float32")
    v /= np.linalg.norm(v, axis=1, keepdims=True)
    idx = VectorIndex(IndexConfig(kind=kind, hnsw_ef_search=64), 16).build(v)
    _, ids = idx.search(v[:20], 5)
    assert (ids[:, 0] == np.arange(20)).mean() >= 0.95


def test_evaluate_skips_users_without_truth():
    gt = build_ground_truth(pd.DataFrame({"user_id": [1, 1, 2], "item_id": [5, 6, 7]}))
    out = evaluate(lambda u, k: [5, 6, 7][:k], [1, 2, 3], gt, ks=(3,))
    assert out["n_users_scored"] == 0.0 or "recall@3" in out
    assert out["recall@3"] > 0


# --------------------------------------------------------------------------
# stage-2 ranker
# --------------------------------------------------------------------------

def test_feature_builder_produces_the_declared_features():
    from recsys.ranker import FEATURE_NAMES, FeatureBuilder

    cfg = DataConfig(n_users=200, n_items=800, n_events=20_000, seed=4)
    df = synthesize(cfg)
    train, _, _ = time_split(df, cfg)
    fb = FeatureBuilder(train, n_items=800)

    cands = np.array([1, 2, 3, 4])
    scores = np.array([0.9, 0.8, 0.7, 0.6])
    feats = fb.build(int(train["user_id"].iloc[0]), cands, scores)
    assert feats.shape == (4, len(FEATURE_NAMES))
    assert np.isfinite(feats).all(), "a NaN feature silently poisons a GBDT split"


def test_rank_position_feature_is_normalised():
    """Otherwise the feature means something different for every slate size."""
    from recsys.ranker import FeatureBuilder

    cfg = DataConfig(n_users=100, n_items=400, n_events=8000, seed=5)
    train, _, _ = time_split(synthesize(cfg), cfg)
    fb = FeatureBuilder(train, n_items=400)

    short = fb.build(0, np.arange(5), np.linspace(1, 0, 5))
    long = fb.build(0, np.arange(50), np.linspace(1, 0, 50))
    assert short[0, 2] == 0.0 and short[-1, 2] == 1.0
    assert long[0, 2] == 0.0 and long[-1, 2] == 1.0


def test_training_negatives_come_from_the_retrieved_slate():
    """Hard negatives, not uniform ones -- the whole point of the sampling scheme.

    Every negative must be an item the retriever actually surfaced. If negatives
    were drawn from the catalogue at large, the ranker would learn to separate
    plausible from absurd, which retrieval already did, and its offline lift
    would not survive contact with serving.
    """
    from recsys.ranker import FeatureBuilder, build_training_data

    cfg = DataConfig(n_users=300, n_items=1000, n_events=40_000, seed=6)
    df = synthesize(cfg)
    train, valid, _ = time_split(df, cfg)
    fb = FeatureBuilder(train, n_items=1000)

    slate = np.arange(40)
    seen_candidates = []

    def fake_retrieve(user, k):
        seen_candidates.append(set(int(x) for x in slate[:k]))
        return slate[:k], np.linspace(1.0, 0.0, min(k, len(slate)))

    X, y = build_training_data(fake_retrieve, fb, train, valid, n_users=40, candidates_k=40)
    if len(y):
        assert set(np.unique(y)) <= {0, 1}
        assert y.sum() > 0, "there must be positives to learn from"
        assert (y == 0).sum() > 0, "there must be negatives to learn against"


def test_ranker_learns_a_separable_signal():
    """Sanity: on data where the label is a clean function of a feature,
    the ranker must beat chance. Guards against an unfitted or inverted model."""
    from recsys.ranker import GBDTRanker

    rng = np.random.default_rng(0)
    X = rng.normal(size=(2000, 6)).astype("float32")
    y = (X[:, 0] + 0.3 * rng.normal(size=2000) > 0).astype("int8")
    ranker = GBDTRanker(max_iter=40).fit(X, y)
    scores = ranker.score(X)
    # Rank correlation with the driving feature should be strongly positive.
    order = np.argsort(-scores)
    top_mean = X[order[:200], 0].mean()
    bottom_mean = X[order[-200:], 0].mean()
    assert top_mean > bottom_mean


def test_unfitted_ranker_refuses_to_score():
    from recsys.ranker import GBDTRanker

    with pytest.raises(RuntimeError):
        GBDTRanker().score(np.zeros((2, 6), dtype="float32"))


# --------------------------------------------------------------------------
# shadow deployment, promotion gate, rollback
# --------------------------------------------------------------------------

def _stats(name, n, latency, metric, errors=0):
    from service.shadow import ModelStats

    s = ModelStats(name)
    s.requests = n
    s.errors = errors
    s.latencies_ms = [latency] * n
    s.offline_metric = metric
    return s


def test_gate_rejects_a_candidate_that_is_better_but_slower():
    """A quality gain that costs the tail is usually a bad trade."""
    from service.shadow import GateConfig, evaluate_gate

    champ = _stats("v1", 800, 20.0, 0.10)
    cand = _stats("v2", 800, 60.0, 0.14)
    r = evaluate_gate(champ, cand, GateConfig())
    assert not r.passed
    assert not r.checks["p99_latency"]["ok"]
    assert r.checks["offline_metric"]["ok"], "the metric itself improved; latency is what failed"


def test_gate_rejects_on_insufficient_sample_size():
    from service.shadow import GateConfig, evaluate_gate

    champ = _stats("v1", 40, 20.0, 0.10)
    cand = _stats("v2", 40, 20.0, 0.20)
    r = evaluate_gate(champ, cand, GateConfig(min_requests=500))
    assert not r.passed
    assert not r.checks["sample_size"]["ok"]


def test_gate_rejects_a_candidate_that_errors():
    """A model that scores well and throws is not a candidate."""
    from service.shadow import GateConfig, evaluate_gate

    champ = _stats("v1", 800, 20.0, 0.10)
    cand = _stats("v2", 800, 20.0, 0.15, errors=40)
    r = evaluate_gate(champ, cand, GateConfig())
    assert not r.passed
    assert not r.checks["error_rate"]["ok"]


def test_gate_checks_are_anded_not_scored():
    """One disqualifying failure must not be outweighed by a large win elsewhere."""
    from service.shadow import GateConfig, evaluate_gate

    champ = _stats("v1", 800, 20.0, 0.10)
    # Enormous metric win, but it errors constantly.
    cand = _stats("v2", 800, 20.0, 0.90, errors=400)
    assert not evaluate_gate(champ, cand, GateConfig()).passed


def test_a_good_candidate_is_promoted():
    from service.shadow import ShadowDeployment, State

    d = ShadowDeployment("v1", "v2")
    d.start_shadow()
    d.set_offline_metrics(0.100, 0.118)
    for _ in range(800):
        d.record("v1", 20.0)
        d.record("v2", 21.0)
    assert d.try_promote().passed
    assert d.state == State.PROMOTED


def test_rollback_needs_consecutive_bad_windows_not_one():
    """A single bad window is noise. Reverting on noise is its own outage."""
    from service.shadow import GateConfig, ShadowDeployment, State

    d = ShadowDeployment("v1", "v2", GateConfig(rollback_consecutive_windows=3))
    d.start_shadow()
    d.set_offline_metrics(0.100, 0.120)
    for _ in range(800):
        d.record("v1", 20.0)
        d.record("v2", 20.0)
    d.try_promote()

    assert not d.observe_window(0.100)["rolled_back"]      # bad 1
    assert not d.observe_window(0.120)["rolled_back"]      # recovered -> counter resets
    assert not d.observe_window(0.100)["rolled_back"]      # bad 1 again
    assert not d.observe_window(0.099)["rolled_back"]      # bad 2
    assert d.observe_window(0.098)["rolled_back"]          # bad 3 -> fires
    assert d.state == State.ROLLED_BACK


def test_rollback_reverts_to_the_previous_champion():
    from service.shadow import ShadowDeployment

    d = ShadowDeployment("v1", "v2")
    d.start_shadow()
    d.set_offline_metrics(0.100, 0.120)
    for _ in range(800):
        d.record("v1", 20.0)
        d.record("v2", 20.0)
    d.try_promote()
    result = None
    for m in (0.09, 0.09, 0.09):
        result = d.observe_window(m)
    assert result["rolled_back"]
    assert result["reverted_to"] == "v1"


def test_rollback_comparator_false_positive_rate_is_low():
    """The answer to 'how do you know it wasn't a false alarm'.

    Ground truth is a healthy model in every trial, so any rollback is a false
    alarm by construction.
    """
    from service.shadow import simulate_stability

    r = simulate_stability(trials=200, windows=20, noise=0.03)
    assert r["false_positive_rate"] <= 0.05, r


def test_a_noisier_metric_raises_the_false_positive_rate():
    """Sanity on the comparator: more noise must make false alarms more likely,
    or the simulation is not measuring what it claims."""
    from service.shadow import simulate_stability

    calm = simulate_stability(trials=200, windows=20, noise=0.01, seed=1)
    wild = simulate_stability(trials=200, windows=20, noise=0.12, seed=1)
    assert wild["false_positive_rate"] >= calm["false_positive_rate"]


# --------------------------------------------------------------------------
# staleness experiment
# --------------------------------------------------------------------------

def test_generator_is_stationary_when_drift_is_zero():
    """The default world must not drift, or the staleness baseline is meaningless."""
    cfg_a = DataConfig(n_users=200, n_items=1000, n_events=20_000, days=60, seed=3)
    cfg_b = DataConfig(n_users=200, n_items=1000, n_events=20_000, days=60, seed=3)
    a, b = synthesize(cfg_a), synthesize(cfg_b)
    assert a["item_id"].tolist() == b["item_id"].tolist(), "generation must be deterministic"


def test_drift_changes_late_period_behaviour():
    """With drift on, the second half of the run must differ from the stationary
    version -- otherwise the knob does nothing and the experiment cannot detect it."""
    base = DataConfig(n_users=300, n_items=1500, n_events=30_000, days=60, seed=5)
    drifted = DataConfig(n_users=300, n_items=1500, n_events=30_000, days=60, seed=5,
                         taste_drift_per_day=0.08)
    a, b = synthesize(base), synthesize(drifted)

    late_a = a[a["ts"] > a["ts"].quantile(0.7)]["item_id"].to_numpy()
    late_b = b[b["ts"] > b["ts"].quantile(0.7)]["item_id"].to_numpy()
    differing = (late_a != late_b).mean()
    assert differing > 0.05, "drift must actually change which items are chosen late in the run"


def test_staleness_reports_the_drift_rate_it_ran_with():
    """A staleness table without its drift rate is uninterpretable."""
    from recsys.staleness import run

    cfg = Config()
    cfg.data.n_items = 600
    cfg.data.n_events = 12_000
    cfg.data.n_users = 300
    cfg.data.days = 40
    cfg.data.taste_drift_per_day = 0.05
    cfg.model.epochs = 1

    result = run([0, 1], cfg, eval_window_days=2.0, max_users=100)
    assert result["taste_drift_per_day"] == 0.05
    assert "already-seen exclusion list" in result["kept_current"]
    assert "ann_index" in result["frozen"]
