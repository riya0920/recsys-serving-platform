"""Two-tower retrieval model.

Trained with in-batch negatives + sampled-softmax logQ correction. The
correction is not decoration: in-batch negatives are drawn from the *interaction*
distribution, so popular items appear as negatives far more often than uniform
sampling would imply, and the model over-penalises them. Subtracting log Q(item)
from the logits recovers an (approximately) unbiased softmax. See
Yi et al., "Sampling-Bias-Corrected Neural Modeling" (RecSys'19).
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import ModelConfig


class TwoTower(nn.Module):
    def __init__(self, n_users: int, n_items: int, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        self.user_emb = nn.Embedding(n_users, cfg.dim)
        self.item_emb = nn.Embedding(n_items, cfg.dim)
        # Small MLP on top of each tower. Deliberately small: the serving budget
        # is 100ms p99 end-to-end and the retrieval tower runs per request.
        self.user_mlp = nn.Sequential(nn.Linear(cfg.dim, cfg.dim), nn.ReLU(), nn.Linear(cfg.dim, cfg.dim))
        self.item_mlp = nn.Sequential(nn.Linear(cfg.dim, cfg.dim), nn.ReLU(), nn.Linear(cfg.dim, cfg.dim))
        nn.init.normal_(self.user_emb.weight, std=0.05)
        nn.init.normal_(self.item_emb.weight, std=0.05)

    def user_vec(self, u: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.user_mlp(self.user_emb(u)), dim=-1)

    def item_vec(self, i: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.item_mlp(self.item_emb(i)), dim=-1)

    def forward(self, u, i):
        return (self.user_vec(u) * self.item_vec(i)).sum(-1)

    @torch.no_grad()
    def all_item_vectors(self, batch: int = 8192) -> np.ndarray:
        """Materialise the item matrix for the ANN index."""
        self.eval()
        out = []
        n = self.item_emb.num_embeddings
        for s in range(0, n, batch):
            idx = torch.arange(s, min(s + batch, n), device=self.item_emb.weight.device)
            out.append(self.item_vec(idx).cpu().numpy())
        return np.concatenate(out).astype("float32")


def in_batch_softmax_loss(
    user_v: torch.Tensor,
    item_v: torch.Tensor,
    log_q: torch.Tensor | None,
    temperature: float,
) -> torch.Tensor:
    """Rows = users, columns = the other items in the batch as negatives.

    log_q: log of the sampling probability of each in-batch item. When supplied,
    logits are corrected by -log_q, which is the whole point of the correction.
    """
    logits = user_v @ item_v.t() / temperature
    if log_q is not None:
        logits = logits - log_q.unsqueeze(0)
    target = torch.arange(user_v.size(0), device=user_v.device)
    return F.cross_entropy(logits, target)


def empirical_log_q(item_ids: np.ndarray, n_items: int, smoothing: float = 1.0) -> np.ndarray:
    """log P(item appears in a batch) estimated from training frequencies.

    A streaming Count-Min / exponential-decay estimator is what you'd run in
    production (frequencies drift); the batch estimate is the honest offline
    equivalent and is what the reported numbers use.
    """
    counts = np.bincount(item_ids, minlength=n_items).astype("float64") + smoothing
    return np.log(counts / counts.sum())
