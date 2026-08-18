"""ANN index wrapper: exact / IVF / HNSW behind one interface.

Kept thin on purpose -- the interesting artifact is bench_ann.py, which produces
the recall-vs-latency curve that justifies the operating point in config.py.
"""
from __future__ import annotations

import numpy as np

from .config import IndexConfig


class VectorIndex:
    def __init__(self, cfg: IndexConfig, dim: int):
        import faiss

        self.cfg = cfg
        self.dim = dim
        self.faiss = faiss
        kind = cfg.kind.lower()
        if kind == "flat":
            self.index = faiss.IndexFlatIP(dim)
        elif kind == "ivf":
            quantizer = faiss.IndexFlatIP(dim)
            self.index = faiss.IndexIVFFlat(quantizer, dim, cfg.ivf_nlist, faiss.METRIC_INNER_PRODUCT)
        elif kind == "hnsw":
            self.index = faiss.IndexHNSWFlat(dim, cfg.hnsw_m, faiss.METRIC_INNER_PRODUCT)
            self.index.hnsw.efConstruction = cfg.hnsw_ef_construction
        else:
            raise ValueError(f"unknown index kind: {cfg.kind}")

    def build(self, vectors: np.ndarray) -> "VectorIndex":
        vectors = np.ascontiguousarray(vectors, dtype="float32")
        if not self.index.is_trained:
            self.index.train(vectors)
        self.index.add(vectors)
        self._apply_search_params()
        return self

    def _apply_search_params(self):
        kind = self.cfg.kind.lower()
        if kind == "ivf":
            self.index.nprobe = self.cfg.ivf_nprobe
        elif kind == "hnsw":
            self.index.hnsw.efSearch = self.cfg.hnsw_ef_search

    def search(self, queries: np.ndarray, k: int):
        queries = np.ascontiguousarray(queries, dtype="float32")
        return self.index.search(queries, k)

    def save(self, path: str):
        self.faiss.write_index(self.index, path)

    @classmethod
    def load(cls, path: str, cfg: IndexConfig, dim: int) -> "VectorIndex":
        import faiss

        obj = cls.__new__(cls)
        obj.cfg, obj.dim, obj.faiss = cfg, dim, faiss
        obj.index = faiss.read_index(path)
        obj._apply_search_params()
        return obj
