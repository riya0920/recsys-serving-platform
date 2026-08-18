"""Single source of truth for every knob. Anything tuned in an experiment lives here."""
from dataclasses import dataclass, asdict, field
import json


@dataclass
class DataConfig:
    n_users: int = 6_000
    n_items: int = 50_000          # spec floor: >=50K items for retrieval to be non-trivial
    n_events: int = 1_200_000
    # Time-based split boundaries expressed as quantiles of the event timestamp.
    # Rationale in docs/SPLIT_JUSTIFICATION.md -- a random split leaks the future.
    train_end_q: float = 0.85
    valid_end_q: float = 0.925
    zipf_a: float = 1.15           # item popularity skew; also used by the load generator
    seed: int = 17


@dataclass
class ModelConfig:
    dim: int = 64
    n_negatives: str = "in-batch"  # in-batch sampled softmax with logQ correction
    logq_correction: bool = True
    lr: float = 3e-3
    weight_decay: float = 1e-6
    batch_size: int = 4096
    epochs: int = 3
    temperature: float = 0.07


@dataclass
class IndexConfig:
    # flat | ivf | hnsw. Measured, not assumed: at 50K items exact search beats
    # HNSW on BOTH recall and p50/p99 latency (docs/ANN_OPERATING_POINT.md).
    # Revisit at ~1-5M items, where exact search starts eating the p99 budget.
    kind: str = "flat"
    ivf_nlist: int = 512
    ivf_nprobe: int = 16
    hnsw_m: int = 32
    hnsw_ef_construction: int = 80
    hnsw_ef_search: int = 64
    retrieval_k: int = 500         # spec: top-500 candidates handed to the ranker


@dataclass
class ServiceConfig:
    fallback_popularity_k: int = 500
    model_timeout_ms: int = 40     # budget before we shed to the fallback path
    p99_target_ms: int = 100


@dataclass
class Config:
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    index: IndexConfig = field(default_factory=IndexConfig)
    service: ServiceConfig = field(default_factory=ServiceConfig)

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2)


DEFAULT = Config()
