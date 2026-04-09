"""Experiment / ablation flags matching project_proposal.tex."""

from __future__ import annotations

from dataclasses import dataclass


# Fixed training configuration for ALL experiments.
# Tune on the validation split of the full TGN baseline once; then keep fixed so
# ablations differ only in data/model flags (control variables).
@dataclass
class TrainingConfig:
    learning_rate: float = 1e-4
    batch_size: int = 512
    epochs: int = 1
    memory_dim: int = 64
    time_dim: int = 64
    embedding_dim: int = 64
    seed: int = 42
    # Early stopping on validation BCE (same protocol as train_epoch; lower is better).
    # None disables. When set, best weights are restored (train may stop before epochs).
    early_stop_patience: int | None = None
    early_stop_min_delta: float = 0.0
    # Cap val edges for the monitoring pass only (None = full val window).
    early_stop_val_max_edges: int | None = None


@dataclass
class AblationConfig:
    """Maps to proposal ablations: Static, Homogeneous, No Features, No Memory."""

    # RQ1: temporality — if True, omit wall-clock edge times: constant zeros + time_delta "r"
    # (see adapter.build_dgdata; no global ordinal ranks, no calendar seconds).
    static_graph: bool = False
    # RQ2: heterogeneity — if True, drop explicit node_type / edge_type tensors.
    homogeneous: bool = False
    # RQ3: features — if False, omit edge_x and static_node_x (no extra input channels).
    use_features: bool = True
    # RQ4: memory aggregation is implemented in tgn_model (Last vs Mean), not here.
    # If False, build_dgdata raises until a no-memory baseline exists.
    use_memory: bool = True

    # Data limits (for debugging before full-scale runs). Applied when building
    # the train graph; eval in run_eval_job uses a copy with this cleared so val/test
    # are not truncated by the same cap unless you only use one build path.
    max_review_edges: int | None = None  # None = use all edges after filters

    def slug(self) -> str:
        parts = ["full"]
        if self.static_graph:
            parts.append("static")
        if self.homogeneous:
            parts.append("homo")
        if not self.use_features:
            parts.append("nofeat")
        if not self.use_memory:
            parts.append("nomem")
        return "_".join(parts)
