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


@dataclass
class AblationConfig:
    """Maps to proposal ablations: Static, Homogeneous, No Features, No Memory."""

    # RQ1: temporality — if True, use event-rank times only (no calendar time).
    static_graph: bool = False
    # RQ2: heterogeneity — if True, drop explicit node_type / edge_type tensors.
    homogeneous: bool = False
    # RQ3: features — if False, zero out edge and static node features.
    use_features: bool = True
    # RQ4: memory — handled in training code (swap TGNMemory vs mean pooling), not in DGData.
    use_memory: bool = True

    # Data limits (for debugging before full-scale runs)
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
