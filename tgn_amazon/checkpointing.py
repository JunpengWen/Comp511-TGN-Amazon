"""Save / load TGN training checkpoints (weights + config for eval without retraining)."""

from __future__ import annotations

from dataclasses import asdict, fields
from pathlib import Path
from typing import Any

import torch
from torch import nn

from tgn_amazon.config import AblationConfig, TrainingConfig


def save_training_checkpoint(
    path: str | Path,
    *,
    memory: nn.Module,
    gnn: nn.Module,
    link_pred: nn.Module,
    static_proj: nn.Module | None,
    num_nodes: int,
    raw_dim: int,
    static_dim: int,
    use_last_aggregator: bool,
    abl: AblationConfig,
    tc: TrainingConfig,
    run_id: str | None = None,
) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {
        'memory': memory.state_dict(),
        'gnn': gnn.state_dict(),
        'link_pred': link_pred.state_dict(),
        'static_proj': static_proj.state_dict() if static_proj is not None else None,
        'num_nodes': num_nodes,
        'raw_dim': raw_dim,
        'static_dim': static_dim,
        'use_last_aggregator': use_last_aggregator,
        'ablation': asdict(abl),
        'training_config': asdict(tc),
        'run_id': run_id,
        'format_version': 1,
    }
    torch.save(payload, path)
    return path


def load_training_checkpoint_dict(path: str | Path, map_location: str | torch.device) -> dict[str, Any]:
    """Load checkpoint dict (pickled payload; not weights-only)."""
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)


def _merge_dataclass_dict(cls: type, saved: dict[str, Any]) -> dict[str, Any]:
    """Fill missing keys from ``cls()`` defaults (older checkpoints, new fields)."""
    defaults = {f.name: getattr(cls(), f.name) for f in fields(cls)}
    return {**defaults, **saved}


def configs_from_checkpoint(ckpt: dict[str, Any]) -> tuple[AblationConfig, TrainingConfig]:
    return (
        AblationConfig(**_merge_dataclass_dict(AblationConfig, ckpt['ablation'])),
        TrainingConfig(**_merge_dataclass_dict(TrainingConfig, ckpt['training_config'])),
    )
