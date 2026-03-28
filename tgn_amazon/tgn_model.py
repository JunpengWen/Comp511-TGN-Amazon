"""TGM TGN stack (memory + graph attention + link predictor) for RelBench Amazon."""

from __future__ import annotations

from typing import Tuple

import torch
from torch import nn

from tgm.nn import LinkPredictor
from tgm.nn.encoder.tgn import (
    GraphAttentionEmbedding,
    IdentityMessage,
    LastAggregator,
    MeanAggregator,
    TGNMemory,
)


def build_tgn_stack(
    num_nodes: int,
    raw_msg_dim: int,
    memory_dim: int,
    time_dim: int,
    embedding_dim: int,
    static_dim: int,
    *,
    use_last_aggregator: bool = True,
    device: torch.device | None = None,
) -> Tuple[TGNMemory, GraphAttentionEmbedding, LinkPredictor, nn.Module | None]:
    """Assemble TGNMemory + GraphAttentionEmbedding + LinkPredictor.

    When ``use_last_aggregator`` is False, a MeanAggregator is used as the
    message aggregator inside TGNMemory. A MeanAggregator is a simplified
    baseline for TGNMemory's default last-message aggregation, measuring how
    much a richer aggregation rule matters for this task (RQ4).

    If ``static_dim`` > 0, a linear map projects concatenated
    ``[memory || static_node_x]`` down to ``memory_dim`` before the GNN.
    """
    dev = device or torch.device('cpu')
    aggr = LastAggregator() if use_last_aggregator else MeanAggregator()
    msg_mod = IdentityMessage(raw_msg_dim, memory_dim, time_dim)
    memory = TGNMemory(
        num_nodes,
        raw_msg_dim,
        memory_dim,
        time_dim,
        message_module=msg_mod,
        aggregator_module=aggr,
    ).to(dev)

    gnn_in = memory_dim + static_dim
    static_proj: nn.Module | None = None
    if static_dim > 0:
        static_proj = nn.Linear(gnn_in, memory_dim, bias=False).to(dev)

    gnn = GraphAttentionEmbedding(
        in_channels=memory_dim,
        out_channels=embedding_dim,
        msg_dim=raw_msg_dim,
        time_enc=memory.time_enc,
    ).to(dev)

    link_pred = LinkPredictor(
        node_dim=embedding_dim,
        out_dim=1,
        nlayers=2,
        hidden_dim=embedding_dim,
        merge_op='concat',
    ).to(dev)

    return memory, gnn, link_pred, static_proj
