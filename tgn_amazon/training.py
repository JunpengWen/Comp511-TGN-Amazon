"""Training loop helpers for TGM TGN on bipartite DGData."""

from __future__ import annotations

from typing import List

import torch
import torch.nn.functional as F
from torch import nn

from tgm import DGraph
from tgm.data import DGDataLoader
from tgm.hooks import HookManager
from tgm.nn import LinkPredictor
from tgm.nn.encoder.tgn import GraphAttentionEmbedding, TGNMemory

from tgn_amazon.adapter import RelbenchAmazonAdapter
from tgn_amazon.config import AblationConfig, TrainingConfig
from tgn_amazon.hooks import BipartiteProductNegativeHook
from tgn_amazon.tgn_model import build_tgn_stack


def _move_batch(batch: DGBatch, device: torch.device) -> DGBatch:
    batch.edge_src = batch.edge_src.to(device)
    batch.edge_dst = batch.edge_dst.to(device)
    batch.edge_time = batch.edge_time.to(device)
    if batch.edge_x is not None:
        batch.edge_x = batch.edge_x.to(device)
    if batch.neg is not None:
        batch.neg = batch.neg.to(device)
    if batch.neg_time is not None:
        batch.neg_time = batch.neg_time.to(device)
    return batch


def _embed_nodes(
    memory: TGNMemory,
    gnn: GraphAttentionEmbedding,
    static_proj: nn.Module | None,
    static_node_x: torch.Tensor | None,
    n_id: torch.Tensor,
    edge_index: torch.Tensor,
    t: torch.Tensor,
    raw_msg: torch.Tensor,
) -> torch.Tensor:
    z, last_update = memory(n_id)
    if static_proj is not None and static_node_x is not None:
        z = static_proj(torch.cat([z, static_node_x[n_id]], dim=-1))
    return gnn(z, last_update, edge_index, t, raw_msg)


def train_epoch(
    memory: TGNMemory,
    gnn: GraphAttentionEmbedding,
    link_pred: LinkPredictor,
    static_proj: nn.Module | None,
    loader: DGDataLoader,
    dg: DGraph,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    static_node_x: torch.Tensor | None,
    raw_msg_dim: int,
    hook_manager: HookManager | None,
) -> float:
    memory.train()
    gnn.train()
    link_pred.train()
    if static_proj is not None:
        static_proj.train()
    memory.reset_state()

    sx = static_node_x.to(device) if static_node_x is not None else None

    if hook_manager is not None:
        hook_manager.set_active_hooks('train')

    total_loss = 0.0
    n_edges = 0

    for batch in loader:
        batch = _move_batch(batch, device)
        if batch.edge_src.numel() == 0:
            continue
        if batch.neg is None:
            raise RuntimeError('batch.neg is missing; register BipartiteProductNegativeHook on the loader.')

        src = batch.edge_src.long()
        dst = batch.edge_dst.long()
        t = batch.edge_time.long()
        if batch.edge_x is not None:
            raw_msg = batch.edge_x.float()
        else:
            raw_msg = torch.zeros(
                (src.size(0), raw_msg_dim), dtype=torch.float32, device=device
            )

        neg = batch.neg.long()
        if neg.shape != dst.shape:
            raise ValueError(f'neg shape {neg.shape} != dst shape {dst.shape}')

        n_id = torch.cat([src, dst, neg], dim=0).unique()
        assoc = torch.empty(dg.num_nodes, dtype=torch.long, device=device)
        assoc[n_id] = torch.arange(n_id.size(0), device=device)

        # GNN expects edge_index with local ids 0..len(n_id)-1 (same layout as memory(n_id)).
        edge_index = torch.stack([assoc[src], assoc[dst]], dim=0)

        optimizer.zero_grad(set_to_none=True)
        z = _embed_nodes(
            memory,
            gnn,
            static_proj,
            sx,
            n_id,
            edge_index,
            t,
            raw_msg,
        )
        pos_out = link_pred(z[assoc[src]], z[assoc[dst]])
        neg_out = link_pred(z[assoc[src]], z[assoc[neg]])
        loss = F.binary_cross_entropy_with_logits(
            pos_out, torch.ones_like(pos_out)
        ) + F.binary_cross_entropy_with_logits(
            neg_out, torch.zeros_like(neg_out)
        )
        loss.backward()
        optimizer.step()
        memory.detach()
        memory.update_state(src, dst, t, raw_msg)

        total_loss += float(loss.item()) * src.size(0)
        n_edges += int(src.size(0))

    return total_loss / max(n_edges, 1)


def make_train_loader(
    dg: DGraph,
    batch_size: int,
    hook_manager: HookManager | None,
) -> DGDataLoader:
    return DGDataLoader(
        dg,
        batch_size=batch_size,
        batch_unit='r',
        on_empty='skip',
        hook_manager=hook_manager,
    )


def raw_msg_dim_from_config(abl_cfg: AblationConfig) -> int:
    return 2 if abl_cfg.use_features else 1


def run_training_job(
    adapter: RelbenchAmazonAdapter,
    abl_cfg: AblationConfig,
    train_cfg: TrainingConfig,
    *,
    use_last_aggregator: bool = True,
    label: str = 'TGN',
) -> List[float]:
    """Load train split, build TGN stack, run ``train_cfg.epochs`` epochs; return mean loss per epoch."""
    torch.manual_seed(train_cfg.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    dg_data, meta = adapter.build_dgdata(
        abl_cfg, until_timestamp=adapter.dataset.val_timestamp
    )
    dg = DGraph(dg_data, device=device)

    static_node_x = dg_data.static_node_x
    static_dim = int(static_node_x.shape[1]) if static_node_x is not None else 0
    raw_dim = raw_msg_dim_from_config(abl_cfg)

    memory, gnn, link_pred, static_proj = build_tgn_stack(
        meta.num_nodes,
        raw_dim,
        train_cfg.memory_dim,
        train_cfg.time_dim,
        train_cfg.embedding_dim,
        static_dim,
        use_last_aggregator=use_last_aggregator,
        device=device,
    )

    hm = HookManager(keys=['train'])
    hm.register(
        'train',
        BipartiteProductNegativeHook(
            meta.num_customers, meta.num_customers + meta.num_products
        ),
    )

    def _unique_params(*mods: nn.Module | None) -> list[nn.Parameter]:
        seen: set[int] = set()
        out: list[nn.Parameter] = []
        for m in mods:
            if m is None:
                continue
            for p in m.parameters():
                i = id(p)
                if i not in seen:
                    seen.add(i)
                    out.append(p)
        return out

    opt = torch.optim.Adam(
        _unique_params(memory, gnn, link_pred, static_proj),
        lr=train_cfg.learning_rate,
    )

    loader = make_train_loader(dg, train_cfg.batch_size, hm)
    epoch_losses: List[float] = []

    for ep in range(1, train_cfg.epochs + 1):
        loss = train_epoch(
            memory,
            gnn,
            link_pred,
            static_proj,
            loader,
            dg,
            opt,
            device,
            static_node_x,
            raw_dim,
            hm,
        )
        epoch_losses.append(loss)
        print(f'  [{label}] epoch {ep}/{train_cfg.epochs}  mean_loss={loss:.6f}')

    return epoch_losses, memory, gnn, link_pred, static_proj
