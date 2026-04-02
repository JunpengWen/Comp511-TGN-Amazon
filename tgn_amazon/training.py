"""Training loop helpers for TGM TGN on bipartite DGData."""

from __future__ import annotations

from typing import List, Tuple

import torch
import torch.nn.functional as F
from torch import nn

from tgn_amazon.RunLogger import RunLogger

from tgm import DGraph, DGBatch
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


def _set_tgn_memory_eval_mode(memory: TGNMemory) -> None:
    """Inference-style memory without ``memory.eval()`` (TGM flushes all nodes on eval → OOM)."""
    memory.training = False
    memory.msg_s_module.eval()
    memory.msg_d_module.eval()
    memory.aggr_module.eval()
    memory.time_enc.eval()
    memory.memory_updater.eval()


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
    n_logits = 0

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
        # Use memory.num_nodes (adapter metadata), not dg.num_nodes: negatives are
        # sampled in [product_lo, product_hi) and can index any global id < num_nodes,
        # while DGraph.num_nodes may be smaller than the full bipartite id space.
        assoc = torch.empty(memory.num_nodes, dtype=torch.long, device=device)
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
        valid = neg != dst
        if not valid.any():
            memory.detach()
            memory.update_state(src, dst, t, raw_msg)
            continue
        pos_logits = pos_out.squeeze(-1)[valid]
        neg_logits = neg_out.squeeze(-1)[valid]
        logits = torch.cat([pos_logits, neg_logits], dim=0)
        targets = torch.cat(
            [torch.ones_like(pos_logits), torch.zeros_like(neg_logits)], dim=0
        )
        loss = F.binary_cross_entropy_with_logits(logits, targets, reduction='sum')
        loss.backward()
        optimizer.step()
        memory.detach()
        memory.update_state(src, dst, t, raw_msg)

        total_loss += float(loss.item())
        n_logits += int(logits.numel())

    return total_loss / max(n_logits, 1)


@torch.no_grad()
def replay_train_loader_for_memory(
    memory: TGNMemory,
    gnn: GraphAttentionEmbedding,
    link_pred: LinkPredictor,
    static_proj: nn.Module | None,
    loader: DGDataLoader,
    dg: DGraph,
    device: torch.device,
    static_node_x: torch.Tensor | None,
    raw_msg_dim: int,
    hook_manager: HookManager | None,
    *,
    num_replay_epochs: int = 1,
) -> None:
    """Replay training edges in no_grad (no loss/optimizer).

    Uses eval mode on GNN / link predictor / static projection and the TGN memory
    eval forward path (via ``_set_tgn_memory_eval_mode``) so dropout does not
    perturb the replay. Hook negatives are still resampled each pass, so replay is
    not bit-identical to training.

    Each replay epoch resets memory (like ``train_epoch``), then iterates the
    loader once. ``num_replay_epochs`` should match ``TrainingConfig.epochs`` if
    you want the same number of sweeps as training; weights still differ from
    real training because there is no optimizer step.

    This is a **heuristic** stream warm-up, not a faithful replay of learned
    weights or of the exact memory tensors after SGD—do not describe it as exact
    train memory in papers unless you add that caveat.

    Unlike ``train_epoch``, replay does not mask ``neg == dst`` in the forward:
    there is no BCE. Memory updates match training (full batch ``update_state``),
    which is intentional so replay stays aligned with ``train_epoch`` memory.
    """
    _set_tgn_memory_eval_mode(memory)
    gnn.eval()
    link_pred.eval()
    if static_proj is not None:
        static_proj.eval()
    sx = static_node_x.to(device) if static_node_x is not None else None
    if hook_manager is not None:
        hook_manager.set_active_hooks('train')
    for _ in range(max(1, num_replay_epochs)):
        memory.reset_state()
        for batch in loader:
            batch = _move_batch(batch, device)
            if batch.edge_src.numel() == 0:
                continue
            if batch.neg is None:
                raise RuntimeError(
                    'batch.neg is missing; register BipartiteProductNegativeHook on the loader.'
                )
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
            n_id = torch.cat([src, dst, neg], dim=0).unique()
            assoc = torch.empty(memory.num_nodes, dtype=torch.long, device=device)
            assoc[n_id] = torch.arange(n_id.size(0), device=device)
            edge_index = torch.stack([assoc[src], assoc[dst]], dim=0)
            _embed_nodes(
                memory,
                gnn,
                static_proj,
                sx,
                n_id,
                edge_index,
                t,
                raw_msg,
            )
            memory.detach()
            memory.update_state(src, dst, t, raw_msg)


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
    logger: RunLogger | None = None,
) -> Tuple[
    List[float],
    TGNMemory,
    GraphAttentionEmbedding,
    LinkPredictor,
    nn.Module | None,
]:
    """Load train split, build TGN stack, run ``train_cfg.epochs`` epochs.

    Returns per-epoch mean losses and the trained modules for downstream eval.
    """
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

    hook_gen = torch.Generator(device=device)
    hook_gen.manual_seed(train_cfg.seed)
    hm = HookManager(keys=['train'])
    hm.register(
        'train',
        BipartiteProductNegativeHook(
            meta.num_customers,
            meta.num_customers + meta.num_products,
            generator=hook_gen,
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
        if logger is not None:
            logger.log_epoch(epoch=ep, loss=loss)

    return epoch_losses, memory, gnn, link_pred, static_proj
