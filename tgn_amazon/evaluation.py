from __future__ import annotations

import copy
from typing import Dict

import torch
from torch import nn

from tgm import DGraph
from tgm.data import DGDataLoader
from tgm.hooks import HookManager
from tgm.nn import LinkPredictor
from tgm.nn.encoder.tgn import GraphAttentionEmbedding, TGNMemory

from tgn_amazon.adapter import RelbenchAmazonAdapter
from tgn_amazon.config import AblationConfig, TrainingConfig
from tgn_amazon.hooks import BipartiteProductNegativeHook
from tgn_amazon.training import (
    _embed_nodes,
    make_train_loader,
    raw_msg_dim_from_config,
)

def eval_mrr(
    memory: TGNMemory,
    gnn: GraphAttentionEmbedding,
    link_pred: LinkPredictor,
    static_proj: nn.Module | None,
    loader: DGDataLoader,
    dg: DGraph,
    device: torch.device,
    hook_manager: HookManager | None,
    static_node_x: torch.Tensor | None,
    raw_msg_dim: int,
    num_nodes: int,
    num_negatives: int | None,
    seed: int = 0,
) -> Dict[str, float]:
    memory.eval()
    gnn.eval()
    link_pred.eval()
    if static_proj is not None:
        static_proj.eval()

    sx = static_node_x.to(device) if static_node_x is not None else None

    if hook_manager is not None:
        hook_manager.set_active_hooks('test')

    rng = torch.Generator(device=device)
    rng.manual_seed(seed)

    sum_rr = 0.0
    n_queries = 0

    with torch.no_grad():
        for batch in loader:
            batch.edge_src  = batch.edge_src.to(device)
            batch.edge_dst  = batch.edge_dst.to(device)
            batch.edge_time = batch.edge_time.to(device)
            if batch.edge_x is not None:
                batch.edge_x = batch.edge_x.to(device)
            if hasattr(batch, 'neg') and batch.neg is not None:
                batch.neg = batch.neg.to(device)

            if batch.edge_src.numel() == 0:
                continue
            if not hasattr(batch, 'neg') or batch.neg is None:
                raise RuntimeError('batch.neg is missing; register BipartiteProductNegativeHook on the loader.')

            src = batch.edge_src.long()
            dst = batch.edge_dst.long()
            t   = batch.edge_time.long()
            neg = batch.neg.long()

            known = (src < num_nodes) & (dst < num_nodes)
            src, dst, t, neg = src[known], dst[known], t[known], neg[known]
            if src.numel() == 0:
                continue

            if batch.edge_x is not None:
                raw_msg = batch.edge_x[known].float()
            else:
                raw_msg = torch.zeros(
                    (src.size(0), raw_msg_dim), dtype=torch.float32, device=device
                )

            if neg.shape != dst.shape:
                raise ValueError(f'neg shape {neg.shape} != dst shape {dst.shape}')

            n_id = torch.cat([src, dst, neg], dim=0).unique()
            n_id = n_id[n_id < num_nodes]

            assoc = torch.empty(num_nodes, dtype=torch.long, device=device)
            assoc[n_id] = torch.arange(n_id.size(0), device=device)

            edge_index = torch.stack([assoc[src], assoc[dst]], dim=0)

            z = _embed_nodes(
                memory, gnn, static_proj, sx,
                n_id, edge_index, t, raw_msg,
            )

            candidates = torch.cat([dst.unsqueeze(1), neg.unsqueeze(1)], dim=1)
            src_emb  = z[assoc[src]]
            cand_emb = z[assoc[candidates.reshape(-1)]].view(
                candidates.size(0), candidates.size(1), -1
            )

            src_exp = src_emb.unsqueeze(1).expand_as(cand_emb)
            scores  = link_pred(
                src_exp.reshape(-1, src_emb.size(-1)),
                cand_emb.reshape(-1, src_emb.size(-1)),
            ).reshape(candidates.size(0), candidates.size(1))

            true_scores = scores[:, 0].unsqueeze(1)
            rank = (scores > true_scores).sum(dim=1) + 1

            sum_rr += (1.0 / rank.float()).sum().item()
            n_queries += src.size(0)

    if n_queries == 0:
        return {"mrr": 0.0}

    return {"mrr": sum_rr / n_queries}


def run_eval_job(
    adapter: RelbenchAmazonAdapter,
    abl_cfg: AblationConfig,
    train_cfg: TrainingConfig,
    memory: TGNMemory,
    gnn: GraphAttentionEmbedding,
    link_pred: LinkPredictor,
    static_proj: nn.Module | None,
    num_negatives: int | None,
    split: str = "val",
    label: str = "TGN",
) -> Dict[str, float]:
    torch.manual_seed(train_cfg.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if split == "val":
        t_end = adapter.dataset.test_timestamp
    elif split == "test":
        t_end = None
    else:
        raise ValueError(f"split must be 'val' or 'test', got '{split}'")

    dg_train_data, train_meta = adapter.build_dgdata(
        abl_cfg, until_timestamp=adapter.dataset.val_timestamp
    )
    dg_eval_data, eval_meta = adapter.build_dgdata(
        abl_cfg, until_timestamp=t_end
    )
    dg_eval = DGraph(dg_eval_data, device=device)

    static_node_x = dg_eval_data.static_node_x
    raw_dim = raw_msg_dim_from_config(abl_cfg)

    hm = HookManager(keys=['test'])
    hm.register(
        'test',
        BipartiteProductNegativeHook(
            eval_meta.num_customers, eval_meta.num_customers + eval_meta.num_products
        ),
    )
    eval_loader = make_train_loader(dg_eval, train_cfg.batch_size, hook_manager=hm)

    memory_snapshot = copy.deepcopy(memory.state_dict())

    print(f"  [{label}] evaluating on {split} split")

    metrics = eval_mrr(
        memory=memory,
        gnn=gnn,
        link_pred=link_pred,
        static_proj=static_proj,
        loader=eval_loader,
        dg=dg_eval,
        device=device,
        hook_manager=hm,
        static_node_x=static_node_x,
        raw_msg_dim=raw_dim,
        num_nodes=train_meta.num_nodes,
        num_negatives=num_negatives,
    )

    memory.load_state_dict(memory_snapshot)

    print(f"  [{label}] {split}  MRR={metrics['mrr']:.4f}")

    return metrics
