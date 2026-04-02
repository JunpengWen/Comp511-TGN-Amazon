from __future__ import annotations

import copy
from dataclasses import replace
from typing import Any, Dict

import numpy as np
import torch
from torch import nn

from tgm import DGraph
from tgm.data import DGDataLoader
from tgm.hooks import HookManager
from tgm.nn import LinkPredictor
from tgm.nn.encoder.tgn import GraphAttentionEmbedding, TGNMemory
from tgn_amazon.RunLogger import RunLogger

from tgn_amazon.adapter import AdapterMetadata, RelbenchAmazonAdapter
from tgn_amazon.config import AblationConfig, TrainingConfig
from tgn_amazon.hooks import BipartiteProductNegativeHook
from tgn_amazon.training import (
    _embed_nodes,
    _set_tgn_memory_eval_mode,
    make_train_loader,
    raw_msg_dim_from_config,
    replay_train_loader_for_memory,
)


# Small enough to materialize [lo, hi) \ {dv} for exact torch.randperm sampling.
_SMALL_PRODUCT_POOL = 8192


def _validate_num_negatives_for_eval(
    num_negatives: int,
    train_meta: AdapterMetadata,
) -> None:
    """Fail fast before ``eval_mrr`` if ``num_negatives`` would force k_eff == max_uniq on a large catalog."""
    k = max(1, int(num_negatives))
    max_uniq = train_meta.num_products - 1
    if max_uniq <= 0:
        return
    if train_meta.num_products > _SMALL_PRODUCT_POOL + 1 and k >= max_uniq:
        raise ValueError(
            f'num_negatives ({num_negatives}) must be less than num_products - 1 = {max_uniq} '
            f'(num_products={train_meta.num_products}) on large catalogs. '
            f'Otherwise k_eff == max_uniq and negatives cannot be sampled without materializing all product ids.'
        )


def _indices_to_product_ids(
    indices: torch.Tensor,
    lo: int,
    hi: int,
    dv: int,
) -> torch.Tensor:
    """Map indices in [0, max_uniq) to distinct product ids in [lo, hi), skipping dv."""
    in_range = lo <= dv < hi
    if not in_range:
        return lo + indices
    offset = dv - lo
    out = torch.empty_like(indices)
    mask = indices < offset
    out[mask] = lo + indices[mask]
    out[~mask] = lo + indices[~mask] + 1
    return out


def _sample_negatives_one(
    dv: int,
    k: int,
    lo: int,
    hi: int,
    device: torch.device,
    gen: torch.Generator,
) -> tuple[torch.Tensor | None, bool]:
    """Sample up to ``min(k, max_uniq)`` distinct ids in ``[lo, hi)``, all ``!= dv``.

    Returns ``(tensor, False)`` on success; ``(None, False)`` if no valid negative
    exists; ``(None, True)`` if ``k_eff == max_uniq`` on a large pool (cannot
    materialize all ids — skip MRR for this query; ``run_eval_job`` validates upfront).

    Uses O(max_uniq) memory only when ``max_uniq`` is small; for large product
    vocabularies samples distinct indices via NumPy (no full ``arange(lo, hi)``).
    Large ``k_eff`` makes ``numpy.choice(..., replace=False)`` costly — keep
    ``num_negatives`` modest unless you accept slower eval.
    RNG: one ``torch.randint`` advances ``gen``, then ``numpy.random.Generator``
    draws the negative set (not a single pure-PyTorch stream).
    """
    pool = hi - lo
    if pool <= 0:
        raise ValueError('product range [lo, hi) is empty')
    in_range = lo <= dv < hi
    max_uniq = pool - (1 if in_range else 0)
    if max_uniq == 0:
        return None, False
    k_eff = min(k, max_uniq)

    if max_uniq <= _SMALL_PRODUCT_POOL:
        valid = torch.arange(lo, hi, device=device, dtype=torch.long)
        if in_range:
            valid = valid[valid != dv]
        perm = torch.randperm(valid.numel(), generator=gen, device=device)
        return valid[perm[:k_eff]], False

    # k_eff == max_uniq here implies max_uniq > _SMALL_PRODUCT_POOL (else handled above).
    if k_eff == max_uniq:
        return None, True

    seed_t = torch.randint(
        0, 2**31, (1,), device=device, dtype=torch.int64, generator=gen
    )
    nrng = np.random.default_rng(int(seed_t.item()))
    idx = nrng.choice(max_uniq, size=k_eff, replace=False)
    indices = torch.tensor(np.asarray(idx, dtype=np.int64), dtype=torch.long, device=device)
    return _indices_to_product_ids(indices, lo, hi, dv), False


def _sample_negatives(
    dst: torch.Tensor,
    k: int,
    lo: int,
    hi: int,
    gen: torch.Generator,
) -> tuple[torch.Tensor | None, bool]:
    """Sample up to ``k`` distinct product ids in [lo, hi), all != dst when possible.

    Only ``dst.shape[0] == 1`` is supported (``eval_mrr`` contract).

    Returns ``(tensor, False)`` on success; ``(None, False)`` if no valid negative;
    ``(None, True)`` if the query would require materializing the full negative set
    on a large catalog. Does **not** pad to ``k`` with duplicate draws.
    """
    if dst.size(0) != 1:
        raise ValueError(
            '_sample_negatives only supports batch size 1 (eval_mrr); '
            f'got dst.shape[0]={dst.size(0)}.'
        )
    device = dst.device
    dv = int(dst.squeeze(0).item())
    row, skip_full = _sample_negatives_one(dv, k, lo, hi, device, gen)
    if row is None:
        return None, skip_full
    return row.unsqueeze(0), False


def _restore_all_train_mode(
    memory: TGNMemory,
    gnn: GraphAttentionEmbedding,
    link_pred: LinkPredictor,
    static_proj: nn.Module | None,
) -> None:
    memory.train()
    gnn.train()
    link_pred.train()
    if static_proj is not None:
        static_proj.train()


def eval_mrr(
    memory: TGNMemory,
    gnn: GraphAttentionEmbedding,
    link_pred: LinkPredictor,
    static_proj: nn.Module | None,
    loader: DGDataLoader,
    device: torch.device,
    static_node_x: torch.Tensor | None,
    raw_msg_dim: int,
    train_meta: AdapterMetadata,
    num_negatives: int,
    seed: int = 0,
    assoc_buf: torch.Tensor | None = None,
) -> Dict[str, Any]:
    """MRR with 1 positive + up to ``num_negatives`` distinct random product negatives.

    When the product pool cannot supply ``k`` distinct negatives != ``dst``, fewer
    negatives are used (no padding with duplicates). Edges with no valid negative
    at all are skipped for MRR but still advance ``update_state`` so memory stays
    aligned with the stream. Skips are counted in ``n_skipped_no_negative_pool``;
    ``n_skipped_would_materialize_full_catalog`` counts large-pool queries that
    would require sampling every distinct negative (defensive; ``run_eval_job``
    validates ``num_negatives`` upfront).

    Does **not** reset memory at entry: starts from post-training or post-replay state.
    """
    _set_tgn_memory_eval_mode(memory)
    gnn.eval()
    link_pred.eval()
    if static_proj is not None:
        static_proj.eval()

    num_nodes = train_meta.num_nodes
    product_lo = train_meta.num_customers
    product_hi = train_meta.num_nodes
    k = max(1, int(num_negatives))

    sx = static_node_x.to(device) if static_node_x is not None else None

    rng = torch.Generator(device=device)
    rng.manual_seed(seed)

    sum_rr = 0.0
    n_queries = 0
    n_skipped_no_pool = 0
    n_skipped_would_materialize_full_catalog = 0

    if assoc_buf is None:
        assoc_buf = torch.empty(num_nodes, dtype=torch.long, device=device)

    with torch.no_grad():
        for batch in loader:
            batch.edge_src = batch.edge_src.to(device)
            batch.edge_dst = batch.edge_dst.to(device)
            batch.edge_time = batch.edge_time.to(device)
            if batch.edge_x is not None:
                batch.edge_x = batch.edge_x.to(device)

            if batch.edge_src.numel() == 0:
                continue

            src = batch.edge_src.long()
            dst = batch.edge_dst.long()
            t = batch.edge_time.long()

            known = (src < num_nodes) & (dst < num_nodes)
            src = src[known]
            dst = dst[known]
            t = t[known]
            if src.numel() == 0:
                continue

            if batch.edge_x is not None:
                edge_x = batch.edge_x[known].float()
            else:
                edge_x = None

            for i in range(src.size(0)):
                if edge_x is not None:
                    raw_msg_i = edge_x[i : i + 1]
                else:
                    raw_msg_i = torch.zeros(
                        (1, raw_msg_dim), dtype=torch.float32, device=device
                    )

                negs_t, skip_full_catalog = _sample_negatives(
                    dst[i : i + 1], k, product_lo, product_hi, rng
                )
                if negs_t is None:
                    if skip_full_catalog:
                        n_skipped_would_materialize_full_catalog += 1
                    else:
                        n_skipped_no_pool += 1
                    memory.detach()
                    memory.update_state(
                        src[i : i + 1],
                        dst[i : i + 1],
                        t[i : i + 1],
                        raw_msg_i,
                    )
                    continue
                negs = negs_t.squeeze(0)

                cand = torch.cat([dst[i : i + 1], negs], dim=0)
                n_id = torch.cat([src[i : i + 1], cand], dim=0).unique()
                n_id = n_id[n_id < num_nodes]

                assoc_buf[n_id] = torch.arange(n_id.size(0), device=device)

                edge_index = torch.stack(
                    [assoc_buf[src[i]], assoc_buf[dst[i]]], dim=0
                ).unsqueeze(1)

                z = _embed_nodes(
                    memory,
                    gnn,
                    static_proj,
                    sx,
                    n_id,
                    edge_index,
                    t[i : i + 1],
                    raw_msg_i,
                )

                s_idx = assoc_buf[src[i]]
                scores = link_pred(
                    z[s_idx].unsqueeze(0).expand(cand.size(0), -1),
                    z[assoc_buf[cand]],
                ).squeeze(-1)

                s0 = scores[0]
                eq = torch.isclose(scores, s0.expand_as(scores), rtol=1e-5, atol=1e-8)
                tie = eq.sum().item()
                better = ((scores > s0) & ~eq).sum().item()
                avg_rank = 1.0 + better + (tie - 1) / 2.0
                sum_rr += 1.0 / avg_rank
                n_queries += 1

                memory.detach()
                memory.update_state(
                    src[i : i + 1],
                    dst[i : i + 1],
                    t[i : i + 1],
                    raw_msg_i,
                )

    if n_queries == 0:
        return {
            "mrr": 0.0,
            "n_queries": 0,
            "n_skipped_no_negative_pool": n_skipped_no_pool,
            "n_skipped_would_materialize_full_catalog": n_skipped_would_materialize_full_catalog,
        }

    return {
        "mrr": sum_rr / n_queries,
        "n_queries": n_queries,
        "n_skipped_no_negative_pool": n_skipped_no_pool,
        "n_skipped_would_materialize_full_catalog": n_skipped_would_materialize_full_catalog,
    }


def run_eval_job(
    adapter: RelbenchAmazonAdapter,
    abl_cfg: AblationConfig,
    train_cfg: TrainingConfig,
    memory: TGNMemory,
    gnn: GraphAttentionEmbedding,
    link_pred: LinkPredictor,
    static_proj: nn.Module | None,
    num_negatives: int,
    split: str = "val",
    label: str = "TGN",
    logger: RunLogger | None = None,
    *,
    replay_train_before_eval: bool = False,
) -> Dict[str, Any]:
    torch.manual_seed(train_cfg.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dg_train_data, train_meta = adapter.build_dgdata(
        abl_cfg, until_timestamp=adapter.dataset.val_timestamp
    )
    _validate_num_negatives_for_eval(num_negatives, train_meta)

    dg_train: DGraph | None = None
    if replay_train_before_eval:
        dg_train = DGraph(dg_train_data, device=device)

    abl_eval = replace(abl_cfg, max_review_edges=None)

    if split == "val":
        dg_eval_data, eval_meta = adapter.build_dgdata(
            abl_eval,
            from_timestamp=adapter.dataset.val_timestamp,
            until_timestamp=adapter.dataset.test_timestamp,
            reuse_node_maps=train_meta,
        )
    elif split == "test":
        dg_eval_data, eval_meta = adapter.build_dgdata(
            abl_eval,
            from_timestamp=adapter.dataset.test_timestamp,
            until_timestamp=None,
            reuse_node_maps=train_meta,
        )
    else:
        raise ValueError(f"split must be 'val' or 'test', got '{split}'")

    dg_eval = DGraph(dg_eval_data, device=device)

    static_node_x = dg_eval_data.static_node_x
    raw_dim = raw_msg_dim_from_config(abl_cfg)

    eval_loader = make_train_loader(dg_eval, train_cfg.batch_size, hook_manager=None)

    memory_snapshot = copy.deepcopy(memory.state_dict())

    print(
        f"  [{label}] evaluating on {split} split "
        f"({eval_meta.num_edges} edges, {num_negatives} negatives/query; "
        f"eval data uncapped by max_review_edges)"
    )
    if replay_train_before_eval:
        hm_train = HookManager(keys=['train'])
        replay_gen = torch.Generator(device=device)
        replay_gen.manual_seed(train_cfg.seed)
        hm_train.register(
            'train',
            BipartiteProductNegativeHook(
                train_meta.num_customers,
                train_meta.num_customers + train_meta.num_products,
                generator=replay_gen,
            ),
        )
        assert dg_train is not None
        train_loader = make_train_loader(dg_train, train_cfg.batch_size, hook_manager=hm_train)
        print(
            f"  [{label}] replaying train stream in no_grad before val MRR "
            f"({train_cfg.epochs} epoch(s) per TrainingConfig)..."
        )
        replay_train_loader_for_memory(
            memory,
            gnn,
            link_pred,
            static_proj,
            train_loader,
            dg_train,
            device,
            dg_train_data.static_node_x,
            raw_dim,
            hm_train,
            num_replay_epochs=train_cfg.epochs,
        )

    assoc_buf = torch.empty(train_meta.num_nodes, dtype=torch.long, device=device)

    metrics = eval_mrr(
        memory=memory,
        gnn=gnn,
        link_pred=link_pred,
        static_proj=static_proj,
        loader=eval_loader,
        device=device,
        static_node_x=static_node_x,
        raw_msg_dim=raw_dim,
        train_meta=train_meta,
        num_negatives=num_negatives,
        seed=train_cfg.seed,
        assoc_buf=assoc_buf,
    )
    
    if logger is not None:
        logger.log_eval(
            split=split,
            metrics=metrics,
            num_negatives=num_negatives,
        )

    memory.load_state_dict(memory_snapshot)
    _restore_all_train_mode(memory, gnn, link_pred, static_proj)

    skipped = int(metrics.get("n_skipped_no_negative_pool", 0))
    skipped_full = int(metrics.get("n_skipped_would_materialize_full_catalog", 0))
    nq = int(metrics.get("n_queries", 0))
    if skipped or skipped_full:
        extra = []
        if skipped:
            extra.append(f"{skipped} no-pool")
        if skipped_full:
            extra.append(f"{skipped_full} would need full catalog")
        print(
            f"  [{label}] {split}  MRR={metrics['mrr']:.4f}  "
            f"(MRR queries={nq}, skipped: {', '.join(extra)})"
        )
    else:
        print(f"  [{label}] {split}  MRR={metrics['mrr']:.4f}  (MRR queries={nq})")

    return metrics
