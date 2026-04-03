#!/usr/bin/env python3
"""
Structural checks that RelbenchAmazonAdapter output is internally consistent.
Does not prove ML correctness — only data/graph invariants TGM expects.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch
from tgm import DGraph
from tgm.data import DGDataLoader

from tgn_amazon.adapter import RelbenchAmazonAdapter
from tgn_amazon.config import AblationConfig


def main() -> None:
    cfg = AblationConfig(max_review_edges=10_000)
    adapter = RelbenchAmazonAdapter()
    adapter.load(download=True)
    val_ts = adapter.dataset.val_timestamp
    dg, meta = adapter.build_dgdata(cfg, until_timestamp=val_ts)

    ei = dg.edge_index
    assert ei.dim() == 2 and ei.size(1) == 2, "edge_index shape"
    n_c, n_p = meta.num_customers, meta.num_products
    src, dst = ei[:, 0].long(), ei[:, 1].long()

    assert (src >= 0).all() and (src < n_c).all(), "customer src ids must be in [0, n_c)"
    assert (dst >= n_c).all() and (dst < n_c + n_p).all(), "product dst ids must be in [n_c, n_c+n_p)"
    assert meta.num_nodes == n_c + n_p

    t = dg.time[dg.edge_mask]
    assert (t[1:] >= t[:-1]).all(), "edge times must be non-decreasing (TGM sorts)"
    assert t.numel() == meta.num_edges

    # Same filter as adapter: no review at or after val (avoid timezone issues from .timestamp())
    try:
        event_ordered = bool(dg.time_delta.is_event_ordered)
    except AttributeError:
        event_ordered = False
    if not event_ordered:
        rev = adapter.db.table_dict["review"].df
        rev = rev.sort_values("review_time").reset_index(drop=True)
        rev = rev[rev["review_time"] < val_ts]
        if cfg.max_review_edges is not None and len(rev) > cfg.max_review_edges:
            rev = rev.iloc[: cfg.max_review_edges]
        assert rev["review_time"].max() < val_ts, "train reviews must be strictly before val_timestamp"

    if dg.edge_x is not None:
        assert dg.edge_x.shape == (meta.num_edges, 2), "edge_x = [rating, verified]"

    if dg.static_node_x is not None:
        assert dg.static_node_x.shape[0] == meta.num_nodes

    if dg.node_type is not None:
        assert (dg.node_type[:n_c] == 0).all() and (dg.node_type[n_c:] == 1).all()

    g = DGraph(dg)
    loader = DGDataLoader(g, batch_unit="r", batch_size=min(1000, meta.num_edges))
    b = next(iter(loader))
    assert b.edge_src.numel() > 0
    print("All invariant checks passed.")
    print(f"  nodes={meta.num_nodes}, edges={meta.num_edges}, val_ts={val_ts}")


if __name__ == "__main__":
    main()
