#!/usr/bin/env python3
"""
Smoke test: RelBench Amazon → DGData (TGM).

Uses a small edge cap by default so you can iterate without waiting for full DB builds.
For full data: omit --max-edges (requires rel-amazon downloaded; see README.md).

Usage:
  cd c:\\Comp511\\Project
  python scripts/run_adapter_smoke.py --max-edges 50000
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Allow running without installing the package
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tgm import DGraph
from tgm.data import DGDataLoader

from tgn_amazon.adapter import RelbenchAmazonAdapter
from tgn_amazon.config import AblationConfig


def main() -> None:
    p = argparse.ArgumentParser(description="Build DGData from RelBench Amazon")
    p.add_argument("--max-edges", type=int, default=50_000, help="Cap reviews for quick tests")
    p.add_argument("--full", action="store_true", help="Use all edges (no cap)")
    p.add_argument("--static", action="store_true", help="Ablation: static/event-order time")
    p.add_argument("--homo", action="store_true", help="Ablation: homogeneous (no type tensors)")
    p.add_argument("--no-feat", action="store_true", help="Ablation: strip features")
    args = p.parse_args()

    cfg = AblationConfig(
        static_graph=args.static,
        homogeneous=args.homo,
        use_features=not args.no_feat,
        max_review_edges=None if args.full else args.max_edges,
    )
    print(f"Ablation config: {cfg.slug()}")

    adapter = RelbenchAmazonAdapter()
    print("Loading RelBench (first time can download / process a long time)...")
    adapter.load(download=True)

    print("Building DGData (train-only edges before val timestamp)...")
    dg, meta = adapter.build_dgdata(
        cfg, until_timestamp=adapter.dataset.val_timestamp
    )

    print(f"  Nodes: {meta.num_nodes} (customers={meta.num_customers}, products={meta.num_products})")
    print(f"  Edges: {meta.num_edges}")
    print(f"  Val time: {meta.val_timestamp}, Test time: {meta.test_timestamp}")
    print(dg)

    g = DGraph(dg)
    # Event-ordered mini-batches work for both 'r' and 's' underlying time_delta
    bs = min(5000, max(1, meta.num_edges))
    try:
        loader = DGDataLoader(g, batch_unit="r", batch_size=bs)
        batch = next(iter(loader))
        print("First batch OK:", batch.edge_src.shape, batch.edge_dst.shape)
    except Exception as e:
        print("Loader smoke step failed:", e)


if __name__ == "__main__":
    main()
