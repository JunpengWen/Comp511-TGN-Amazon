"""TGM batch hooks for RelBench Amazon bipartite link prediction."""

from __future__ import annotations

import torch

from tgm import DGBatch, DGraph
from tgm.hooks import StatelessHook


class BipartiteProductNegativeHook(StatelessHook):
    """Sample one negative product id per positive edge (customer → product).

    Destinations are restricted to product node ids [product_lo, product_hi).
    """

    requires = {'edge_src', 'edge_dst', 'edge_time'}
    produces = {'neg', 'neg_time'}

    def __init__(self, product_lo: int, product_hi: int) -> None:
        if product_lo >= product_hi:
            raise ValueError(f'product_lo ({product_lo}) must be < product_hi ({product_hi})')
        self.product_lo = product_lo
        self.product_hi = product_hi

    def __call__(self, dg: DGraph, batch: DGBatch) -> DGBatch:
        e = batch.edge_dst.size(0)
        if e == 0:
            batch.neg = torch.empty(0, dtype=torch.int32, device=dg.device)
            batch.neg_time = torch.empty(0, dtype=torch.int64, device=dg.device)
            return batch
        batch.neg = torch.randint(
            self.product_lo,
            self.product_hi,
            (e,),
            dtype=torch.int32,
            device=dg.device,
        )
        batch.neg_time = batch.edge_time.clone()
        return batch
