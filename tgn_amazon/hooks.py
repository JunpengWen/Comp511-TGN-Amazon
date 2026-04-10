"""TGM batch hooks for RelBench Amazon bipartite link prediction."""

from __future__ import annotations

import torch

from tgm import DGBatch, DGraph
from tgm.hooks import StatelessHook


class BipartiteProductNegativeHook(StatelessHook):
    """Sample one negative product id per positive edge (customer → product).

    Destinations are restricted to product node ids [product_lo, product_hi).
    Uses ``generator`` when given so sampling is reproducible with ``manual_seed``.
    If the product range has only one id and it equals ``dst``, ``neg`` may still
    equal ``dst``; training must mask those rows (see ``train_epoch``).
    """

    requires = {'edge_src', 'edge_dst', 'edge_time'}
    produces = {'neg', 'neg_time'}

    def __init__(
        self,
        product_lo: int,
        product_hi: int,
        *,
        generator: torch.Generator | None = None,
    ) -> None:
        if product_lo >= product_hi:
            raise ValueError(f'product_lo ({product_lo}) must be < product_hi ({product_hi})')
        self.product_lo = product_lo
        self.product_hi = product_hi
        self._generator = generator

    def __call__(self, dg: DGraph, batch: DGBatch) -> DGBatch:
        e = batch.edge_dst.size(0)
        if e == 0:
            batch.neg = torch.empty(0, dtype=torch.int32, device=dg.device)
            batch.neg_time = torch.empty(0, dtype=torch.int64, device=dg.device)
            return batch
        dst = batch.edge_dst.long()
        gen = self._generator
        neg = torch.randint(
            self.product_lo,
            self.product_hi,
            (e,),
            dtype=torch.int32,
            device=dg.device,
            generator=gen,
        )
        for _ in range(32):
            clash = neg.long() == dst
            if not clash.any():
                break
            resampled = torch.randint(
                self.product_lo,
                self.product_hi,
                (e,),
                dtype=torch.int32,
                device=dg.device,
                generator=gen,
            )
            neg = torch.where(clash, resampled, neg)
        clash = neg.long() == dst
        if clash.any():
            # Uniform over valid product ids (not deterministic "smallest id") if resampling failed.
            # O(1) memory: sample an index in [0, num_valid) then map to id (skip dst), same idea as
            # evaluation._indices_to_product_ids.
            lo, hi = self.product_lo, self.product_hi
            for i in torch.where(clash)[0].tolist():
                di = int(dst[i].item())
                in_range = lo <= di < hi
                pool = hi - lo
                max_uniq = pool - (1 if in_range else 0)
                if max_uniq <= 0:
                    continue
                u = torch.randint(
                    0,
                    max_uniq,
                    (1,),
                    device=dg.device,
                    dtype=torch.long,
                    generator=gen,
                ).item()
                if not in_range:
                    picked = lo + u
                else:
                    off = di - lo
                    picked = lo + u if u < off else lo + u + 1
                neg[i] = int(picked)
        batch.neg = neg
        batch.neg_time = batch.edge_time.clone()
        return batch
