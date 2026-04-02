"""
RelBench Amazon → TGM DGData.

Builds a bipartite temporal graph: customer nodes [0, n_c), product nodes [n_c, n_c + n_p).
Maps to proposal ablations via AblationConfig.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
import torch
from relbench.base import Database
from relbench.datasets import get_dataset

from tgm.data import DGData

from tgn_amazon.config import AblationConfig


@dataclass
class AdapterMetadata:
    num_customers: int
    num_products: int
    num_nodes: int
    num_edges: int
    customer_id_to_idx: dict[Any, int]
    product_id_to_idx: dict[Any, int]
    val_timestamp: pd.Timestamp
    test_timestamp: pd.Timestamp


class RelbenchAmazonAdapter:
    """Load rel-amazon and produce TGM-compatible DGData."""

    def __init__(self, dataset_name: str = "rel-amazon") -> None:
        self.dataset_name = dataset_name
        self._dataset = None
        self._db: Database | None = None

    def load(self, download: bool = True) -> Database:
        self._dataset = get_dataset(self.dataset_name, download=download)
        self._db = self._dataset.get_db()
        return self._db

    @property
    def db(self) -> Database:
        if self._db is None:
            raise RuntimeError("Call load() first.")
        return self._db

    @property
    def dataset(self):
        if self._dataset is None:
            raise RuntimeError("Call load() first.")
        return self._dataset

    def build_dgdata(
        self,
        cfg: AblationConfig,
        *,
        from_timestamp: pd.Timestamp | None = None,
        until_timestamp: pd.Timestamp | None = None,
        reuse_node_maps: AdapterMetadata | None = None,
    ) -> tuple[DGData, AdapterMetadata]:
        """
        until_timestamp: if set, only include reviews strictly before this time
        (e.g. train on edges before validation time).
        from_timestamp: if set, only include reviews at or after this time
        (inclusive lower bound; aligns val with ``[val_timestamp, test_timestamp)``).
        reuse_node_maps: if set, use this metadata's customer/product id maps and
        node counts so indices match a prior build (required for eval on val/test
        windows with a model trained on the train graph).
        """
        if not cfg.use_memory:
            raise ValueError(
                'AblationConfig.use_memory=False is not implemented: TGNMemory is '
                'always used. Use --mean-agg / aggregator ablations or add a no-memory baseline.'
            )
        db = self.db
        review = db.table_dict["review"].df.copy()
        review = review.sort_values("review_time").reset_index(drop=True)
        # Static ablation uses event order instead of wall-clock. Timestamps must stay
        # on one global axis across train / val / test: per-split torch.arange(len) restarts
        # at 0, so val edges look "before" train in TGN (LastAggregator, last_update max,
        # GraphAttentionEmbedding rel_t). That corrupts memory and can spuriously inflate MRR.
        if cfg.static_graph:
            review["_global_event_rank"] = np.arange(len(review), dtype=np.int64)

        if until_timestamp is not None:
            review = review[review["review_time"] < until_timestamp]

        if from_timestamp is not None:
            review = review[review["review_time"] >= from_timestamp]

        if reuse_node_maps is not None:
            c_map = reuse_node_maps.customer_id_to_idx
            p_map = reuse_node_maps.product_id_to_idx
            n_c = reuse_node_maps.num_customers
            n_p = reuse_node_maps.num_products
            ck = list(c_map.keys())
            pk = list(p_map.keys())
            review = review[
                review["customer_id"].isin(ck) & review["product_id"].isin(pk)
            ]
        else:
            customers = sorted(review["customer_id"].unique())
            products = sorted(review["product_id"].unique())
            n_c, n_p = len(customers), len(products)
            c_map = {c: i for i, c in enumerate(customers)}
            p_map = {p: i + n_c for i, p in enumerate(products)}

        if cfg.max_review_edges is not None and len(review) > cfg.max_review_edges:
            review = review.iloc[: cfg.max_review_edges].copy()

        src = review["customer_id"].map(c_map).to_numpy(np.int64)
        dst = review["product_id"].map(p_map).to_numpy(np.int64)
        edge_index = torch.tensor(np.stack([src, dst], axis=1), dtype=torch.long)

        if cfg.static_graph:
            # RQ1: ignore calendar units; keep global interaction order (see _global_event_rank above).
            edge_time = torch.tensor(
                review.pop("_global_event_rank").to_numpy(), dtype=torch.long
            )
            time_delta: str = "r"
        else:
            # Unix seconds (TimeDelta 's'). Second bucketing ties many edges on rel-amazon.
            ts = review["review_time"].astype("int64") // 10**9
            edge_time = torch.tensor(ts.to_numpy(), dtype=torch.long)
            time_delta = "s"

        # Edge features: rating + verified (RQ3)
        edge_x: torch.Tensor | None = None
        static_node_x: torch.Tensor | None = None
        # Width for static tensors; only index 0 is filled (log1p price) for products.
        feat_dim = 8
        if cfg.use_features:
            rating = torch.tensor(review["rating"].fillna(0.0).to_numpy(), dtype=torch.float32)
            ver = review["verified"]
            if ver.dtype == bool or str(ver.dtype) == "boolean":
                verified_t = torch.tensor(ver.fillna(False).astype(np.float32).to_numpy(), dtype=torch.float32)
            else:
                verified_t = torch.tensor(pd.to_numeric(ver, errors="coerce").fillna(0.0).to_numpy(), dtype=torch.float32)
            edge_x = torch.stack([rating, verified_t], dim=1)

            # Static: customers zero; products log-price (from product table)
            static_node_x = torch.zeros((n_c + n_p, feat_dim), dtype=torch.float32)
            prod_df = db.table_dict["product"].df.set_index("product_id")
            for p, nid in p_map.items():
                if p in prod_df.index:
                    price = prod_df.loc[p, "price"]
                    if price is not None and not (isinstance(price, float) and np.isnan(price)):
                        static_node_x[nid, 0] = float(np.log1p(float(price)))
        else:
            static_node_x = torch.zeros((n_c + n_p, feat_dim), dtype=torch.float32)

        node_type: torch.Tensor | None = None
        edge_type: torch.Tensor | None = None
        if not cfg.homogeneous:
            # 0 = customer, 1 = product (RQ2)
            node_type = torch.zeros(n_c + n_p, dtype=torch.long)
            node_type[n_c:] = 1
            edge_type = torch.zeros(len(review), dtype=torch.long)

        dg = DGData.from_raw(
            edge_time=edge_time,
            edge_index=edge_index,
            edge_x=edge_x,
            static_node_x=static_node_x,
            time_delta=time_delta,
            edge_type=edge_type,
            node_type=node_type,
        )

        meta = AdapterMetadata(
            num_customers=n_c,
            num_products=n_p,
            num_nodes=n_c + n_p,
            num_edges=len(review),
            customer_id_to_idx=c_map,
            product_id_to_idx=p_map,
            val_timestamp=self.dataset.val_timestamp,
            test_timestamp=self.dataset.test_timestamp,
        )
        return dg, meta
