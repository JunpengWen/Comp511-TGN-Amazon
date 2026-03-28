#!/usr/bin/env python3
"""
Training smoke test: tiny graph, 2 epochs, compare TGN+LastAggregator vs TGN+MeanAggregator.

Success: both runs finish; mean batch losses print (may differ between aggregators).

  python scripts/run_training_smoke.py
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tgn_amazon.adapter import RelbenchAmazonAdapter
from tgn_amazon.config import AblationConfig, TrainingConfig
from tgn_amazon.training import run_training_job


def main() -> None:
    abl = AblationConfig(max_review_edges=1000, use_features=True)
    tc = TrainingConfig(epochs=2, batch_size=128, learning_rate=1e-4)

    adapter = RelbenchAmazonAdapter()
    print('Loading RelBench (cached after first download)...')
    adapter.load(download=True)

    print('\n=== Baseline: LastAggregator (default TGN memory aggregation) ===')
    run_training_job(
        adapter,
        abl,
        tc,
        use_last_aggregator=True,
        label='LastAgg',
    )

    print('\n=== Ablation RQ4: MeanAggregator (simpler memory aggregation) ===')
    run_training_job(
        adapter,
        abl,
        tc,
        use_last_aggregator=False,
        label='MeanAgg',
    )

    print('\nSmoke finished.')


if __name__ == '__main__':
    main()
