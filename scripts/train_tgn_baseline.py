#!/usr/bin/env python3
"""
TGN training on RelBench Amazon (TGM): TGNMemory + GraphAttentionEmbedding + LinkPredictor.

Uses bipartite negative sampling (random product nodes) and BCE link loss.
RQ4: pass --mean-agg to use MeanAggregator instead of LastAggregator inside TGNMemory.

Example:
  python scripts/train_tgn_baseline.py --max-edges 50000 --epochs 1
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tgn_amazon.adapter import RelbenchAmazonAdapter
from tgn_amazon.config import AblationConfig, TrainingConfig
from tgn_amazon.training import run_training_job
from tgn_amazon.evaluation import run_eval_job
from tgn_amazon.RunLogger import RunLogger

def main() -> None:
    p = argparse.ArgumentParser(description='Train TGN baseline (TGM) on RelBench Amazon')
    p.add_argument('--max-edges', type=int, default=None, help='Cap reviews (default: full train split)')
    p.add_argument('--epochs', type=int, default=1)
    p.add_argument('--batch-size', type=int, default=512)
    p.add_argument('--lr', type=float, default=1e-4)
    p.add_argument('--mean-agg', action='store_true', help='Use MeanAggregator instead of LastAggregator (RQ4)')
    p.add_argument('--static', action='store_true', help='Static/event-order time (ablation)')
    p.add_argument('--homo', action='store_true', help='Homogeneous graph (ablation)')
    p.add_argument('--no-feat', action='store_true', help='Strip edge/static features (ablation)')
    p.add_argument('--split', choices=['val', 'test'], default='val')
    p.add_argument(
        '--num-negatives',
        type=int,
        default=99,
        help=(
            'Random product negatives per edge for MRR (1 = one neg, two candidates total). '
            'Must be < num_products - 1 on large catalogs (validated in run_eval_job).'
        ),
    )
    p.add_argument(
        '--replay-train-eval',
        action='store_true',
        help='Before val MRR, replay the capped train stream in no_grad to rebuild memory (slow).',
    )
    args = p.parse_args()

    abl = AblationConfig(
        static_graph=args.static,
        homogeneous=args.homo,
        use_features=not args.no_feat,
        max_review_edges=args.max_edges,
    )
    tc = TrainingConfig(
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
    )
    label = 'TGN+MeanAgg' if args.mean_agg else 'TGN+LastAgg'
    logger = RunLogger(
        log_dir="logs",
        label=label,
        config_slug=abl.slug(),
    )
    print(f"Run ID: {logger.run_id}")

    adapter = RelbenchAmazonAdapter()
    print('Loading RelBench rel-amazon (first run may take a long time)...')
    adapter.load(download=True)
    print(f'Ablation: {abl.slug()}  |  training: {tc}  |  {label}')
    _, memory, gnn, link_pred, static_proj = run_training_job(
        adapter,
        abl,
        tc,
        use_last_aggregator=not args.mean_agg,
        label=label,
        logger=logger,
    )

    num_neg = max(1, args.num_negatives)
    run_eval_job(
        adapter,
        abl,
        tc,
        memory,
        gnn,
        link_pred,
        static_proj,
        split=args.split,
        num_negatives=num_neg,
        label=label,
        replay_train_before_eval=args.replay_train_eval,
        logger=logger,
    )

    print('Done.')


if __name__ == '__main__':
    main()
