#!/usr/bin/env python3
"""
TGN training on RelBench Amazon (TGM): TGNMemory + GraphAttentionEmbedding + LinkPredictor.

Uses bipartite negative sampling (random product nodes) and BCE link loss.
RQ4: pass --mean-agg to use MeanAggregator instead of LastAggregator inside TGNMemory.

Example:
  python scripts/train_tgn_baseline.py --max-edges 50000 --epochs 1

Checkpoint (default save under checkpoints/ after training):
  python scripts/train_tgn_baseline.py --epochs 3 --checkpoint-dir checkpoints

Eval only from checkpoint (skip training):
  python scripts/train_tgn_baseline.py --load-checkpoint checkpoints/20260408_142533_full_lastagg.pt --split val
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import replace
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch

from tgn_amazon.adapter import RelbenchAmazonAdapter
from tgn_amazon.checkpointing import (
    configs_from_checkpoint,
    load_training_checkpoint_dict,
    save_training_checkpoint,
)
from tgn_amazon.config import AblationConfig, TrainingConfig
from tgn_amazon.evaluation import run_eval_job
from tgn_amazon.RunLogger import RunLogger
from tgn_amazon.tgn_model import build_tgn_stack
from tgn_amazon.training import run_training_job


def _restore_modules_from_checkpoint(
    ckpt: dict,
    device: torch.device,
) -> tuple:
    """Return (memory, gnn, link_pred, static_proj, abl, tc)."""
    abl, tc = configs_from_checkpoint(ckpt)
    n = int(ckpt['num_nodes'])
    if ckpt.get('format_version', 1) != 1:
        raise ValueError(f"Unsupported checkpoint format_version: {ckpt.get('format_version')}")

    memory, gnn, link_pred, static_proj = build_tgn_stack(
        n,
        int(ckpt['raw_dim']),
        tc.memory_dim,
        tc.time_dim,
        tc.embedding_dim,
        int(ckpt['static_dim']),
        use_last_aggregator=bool(ckpt['use_last_aggregator']),
        device=device,
    )
    memory.load_state_dict(ckpt['memory'])
    gnn.load_state_dict(ckpt['gnn'])
    link_pred.load_state_dict(ckpt['link_pred'])
    sp_sd = ckpt.get('static_proj')
    if sp_sd is not None:
        if static_proj is None:
            raise ValueError('Checkpoint has static_proj weights but static_dim implies no projection layer')
        static_proj.load_state_dict(sp_sd)
    elif static_proj is not None:
        raise ValueError('Checkpoint has no static_proj but build_tgn_stack created one')

    return memory, gnn, link_pred, static_proj, abl, tc


def _parse_recall_ks(s: str | None) -> list[int] | None:
    if s is None or not str(s).strip():
        return None
    out: list[int] = []
    for part in str(s).replace(' ', '').split(','):
        if not part:
            continue
        out.append(int(part))
    return out or None


def main() -> None:
    p = argparse.ArgumentParser(description='Train TGN baseline (TGM) on RelBench Amazon')
    p.add_argument('--max-edges', type=int, default=None, help='Cap reviews (default: full train split)')
    p.add_argument('--epochs', type=int, default=1)
    p.add_argument('--batch-size', type=int, default=512)
    p.add_argument('--lr', type=float, default=1e-4)
    p.add_argument(
        '--seed',
        type=int,
        default=None,
        metavar='S',
        help=(
            'RNG seed for torch / training negatives / early-stop val hook / MRR sampling '
            '(default when omitted: 42). With --load-checkpoint, overrides the checkpoint seed for eval only.'
        ),
    )
    p.add_argument('--mean-agg', action='store_true', help='Use MeanAggregator instead of LastAggregator (RQ4)')
    p.add_argument(
        '--static',
        action='store_true',
        help='Static graph ablation: constant edge times (no wall-clock; adapter uses zeros, time_delta r)',
    )
    p.add_argument('--homo', action='store_true', help='Homogeneous graph (ablation)')
    p.add_argument(
        '--no-feat',
        action='store_true',
        help='No-features ablation: omit edge_x and static_node_x (no static_proj)',
    )
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
    p.add_argument(
        '--checkpoint-dir',
        type=str,
        default='checkpoints',
        help='Directory to save model.pt after training (set empty string or use --no-save-checkpoint to skip)',
    )
    p.add_argument(
        '--no-save-checkpoint',
        action='store_true',
        help='Do not write a checkpoint file after training',
    )
    p.add_argument(
        '--load-checkpoint',
        type=str,
        default=None,
        help='Path to .pt from a prior run: skip training and run eval only (uses ablation/config stored in file)',
    )
    p.add_argument(
        '--early-stop-patience',
        type=int,
        default=None,
        metavar='N',
        help=(
            'Early stopping: stop training if val BCE does not improve for N epochs '
            '(monitors rel-amazon val window; restores best weights). Omit to disable.'
        ),
    )
    p.add_argument(
        '--early-stop-min-delta',
        type=float,
        default=0.0,
        help='Minimum val loss decrease to count as improvement (default: 0)',
    )
    p.add_argument(
        '--early-stop-val-max-edges',
        type=int,
        default=None,
        metavar='M',
        help='Cap val edges for early-stop monitoring only (default: full val split)',
    )
    p.add_argument(
        '--recall-ks',
        type=str,
        default=None,
        metavar='K1,K2,...',
        help=(
            'Comma-separated K for Recall@K (e.g. 10,50,100). Same random negatives as MRR '
            '(tie-aware rank). Omit to skip Recall@K.'
        ),
    )
    p.add_argument(
        '--eval-max-edges',
        type=int,
        default=None,
        metavar='N',
        help=(
            'Cap val/test eval stream to the first N edges after filters (faster smoke tests). '
            'Omit for full split (recommended for reported metrics).'
        ),
    )
    args = p.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if args.load_checkpoint:
        ckpt_path = Path(args.load_checkpoint)
        if not ckpt_path.is_file():
            print(f'Error: checkpoint not found: {ckpt_path}', file=sys.stderr)
            sys.exit(1)
        ckpt = load_training_checkpoint_dict(ckpt_path, map_location=device)
        memory, gnn, link_pred, static_proj, abl, tc = _restore_modules_from_checkpoint(ckpt, device)
        if args.seed is not None:
            tc = replace(tc, seed=args.seed)
        use_mean = not bool(ckpt['use_last_aggregator'])
        label = 'TGN+MeanAgg' if use_mean else 'TGN+LastAgg'
        logger = RunLogger(log_dir='logs', label=label, config_slug=abl.slug())
        print(f'Run ID (eval-only): {logger.run_id}')
        print(f'Loaded checkpoint: {ckpt_path}')
        if ckpt.get('run_id'):
            print(f'  (trained run_id was {ckpt["run_id"]})')

        adapter = RelbenchAmazonAdapter()
        print('Loading RelBench rel-amazon...')
        adapter.load(download=True)

        dg_train, train_meta = adapter.build_dgdata(
            abl, until_timestamp=adapter.dataset.val_timestamp
        )
        if train_meta.num_nodes != ckpt['num_nodes']:
            raise ValueError(
                f'Checkpoint num_nodes={ckpt["num_nodes"]} != rebuilt train graph {train_meta.num_nodes}. '
                'Data or ablation mismatch.'
            )
    else:
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
            early_stop_patience=args.early_stop_patience,
            early_stop_min_delta=args.early_stop_min_delta,
            early_stop_val_max_edges=args.early_stop_val_max_edges,
            **({} if args.seed is None else {'seed': args.seed}),
        )
        label = 'TGN+MeanAgg' if args.mean_agg else 'TGN+LastAgg'
        logger = RunLogger(
            log_dir='logs',
            label=label,
            config_slug=abl.slug(),
        )
        print(f'Run ID: {logger.run_id}')

        adapter = RelbenchAmazonAdapter()
        print('Loading RelBench rel-amazon (first run may take a long time)...')
        adapter.load(download=True)
        print(f'Ablation: {abl.slug()}  |  training: {tc}  |  {label}')
        _, memory, gnn, link_pred, static_proj, meta, raw_dim, static_dim = run_training_job(
            adapter,
            abl,
            tc,
            use_last_aggregator=not args.mean_agg,
            label=label,
            logger=logger,
        )

        save_ckpt = not args.no_save_checkpoint and bool(args.checkpoint_dir)
        if save_ckpt:
            out_dir = Path(args.checkpoint_dir)
            agg_tag = 'meanagg' if args.mean_agg else 'lastagg'
            fname = f'{logger.run_id}_{abl.slug()}_{agg_tag}.pt'
            out_path = out_dir / fname
            save_training_checkpoint(
                out_path,
                memory=memory,
                gnn=gnn,
                link_pred=link_pred,
                static_proj=static_proj,
                num_nodes=meta.num_nodes,
                raw_dim=raw_dim,
                static_dim=static_dim,
                use_last_aggregator=not args.mean_agg,
                abl=abl,
                tc=tc,
                run_id=logger.run_id,
            )
            print(f'Saved checkpoint: {out_path.resolve()}')

    cached_train_meta = None
    if not args.replay_train_eval:
        if args.load_checkpoint:
            cached_train_meta = train_meta
        else:
            cached_train_meta = meta

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
        recall_ks=_parse_recall_ks(args.recall_ks),
        cached_train_meta=cached_train_meta,
        eval_max_edges=args.eval_max_edges,
    )

    print('Done.')


if __name__ == '__main__':
    main()
