from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd

"""
Usage:
    python plot_run_logs.py                     # reads from ./logs/, saves PNGs to ./plots/
    python plot_run_logs.py --log-dir my/logs   # custom log directory
    python plot_run_logs.py --show              # display interactively instead of saving
"""

# 8 plots: Training Loss, MRR by Split, num_negatives vs MRR, 
# Skipped breakdown, Early stop scatter, Epochs completed vs best, Loss vs MRR, Recall@K curves

plt.rcParams.update(
    {
        "figure.facecolor": "#0f1117",
        "axes.facecolor": "#1a1d27",
        "axes.edgecolor": "#3a3d4d",
        "axes.labelcolor": "#c8ccd8",
        "axes.titlecolor": "#e8eaf0",
        "axes.titlesize": 13,
        "axes.labelsize": 11,
        "axes.grid": True,
        "grid.color": "#2a2d3d",
        "grid.linewidth": 0.6,
        "xtick.color": "#7a7d8d",
        "ytick.color": "#7a7d8d",
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.facecolor": "#1a1d27",
        "legend.edgecolor": "#3a3d4d",
        "legend.labelcolor": "#c8ccd8",
        "legend.fontsize": 9,
        "lines.linewidth": 2,
        "figure.dpi": 130,
        "savefig.facecolor": "#0f1117",
        "savefig.bbox": "tight",
    }
)

PALETTE = [
    "#7eb8f7", "#f7a07e", "#7ef7a0", "#f7e07e",
    "#c07ef7", "#f77eb8", "#7ef7f0", "#f7c07e",
]


def _color(i: int) -> str:
    return PALETTE[i % len(PALETTE)]


def load_csv(path: Path) -> pd.DataFrame | None:
    if not path.exists():
        warnings.warn(f"File not found, skipping: {path}", UserWarning)
        return None
    df = pd.read_csv(path)
    if df.empty:
        warnings.warn(f"File is empty, skipping: {path}", UserWarning)
        return None
    return df


# 1. Training loss curve  (training.csv)

def plot_training_loss(df: pd.DataFrame, out: Path | None) -> None:
    fig, ax = plt.subplots(figsize=(9, 4.5))
    fig.suptitle("Training Loss Curve", color="#e8eaf0", fontsize=15, y=1.01)

    groups = df.groupby(["run_id", "config"])
    for i, ((run_id, cfg), g) in enumerate(groups):
        g = g.sort_values("epoch")
        label = f"{run_id} [{cfg}]"
        ax.plot(g["epoch"], g["mean_loss"], color=_color(i), label=label, marker="o", markersize=3)

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Mean Loss")
    ax.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))
    ax.legend(loc="upper right")
    _save_or_show(fig, out, "01_training_loss.png")


# 2. MRR over splits  (eval.csv)

def plot_mrr_by_split(df: pd.DataFrame, out: Path | None) -> None:
    fig, ax = plt.subplots(figsize=(9, 4.5))
    fig.suptitle("MRR by Evaluation Split", color="#e8eaf0", fontsize=15, y=1.01)

    splits = df["split"].unique()
    configs = df["config"].unique()

    for i, (split, cfg) in enumerate(
        [(s, c) for s in splits for c in configs]
    ):
        sub = df[(df["split"] == split) & (df["config"] == cfg)].sort_values("run_id")
        if sub.empty:
            continue
        ax.bar(
            np.arange(len(sub)) + i * 0.25,
            sub["mrr"],
            width=0.22,
            color=_color(i),
            label=f"{split} / {cfg}",
            alpha=0.85,
        )

    ax.set_xlabel("Run")
    ax.set_ylabel("MRR")
    ax.set_ylim(0, 1)
    ax.legend(loc="upper right")
    _save_or_show(fig, out, "02_mrr_by_split.png")


# 3. num_negatives vs MRR  (eval.csv)

def plot_negatives_vs_mrr(df: pd.DataFrame, out: Path | None) -> None:
    fig, ax = plt.subplots(figsize=(9, 4.5))
    fig.suptitle("num_negatives vs MRR", color="#e8eaf0", fontsize=15, y=1.01)

    splits = df["split"].unique()
    for i, split in enumerate(splits):
        sub = df[df["split"] == split].groupby("num_negatives")["mrr"].mean().reset_index()
        sub = sub.sort_values("num_negatives")
        ax.plot(
            sub["num_negatives"], sub["mrr"],
            color=_color(i), label=split, marker="o", markersize=5,
        )

    ax.set_xlabel("num_negatives")
    ax.set_ylabel("Mean MRR (across runs)")
    ax.set_ylim(0, 1)
    ax.legend(loc="upper right")
    _save_or_show(fig, out, "03_negatives_vs_mrr.png")


# 4. Skipped queries breakdown  (eval.csv)

def plot_skipped_breakdown(df: pd.DataFrame, out: Path | None) -> None:
    skip_cols = [
        "n_skipped_no_negative_pool",
        "n_skipped_would_materialize_full_catalog",
        "n_skipped_invalid_node_ids",
    ]
    short_labels = ["No neg pool", "Full catalog", "Invalid IDs"]

    # Aggregate per run
    agg = df.groupby("run_id")[skip_cols].sum()
    if agg.empty:
        return

    x = np.arange(len(agg))
    width = 0.25
    fig, ax = plt.subplots(figsize=(max(7, len(agg) * 1.2), 4.5))
    fig.suptitle("Skipped Queries Breakdown per Run", color="#e8eaf0", fontsize=15, y=1.01)

    for j, (col, lbl) in enumerate(zip(skip_cols, short_labels)):
        ax.bar(x + j * width, agg[col], width=width, color=_color(j), label=lbl, alpha=0.85)

    ax.set_xticks(x + width)
    ax.set_xticklabels(agg.index, rotation=20, ha="right")
    ax.set_ylabel("Count")
    ax.legend()
    _save_or_show(fig, out, "04_skipped_breakdown.png")


# 5. Early stop: best_epoch vs best_val_loss scatter  (early_stop.csv)

def plot_early_stop_scatter(df: pd.DataFrame, out: Path | None) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    fig.suptitle("Early Stop: Best Epoch vs Best Val Loss", color="#e8eaf0", fontsize=15, y=1.01)

    stopped = df[df["stopped_early"] == 1]
    ran_full = df[df["stopped_early"] == 0]

    ax.scatter(
        stopped["best_epoch"], stopped["best_val_loss"],
        color=_color(0), label="Stopped early", s=70, zorder=3, edgecolors="#0f1117", linewidths=0.5,
    )
    ax.scatter(
        ran_full["best_epoch"], ran_full["best_val_loss"],
        color=_color(1), label="Ran full", marker="^", s=70, zorder=3, edgecolors="#0f1117", linewidths=0.5,
    )

    for _, row in df.iterrows():
        ax.annotate(
            row["run_id"],
            (row["best_epoch"], row["best_val_loss"]),
            textcoords="offset points", xytext=(5, 4),
            fontsize=7, color="#7a7d8d",
        )

    ax.set_xlabel("Best Epoch")
    ax.set_ylabel("Best Val Loss")
    ax.legend()
    _save_or_show(fig, out, "05_early_stop_scatter.png")


# 6. epochs_completed vs best_epoch  (early_stop.csv)

def plot_epochs_completed_vs_best(df: pd.DataFrame, out: Path | None) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    fig.suptitle("Epochs Completed vs Best Epoch", color="#e8eaf0", fontsize=15, y=1.01)

    configs = df["config"].unique()
    for i, cfg in enumerate(configs):
        sub = df[df["config"] == cfg]
        ax.scatter(sub["best_epoch"], sub["epochs_completed"], color=_color(i), label=cfg, s=70, zorder=3)

    # Diagonal reference line
    lim = max(df["epochs_completed"].max(), df["best_epoch"].max()) * 1.05
    ax.plot([0, lim], [0, lim], color="#3a3d4d", linewidth=1, linestyle="--", label="best = completed")

    ax.set_xlabel("Best Epoch")
    ax.set_ylabel("Epochs Completed")
    ax.legend()
    _save_or_show(fig, out, "06_epochs_completed_vs_best.png")


# 7. Cross-file: training loss + val MRR on dual y-axis

def plot_loss_vs_mrr(train_df: pd.DataFrame, eval_df: pd.DataFrame, out: Path | None) -> None:
    val_df = eval_df[eval_df["split"] == "val"]
    if val_df.empty:
        warnings.warn("No 'val' split rows in eval.csv — skipping cross-file plot.", UserWarning)
        return

    run_ids = train_df["run_id"].unique()
    for run_id in run_ids:
        t = train_df[train_df["run_id"] == run_id].sort_values("epoch")
        v = val_df[val_df["run_id"] == run_id].sort_values("run_id")

        if t.empty:
            continue

        fig, ax1 = plt.subplots(figsize=(9, 4.5))
        fig.suptitle(f"Loss vs Val MRR — {run_id}", color="#e8eaf0", fontsize=15, y=1.01)

        ax1.plot(t["epoch"], t["mean_loss"], color=_color(0), label="Train Loss")
        ax1.set_xlabel("Epoch")
        ax1.set_ylabel("Mean Loss", color=_color(0))
        ax1.tick_params(axis="y", labelcolor=_color(0))

        if not v.empty and len(v) > 1:
            ax2 = ax1.twinx()
            ax2.plot(
                np.linspace(t["epoch"].min(), t["epoch"].max(), len(v)),
                v["mrr"].values,
                color=_color(1), linestyle="--", label="Val MRR",
            )
            ax2.set_ylabel("Val MRR", color=_color(1))
            ax2.tick_params(axis="y", labelcolor=_color(1))
            ax2.set_ylim(0, 1)
            ax2.spines["right"].set_edgecolor(_color(1))

        lines1, labels1 = ax1.get_legend_handles_labels()
        try:
            lines2, labels2 = ax2.get_legend_handles_labels()
        except NameError:
            lines2, labels2 = [], []
        ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper right")

        safe_id = run_id.replace(":", "-").replace("/", "-")
        _save_or_show(fig, out, f"07_loss_vs_mrr_{safe_id}.png")


# 8. Recall@K curves  (eval.csv, if recalls_json populated)

def plot_recall_at_k(df: pd.DataFrame, out: Path | None) -> None:
    has_recalls = df["recalls_json"].notna() & (df["recalls_json"] != "")
    if not has_recalls.any():
        return

    fig, ax = plt.subplots(figsize=(9, 4.5))
    fig.suptitle("Recall@K by Split / Run", color="#e8eaf0", fontsize=15, y=1.01)

    i = 0
    for _, row in df[has_recalls].iterrows():
        try:
            recalls = json.loads(row["recalls_json"])
        except (json.JSONDecodeError, TypeError):
            continue
        ks = sorted(int(k) for k in recalls)
        vals = [recalls[str(k)] for k in ks]
        label = f"{row['run_id']} / {row['split']}"
        ax.plot(ks, vals, color=_color(i), marker="o", markersize=5, label=label)
        i += 1

    ax.set_xlabel("K")
    ax.set_ylabel("Recall@K")
    ax.set_ylim(0, 1)
    ax.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))
    ax.legend(loc="lower right")
    _save_or_show(fig, out, "08_recall_at_k.png")


# helper functions

def _save_or_show(fig: plt.Figure, out: Path | None, filename: str) -> None:
    if out is None:
        plt.show()
    else:
        out.mkdir(parents=True, exist_ok=True)
        path = out / filename
        fig.savefig(path)
        print(f"  Saved → {path}")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot RunLogger CSV outputs.")
    parser.add_argument("--log-dir", default="logs", help="Directory containing the CSV files")
    parser.add_argument("--out-dir", default="plots", help="Directory to write PNG files")
    parser.add_argument("--show", action="store_true", help="Display plots interactively instead of saving")
    args = parser.parse_args()

    log_dir = Path(args.log_dir)
    out_dir = None if args.show else Path(args.out_dir)

    train_df = load_csv(log_dir / "training.csv")
    eval_df = load_csv(log_dir / "RQ4_eval.csv")
    early_df = load_csv(log_dir / "early_stop.csv")

    print("Generating plots…")

    if train_df is not None:
        plot_training_loss(train_df, out_dir)

    if eval_df is not None:
        plot_mrr_by_split(eval_df, out_dir)
        plot_negatives_vs_mrr(eval_df, out_dir)
        plot_skipped_breakdown(eval_df, out_dir)
        plot_recall_at_k(eval_df, out_dir)

    if early_df is not None:
        plot_early_stop_scatter(early_df, out_dir)
        plot_epochs_completed_vs_best(early_df, out_dir)

    if train_df is not None and eval_df is not None:
        plot_loss_vs_mrr(train_df, eval_df, out_dir)

    print("Done.")


if __name__ == "__main__":
    main()