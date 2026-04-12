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


def plot_metrics_summary(df: pd.DataFrame, out: Path | None) -> None:
    """
    Horizontal bar chart of MRR + Recall@K for every (run_id, split) row.
    Works with 1 row or many.
    """
    df, ks = _parse_recalls(df)
    recall_cols = [f"recall@{k}" for k in ks]
    metric_cols = ["mrr"] + recall_cols
    metric_labels = ["MRR"] + [f"Recall@{k}" for k in ks]

    for _, row in df.iterrows():
        values = [row[c] for c in metric_cols]
        run_label = f"{row['run_id']}  |  split={row['split']}  |  neg={row['num_negatives']}"

        fig, ax = plt.subplots(figsize=(8, max(3, len(metric_cols) * 0.55 + 1.5)))
        fig.suptitle("Eval Metrics Summary", color="#e8eaf0", fontsize=15, y=1.01)

        bars = ax.barh(metric_labels, values,
                       color=[_color(i) for i in range(len(metric_cols))],
                       alpha=0.85, height=0.55)

        for bar, val in zip(bars, values):
            if not np.isnan(val):
                ax.text(max(val - 0.03, 0.01), bar.get_y() + bar.get_height() / 2,
                        f"{val:.4f}", va="center", ha="right",
                        color="#0f1117", fontsize=9, fontweight="bold")

        ax.set_xlim(0, 1.05)
        ax.set_xlabel("Score")
        ax.set_title(run_label, color="#7a7d8d", fontsize=9, pad=6)
        ax.invert_yaxis()

        safe = row["run_id"].replace(":", "-").replace("/", "-")
        _save_or_show(fig, out, f"02_metrics_summary_{safe}_{row['split']}.png")


def plot_recall_at_k(df: pd.DataFrame, out: Path | None) -> None:
    """
    Recall@K line chart per (run_id, split).
    Works with 1 row — shows how recall accumulates as K grows.
    """
    df, ks = _parse_recalls(df)
    if not ks:
        return

    fig, ax = plt.subplots(figsize=(8, 4.5))
    fig.suptitle("Recall@K", color="#e8eaf0", fontsize=15, y=1.01)

    for i, (_, row) in enumerate(df.iterrows()):
        vals = [row[f"recall@{k}"] for k in ks]
        if all(np.isnan(v) for v in vals):
            continue
        label = f"{row['run_id']} / {row['split']}"
        ax.plot(ks, vals, color=_color(i), marker="o", markersize=6, label=label)
        for k, v in zip(ks, vals):
            if not np.isnan(v):
                ax.annotate(f"{v:.3f}", (k, v),
                            textcoords="offset points", xytext=(0, 8),
                            ha="center", fontsize=8, color=_color(i))

    ax.set_xlabel("K")
    ax.set_ylabel("Recall@K")
    ax.set_ylim(0, 1.12)
    ax.set_xticks(ks)
    ax.legend(loc="lower right")
    _save_or_show(fig, out, "03_recall_at_k.png")


def plot_mrr_vs_recall_tradeoff(df: pd.DataFrame, out: Path | None) -> None:
    """
    Scatter: MRR (x) vs each Recall@K (y).
    With 1 row shows where each K threshold lands relative to MRR.
    """
    df, ks = _parse_recalls(df)
    if not ks:
        return

    fig, ax = plt.subplots(figsize=(7, 5))
    fig.suptitle("MRR vs Recall@K", color="#e8eaf0", fontsize=15, y=1.01)

    for i, k in enumerate(ks):
        col = f"recall@{k}"
        sub = df[df[col].notna()]
        ax.scatter(sub["mrr"], sub[col], color=_color(i),
                   label=f"Recall@{k}", s=80, zorder=3,
                   edgecolors="#0f1117", linewidths=0.5)
        # Annotate each point with run label
        for _, row in sub.iterrows():
            ax.annotate(f"{row['split']}", (row["mrr"], row[col]),
                        textcoords="offset points", xytext=(5, 3),
                        fontsize=7, color=_color(i))

    ax.plot([0, 1], [0, 1], color="#3a3d4d", linewidth=1,
            linestyle="--", label="MRR = Recall")
    ax.set_xlabel("MRR")
    ax.set_ylabel("Recall@K")
    ax.set_xlim(0, 1.05)
    ax.set_ylim(0, 1.08)
    ax.legend()
    _save_or_show(fig, out, "04_mrr_vs_recall.png")


def plot_query_coverage(df: pd.DataFrame, out: Path | None) -> None:
    """
    Stacked bar: evaluated vs each skip reason — one bar per (run_id, split).
    Works with 1 row.
    """
    skip_cols = [
        "n_skipped_no_negative_pool",
        "n_skipped_would_materialize_full_catalog",
        "n_skipped_invalid_node_ids",
    ]
    skip_labels = ["No neg pool", "Full catalog", "Invalid IDs"]
    skip_colors = [_color(2), _color(3), _color(4)]

    df = df.copy()
    df["n_evaluated"] = (df["n_queries"] - df[skip_cols].sum(axis=1)).clip(lower=0)
    df["row_label"] = df["run_id"] + "\n" + df["split"]

    fig, ax = plt.subplots(figsize=(max(6, len(df) * 1.4), 4.5))
    fig.suptitle("Query Coverage (evaluated vs skipped)", color="#e8eaf0", fontsize=15, y=1.01)

    x = np.arange(len(df))
    bottom = np.zeros(len(df))

    ax.bar(x, df["n_evaluated"], color=_color(0), label="Evaluated", alpha=0.85)
    bottom += df["n_evaluated"].values

    for col, lbl, clr in zip(skip_cols, skip_labels, skip_colors):
        ax.bar(x, df[col], bottom=bottom, color=clr, label=lbl, alpha=0.85)
        bottom += df[col].values

    for xi, (ev, tot) in enumerate(zip(df["n_evaluated"].values, df["n_queries"].values)):
        pct = 100 * ev / tot if tot > 0 else 0
        ax.text(xi, tot * 1.01, f"{pct:.1f}%",
                ha="center", va="bottom", fontsize=8, color="#c8ccd8")

    ax.set_xticks(x)
    ax.set_xticklabels(df["row_label"], fontsize=8)
    ax.set_ylabel("Query Count")
    ax.legend(loc="upper right")
    _save_or_show(fig, out, "05_query_coverage.png")


def plot_recall_gap(df: pd.DataFrame, out: Path | None) -> None:
    """
    Bar chart of recall gain per K interval.
    Shows where the model catches up — e.g. most gains K=10->50 vs K=50->100.
    Works with 1 row.
    """
    df, ks = _parse_recalls(df)
    if len(ks) < 2:
        return

    for _, row in df.iterrows():
        vals = [row[f"recall@{k}"] for k in ks]
        if any(np.isnan(v) for v in vals):
            continue

        gaps = [vals[0]] + [vals[i] - vals[i - 1] for i in range(1, len(vals))]
        gap_labels = [f"0->{ks[0]}"] + [f"{ks[i-1]}->{ks[i]}" for i in range(1, len(ks))]

        fig, ax = plt.subplots(figsize=(7, 4))
        run_label = f"{row['run_id']}  |  {row['split']}"
        fig.suptitle(f"Recall Gain per K Interval\n{run_label}",
                     color="#e8eaf0", fontsize=13, y=1.02)

        bars = ax.bar(gap_labels, gaps,
                      color=[_color(i) for i in range(len(gaps))], alpha=0.85)
        for bar, val in zip(bars, gaps):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                    f"+{val:.3f}", ha="center", va="bottom",
                    fontsize=9, color="#c8ccd8")

        ax.set_ylabel("Recall Gain")
        ax.set_ylim(0, max(gaps) * 1.3)
        safe = row["run_id"].replace(":", "-").replace("/", "-")
        _save_or_show(fig, out, f"06_recall_gap_{safe}_{row['split']}.png")



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
def _parse_recalls(df: pd.DataFrame):
    """Expand recalls_json into recall_at_K columns on a copy of df."""
    df = df.copy()
    all_ks: set[int] = set()
    parsed = []
    for v in df["recalls_json"]:
        try:
            d = json.loads(v) if pd.notna(v) and v != "" else {}
        except (json.JSONDecodeError, TypeError):
            d = {}
        parsed.append({int(k): float(val) for k, val in d.items()})
        all_ks.update(int(k) for k in d.keys())

    for k in sorted(all_ks):
        df[f"recall@{k}"] = [p.get(k, np.nan) for p in parsed]
    return df, sorted(all_ks)


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
    parser.add_argument("--log-dir", default="logs")
    parser.add_argument("--out-dir", default="plots")
    parser.add_argument("--show", action="store_true")
    args = parser.parse_args()

    log_dir = Path(args.log_dir)
    out_dir = None if args.show else Path(args.out_dir)

    train_df = load_csv(log_dir / "training.csv")
    eval_df  = load_csv(log_dir / "RQ4_eval.csv")
    early_df = load_csv(log_dir / "early_stop.csv")

    print("Generating plots...")

    if train_df is not None:
        plot_training_loss(train_df, out_dir)

    if eval_df is not None:
        # Always meaningful (work from 1 row)
        plot_metrics_summary(eval_df, out_dir)
        plot_recall_at_k(eval_df, out_dir)
        plot_mrr_vs_recall_tradeoff(eval_df, out_dir)
        plot_query_coverage(eval_df, out_dir)
        plot_recall_gap(eval_df, out_dir)
        # Only meaningful with multiple rows (guarded internally)
        plot_mrr_by_split(eval_df, out_dir)
        plot_negatives_vs_mrr(eval_df, out_dir)
        plot_skipped_breakdown(eval_df, out_dir)

    if early_df is not None:
        plot_early_stop_scatter(early_df, out_dir)
        plot_epochs_completed_vs_best(early_df, out_dir)

    if train_df is not None and eval_df is not None:
        plot_loss_vs_mrr(train_df, eval_df, out_dir)

    print("Done.")


if __name__ == "__main__":
    main()
