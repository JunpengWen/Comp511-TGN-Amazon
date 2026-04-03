from __future__ import annotations

import csv
from datetime import datetime
from pathlib import Path
from typing import Any, Dict


class RunLogger:
    """Logs training and evaluation metrics to CSV files."""

    def __init__(
        self,
        log_dir: str = "logs",
        label: str = "TGN",
        config_slug: str = "full",
        run_id: str | None = None,
    ) -> None:
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.label = label
        self.config_slug = config_slug
        self.run_id = run_id or datetime.now().strftime("%Y%m%d_%H%M%S")

    
    def log_epoch(self, epoch: int, loss: float) -> None:
        """Append one row to training.csv for this epoch."""
        self._append(
            self.log_dir / "training.csv",
            headers=["run_id", "label", "config", "epoch", "mean_loss", "timestamp"],
            row=[
                self.run_id,
                self.label,
                self.config_slug,
                epoch,
                f"{loss:.6f}",
                datetime.now().isoformat(),
            ],
        )

    def log_eval(
        self,
        split: str,
        metrics: Dict[str, Any],
        num_negatives: int,
    ) -> None:
        self._append(
            self.log_dir / "eval.csv",
            headers=[
                "run_id",
                "label",
                "config",
                "split",
                "num_negatives",
                "mrr",
                "n_queries",
                "n_skipped_no_negative_pool",
                "n_skipped_would_materialize_full_catalog",
                "n_skipped_invalid_node_ids",
                "timestamp",
            ],
            row=[
                self.run_id,
                self.label,
                self.config_slug,
                split,
                num_negatives,
                f"{metrics.get('mrr', 0.0):.6f}",
                metrics.get("n_queries", 0),
                metrics.get("n_skipped_no_negative_pool", 0),
                metrics.get("n_skipped_would_materialize_full_catalog", 0),
                metrics.get("n_skipped_invalid_node_ids", 0),
                datetime.now().isoformat(),
            ],
        )

  
    def _append(self, path: Path, headers: list[str], row: list) -> None:
        """Append a row to a CSV, writing headers if the file is new."""
        write_header = not path.exists()
        with open(path, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            if write_header:
                writer.writerow(headers)
            writer.writerow(row)

