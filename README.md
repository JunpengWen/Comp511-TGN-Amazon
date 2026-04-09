# Comp 511 — TGN ablation on RelBench Amazon

McGill **Network Science** course project (see **`project_proposal.tex`**). Goal: run **Temporal Graph Networks (TGN)** via the **Temporal Graph Modelling (TGM)** library on **RelBench `rel-amazon`**, then ablate **time**, **heterogeneity**, **features**, and **memory** and compare ranking metrics.

**Authors:** Harry Hu, Junpeng Wen, Sophia Kyrychenko.

---

## What is done

| Area | Description |
|------|-------------|
| **Dependencies** | `requirements.txt` pins `tgm-lib`, `relbench`, PyTorch / PyG, etc. |
| **RelBench → TGM** | **`tgn_amazon/adapter.py`**: loads `rel-amazon`, train reviews **before** `val_timestamp`; val/test windows use **`review_time >= from_timestamp`** and **`< until_timestamp`** where applicable; **`reuse_node_maps`** for aligned ids; bipartite graph (customers ↔ products) as **`DGData`**. |
| **Configs** | **`tgn_amazon/config.py`**: **`AblationConfig`** (`static_graph`, `homogeneous`, `use_features`, `use_memory`, `max_review_edges`) and **`TrainingConfig`** (LR, batch, epochs, dims, seed, plus optional **early stopping**: `early_stop_patience`, `early_stop_min_delta`, `early_stop_val_max_edges`). **`use_memory=False`** raises until a no-memory baseline exists. |
| **Data hooks** | **`tgn_amazon/hooks.py`**: **`BipartiteProductNegativeHook`** — one random **product** negative per edge (optional **`torch.Generator`** for reproducibility). |
| **TGN model** | **`tgn_amazon/tgn_model.py`**: **`build_tgn_stack`** — **`TGNMemory`**, **`GraphAttentionEmbedding`**, **`LinkPredictor`**, optional static fusion; **LastAggregator** vs **MeanAggregator** (RQ4). |
| **Training** | **`tgn_amazon/training.py`**: BCE on concatenated pos/neg logits (mean per logit), **`neg != dst`** masked; **`assoc`** sized with **`memory.num_nodes`** (matches global bipartite ids); **`train_epoch`**, **`validation_epoch`** (no grad, same BCE for monitoring), **`run_training_job`** (optional **early stopping** on val loss, restores best weights), **`replay_train_loader_for_memory`**. |
| **Evaluation** | **`tgn_amazon/evaluation.py`**: **MRR** on val/test streams (**`eval_mrr`**, **`run_eval_job`**): random distinct product negatives, memory **`update_state`** per edge; **`_validate_num_negatives_for_eval`**. |
| **Adapter smoke** | **`scripts/run_adapter_smoke.py`**: builds **`DGraph`**, runs **`DGDataLoader`**, prints stats. |
| **Invariant checks** | **`scripts/verify_adapter_invariants.py`**: structural checks on bipartite IDs, times, val cutoff, loader batches. |
| **Training + eval CLI** | **`scripts/train_tgn_baseline.py`**: training then **val** or **test** MRR (`--split`, `--num-negatives`, **`--replay-train-eval`**). Prints a **`run_id`** and appends metrics to CSV (see **Run logging**). |
| **Training smoke** | **`scripts/run_training_smoke.py`**: small graph, 2 epochs, **LastAggregator** vs **MeanAggregator**. |
| **Run logging** | **`tgn_amazon/RunLogger.py`**: appends **`logs/training.csv`** (per-epoch **train** **`mean_loss`** only) and **`logs/eval.csv`** (**`mrr`**, **`n_queries`**, **`n_skipped_no_negative_pool`**, **`n_skipped_would_materialize_full_catalog`**, **`n_skipped_invalid_node_ids`**). Columns **`run_id`**, **`label`** (**`TGN+LastAgg`** / **`TGN+MeanAgg`**), **`config`** (**`AblationConfig.slug()`**) tie rows across files. Early-stop **`val_loss`** is printed to the console, not logged to CSV. |

**Course / writing:** `project_proposal.tex` (proposal); follow your course’s deadlines and submission rules separately.

---

## What we do next

1. **More metrics / tasks** — **Recall@K**, **MAP@K**, or RelBench **task** APIs (e.g. **`user-item-purchase`**) if you need alignment beyond this repo’s bipartite MRR protocol.
2. **Ablations at scale** — Full **`AblationConfig`** sweeps with **`TrainingConfig`** fixed after one validation tuning pass.
3. **Optional** — Plots or dashboards from **`logs/*.csv`**; richer columns if you need **`TrainingConfig`** serialized per run.
4. **Progress / final reports** — Per your course’s schedule.

---

## Setup

**Python 3.10+** recommended (tested with 3.13). You need **internet access** the first time you install packages and the first time RelBench downloads **`rel-amazon`** (large; can take many minutes).

### If you received this project as a zip

1. **Unzip** the archive anywhere on your machine (e.g. Desktop). You should see a folder that contains **`requirements.txt`**, **`tgn_amazon/`**, and **`scripts/`**.
2. Open a terminal and **go to that folder** (the **project root** — the directory that contains `requirements.txt`):

   ```bash
   cd path/to/Project
   ```

   Replace `path/to/Project` with your actual path (on Windows: `cd C:\Users\You\Desktop\Project` or similar).

3. **(Recommended)** Create and activate a virtual environment so dependencies do not clash with other Python projects:

   ```bash
   python -m venv .venv
   ```

   - **Windows (cmd):** `.venv\Scripts\activate.bat`
   - **Windows (PowerShell):** `.venv\Scripts\Activate.ps1`
   - **macOS / Linux:** `source .venv/bin/activate`

4. **Install dependencies:**

   ```bash
   pip install -r requirements.txt
   ```

5. **Run** the scripts from the same project root (see [Running commands](#running-commands) below). The first run that touches RelBench will **download and cache** data (often under `%LOCALAPPDATA%\relbench\...` on Windows, or `~/.cache` / similar on Linux/macOS).

### Quick setup (same steps, minimal)

```bash
cd path/to/Project
pip install -r requirements.txt
```

- **`py-tgb`** is listed for TGM/TGB examples; this project’s Amazon path uses **RelBench** only.

---

## Running commands

**Always run these from the project root** (the folder where `requirements.txt` lives). Use `cd` to that folder first, then:

**Suggested order for a new machine:** (1) `run_adapter_smoke.py` with `--max-edges` — data loads; (2) `verify_adapter_invariants.py` — structural checks; (3) `run_training_smoke.py` — short training; (4) `train_tgn_baseline.py` for training + MRR.

### Data pipeline (no training)

```bash
# Build graph + one loader batch (cap edges for speed)
python scripts/run_adapter_smoke.py --max-edges 50000

# Use all edges after filters (slow; requires cached DB)
python scripts/run_adapter_smoke.py --full

# Ablation-style adapter flags
python scripts/run_adapter_smoke.py --max-edges 50000 --static --homo --no-feat

# Structural assertions (after DB is cached)
python scripts/verify_adapter_invariants.py
```

### Training and evaluation (MRR)

```bash
# Quick sanity: 1000 edges, 2 epochs, LastAgg then MeanAgg
python scripts/run_training_smoke.py

# Train (cap train edges) + val MRR (eval edges uncapped by --max-edges)
python scripts/train_tgn_baseline.py --max-edges 50000 --epochs 1

# Test split; MeanAggregator (RQ4)
python scripts/train_tgn_baseline.py --max-edges 50000 --epochs 1 --split test --mean-agg

# Replay train stream in no_grad before val MRR (slow; memory warm-up heuristic)
python scripts/train_tgn_baseline.py --max-edges 50000 --epochs 1 --replay-train-eval

# Save weights after training (default directory checkpoints/; use --no-save-checkpoint to skip)
python scripts/train_tgn_baseline.py --epochs 3 --checkpoint-dir checkpoints

# Early stopping: val BCE after each epoch; stop if no improvement for N epochs; cap val edges for faster checks
python scripts/train_tgn_baseline.py --epochs 30 --early-stop-patience 3 --early-stop-val-max-edges 50000

# Eval only from a saved checkpoint (skip training; ablation + hyperparams read from the file)
python scripts/train_tgn_baseline.py --load-checkpoint checkpoints/20260408_142533_full_lastagg.pt --split val
```

Common flags for **`train_tgn_baseline.py`**: `--max-edges`, `--epochs`, `--batch-size`, `--lr`, `--mean-agg`, `--static`, `--homo`, `--no-feat`, **`--split` (`val` / `test`)**, **`--num-negatives`** (on large catalogs must be strictly less than `num_products - 1`; validated in **`run_eval_job`**), **`--replay-train-eval`**, **`--checkpoint-dir`**, **`--no-save-checkpoint`**, **`--load-checkpoint`**, **`--early-stop-patience`**, **`--early-stop-min-delta`**, **`--early-stop-val-max-edges`**.

After training, a **`.pt`** file is written under **`checkpoints/`** (by default) named like **`{run_id}_{AblationConfig.slug()}_{lastagg|meanagg}.pt`**. It holds **`state_dict`s** plus **`AblationConfig`** / **`TrainingConfig`** so **`--load-checkpoint`** can reload and run MRR without retraining. If **early stopping** is enabled, the saved weights are the **best validation-loss snapshot** (lowest monitored val BCE), not necessarily the last training epoch. **`checkpoints/`** is gitignored.

Each CLI run prints **`Run ID: …`** and appends to **`logs/training.csv`** and **`logs/eval.csv`** (created under the project root if missing). Ablations are distinguished in the **`config`** column via **`AblationConfig.slug()`** (for example **`--no-feat`** → **`full_nofeat`** when other ablation flags are off).

---

## Repository layout and files

| Path | Role |
|------|------|
| **`project_proposal.tex`** | ACM-style proposal (motivation, RQs, methodology). |
| **`requirements.txt`** | Python dependencies. |
| **`REFERENCES_AND_GUIDE.md`** | Stack, papers, APIs, and implementation detail. |
| **`.gitignore`** | Ignores venvs, caches, LaTeX artifacts. |
| **`tgn_amazon/__init__.py`** | Package exports (`AblationConfig`, `TrainingConfig`, `RelbenchAmazonAdapter`). |
| **`tgn_amazon/config.py`** | **`AblationConfig`**, **`TrainingConfig`**. |
| **`tgn_amazon/adapter.py`** | **`RelbenchAmazonAdapter`**: RelBench → **`DGData`**, **`AdapterMetadata`**. |
| **`tgn_amazon/hooks.py`** | **`BipartiteProductNegativeHook`**. |
| **`tgn_amazon/tgn_model.py`** | **`build_tgn_stack`**. |
| **`tgn_amazon/training.py`** | **`train_epoch`**, **`validation_epoch`**, **`run_training_job`** (optional early stopping), **`make_train_loader`**, replay helper. |
| **`tgn_amazon/evaluation.py`** | **`eval_mrr`**, **`run_eval_job`**. |
| **`tgn_amazon/RunLogger.py`** | **`RunLogger`**: CSV metrics for training and eval. |
| **`tgn_amazon/checkpointing.py`** | Save/load **`.pt`** checkpoints (weights + configs) for **`train_tgn_baseline.py`**. Loading merges missing config keys with current dataclass defaults so older checkpoints stay usable. |
| **`scripts/run_adapter_smoke.py`** | Smoke-test data loading. |
| **`scripts/verify_adapter_invariants.py`** | Graph/loader invariants. |
| **`scripts/run_training_smoke.py`** | Minimal LastAgg vs MeanAgg training. |
| **`scripts/train_tgn_baseline.py`** | Main CLI: train + MRR. |

---

## Limitations

- **Transductive val/test:** When **`reuse_node_maps`** is used for evaluation, the adapter keeps only reviews whose **`customer_id`** and **`product_id`** appear in the train-time maps. Cold-start users or products are **dropped** (not scored). Printed edge counts (for example on val) are **after** this filter—state that explicitly in reports if you compare to “full” RelBench wording.
- **Graph scope:** Single **review** stream (bipartite customers–products), not every entity/relation from the proposal narrative.
- **Metrics:** In-repo **MRR** with random product negatives on time splits; **not** RelBench’s packaged task objects or **Recall@K** / **MAP@K** unless you add them.
- **Neighborhoods:** **In-batch** edges for **`GraphAttentionEmbedding`** (not full **`LastNeighborLoader`** history).
- **TGM warnings:** You may see **int64 → int32** downcasting warnings from **`dg_data.py`**; usually harmless at this graph size.
- **Log CSV schema:** If **`logs/eval.csv`** was written with an older header (fewer columns), archive or delete it before running a newer CLI so appended rows stay aligned with the current column list.
- **Early stopping:** Monitoring uses **validation BCE** on the RelBench val window (same pos/neg protocol as training). **`--early-stop-val-max-edges`** caps only that monitoring pass; **MRR eval** after training still uses the **full** val/test stream unless you change evaluation code. A capped val proxy can disagree with full-val MRR for choosing the best epoch.
- **Early stopping cost:** Each training epoch adds a **full forward pass over the (possibly capped) val stream**, so wall-clock per epoch increases versus training alone.

---

## License / data

Follow **RelBench** and **Amazon review data** terms for `rel-amazon`. Cite **TGM**, **RelBench**, and **TGN** in the report (for example via BibTeX alongside `project_proposal.tex`).
