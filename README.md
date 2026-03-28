# Comp 511 — TGN ablation on RelBench Amazon

McGill **Network Science** course project (see **`project_proposal.tex`**). Goal: run **Temporal Graph Networks (TGN)** via the **Temporal Graph Modelling (TGM)** library on **RelBench `rel-amazon`**, then ablate **time**, **heterogeneity**, **features**, and **memory** and compare ranking metrics.

**Authors:** Harry Hu, Junpeng Wen, Sophia Kyrychenko.

---

## What is done

| Area | Description |
|------|-------------|
| **Dependencies** | `requirements.txt` pins `tgm-lib`, `relbench`, PyTorch / PyG, etc. |
| **RelBench → TGM** | **`tgn_amazon/adapter.py`**: loads `rel-amazon`, keeps reviews **strictly before validation time** for training, builds a **bipartite** graph (customers ↔ products) as TGM **`DGData`**. |
| **Configs** | **`tgn_amazon/config.py`**: **`AblationConfig`** (`static_graph`, `homogeneous`, `use_features`, `use_memory`, `max_review_edges`) and **`TrainingConfig`** (fixed `lr`, `batch_size`, `epochs`, dimensions, `seed` for fair comparisons across ablations). |
| **Data hooks** | **`tgn_amazon/hooks.py`**: **`BipartiteProductNegativeHook`** — one negative **product** id per positive edge (avoids sampling customer nodes as fake items). |
| **TGN model** | **`tgn_amazon/tgn_model.py`**: **`build_tgn_stack`** wires TGM **`TGNMemory`**, **`GraphAttentionEmbedding`**, **`LinkPredictor`**, optional linear fusion of static node features; **LastAggregator** vs **MeanAggregator** for RQ4. |
| **Training** | **`tgn_amazon/training.py`**: BCE link loss, **`train_epoch`**, **`run_training_job`** (loader + hook manager + optimizer with deduplicated params for shared `time_enc`). |
| **Adapter smoke** | **`scripts/run_adapter_smoke.py`**: builds **`DGraph`**, runs **`DGDataLoader`**, prints stats. |
| **Invariant checks** | **`scripts/verify_adapter_invariants.py`**: structural checks on bipartite IDs, times, val cutoff, loader batches. |
| **Training CLI** | **`scripts/train_tgn_baseline.py`**: full training entrypoint (ablation flags, `--mean-agg` for MeanAggregator). |
| **Training smoke** | **`scripts/run_training_smoke.py`**: tiny run (1000 edges, 2 epochs) for **LastAggregator** vs **MeanAggregator** to validate the training path. |

**Course / writing:** `project_proposal.tex`, `511Project.txt` (deadlines, OpenReview).

---

## What we do next

1. **Evaluation** — Align with RelBench tasks (e.g. **`user-item-purchase`** or related link tasks); implement **MRR**, **Recall@K**, and optionally **MAP@K** on held-out time splits (not only training loss).
2. **Logging** — Save runs to **CSV** (and optional plots): config slug, epoch, train/val metrics for reproducibility and the progress/final reports.
3. **Ablations at scale** — Run full **`AblationConfig`** sweeps (static vs temporal, homogeneous vs heterogeneous, no features, memory aggregation) with **`TrainingConfig` held fixed** after a single baseline tuning pass on validation.
4. **Progress report (Apr 3)** — Adapter description, baseline + at least one ablation (e.g. static vs temporal), preliminary numbers.
5. **Final report (Apr 14)** — Full tables, figures, discussion, and limitations (e.g. review-only bipartite view vs full relational schema in the proposal).

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

**Suggested order for a new machine:** (1) `run_adapter_smoke.py` with `--max-edges` — fast-ish check that data loads; (2) `verify_adapter_invariants.py` — structural checks; (3) `run_training_smoke.py` — short training run; (4) `train_tgn_baseline.py` for longer runs.

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

### Training

```bash
# Quick sanity check: 1000 edges, 2 epochs, LastAgg then MeanAgg
python scripts/run_training_smoke.py

# Baseline training (example: cap edges, one epoch)
python scripts/train_tgn_baseline.py --max-edges 50000 --epochs 1

# RQ4: MeanAggregator instead of LastAggregator inside TGNMemory
python scripts/train_tgn_baseline.py --max-edges 50000 --epochs 1 --mean-agg

# Match adapter ablations
python scripts/train_tgn_baseline.py --max-edges 50000 --epochs 1 --static --homo --no-feat
```

Common CLI flags for **`train_tgn_baseline.py`**: `--max-edges`, `--epochs`, `--batch-size`, `--lr`, `--mean-agg`, `--static`, `--homo`, `--no-feat`.

---

## Repository layout and files

| Path | Role |
|------|------|
| **`project_proposal.tex`** | ACM-style proposal (motivation, RQs, methodology). |
| **`511Project.txt`** | Course deadlines and peer-review process. |
| **`requirements.txt`** | Python dependencies. |
| **`.gitignore`** | Ignores venvs, caches, LaTeX artifacts. |
| **`tgn_amazon/__init__.py`** | Package exports (`AblationConfig`, `TrainingConfig`, `RelbenchAmazonAdapter`). |
| **`tgn_amazon/config.py`** | **`AblationConfig`**: data/model ablations; **`TrainingConfig`**: shared training hyperparameters. |
| **`tgn_amazon/adapter.py`** | **`RelbenchAmazonAdapter`**: RelBench → **`DGData`**, **`AdapterMetadata`** (counts, id maps, val/test timestamps). |
| **`tgn_amazon/hooks.py`** | **`BipartiteProductNegativeHook`** for TGM **`HookManager`** / **`DGDataLoader`**. |
| **`tgn_amazon/tgn_model.py`** | **`build_tgn_stack`**: TGM TGN memory + graph attention + link predictor. |
| **`tgn_amazon/training.py`** | **`train_epoch`**, **`run_training_job`**, **`make_train_loader`**. |
| **`scripts/run_adapter_smoke.py`** | Smoke-test data loading and batching only. |
| **`scripts/verify_adapter_invariants.py`** | Asserts graph/loader invariants on a capped subset. |
| **`scripts/run_training_smoke.py`** | Minimal two-model training smoke (LastAgg vs MeanAgg). |
| **`scripts/train_tgn_baseline.py`** | Main training script with CLI. |

---

## Limitations

- **Graph scope:** Single **review** interaction stream (bipartite customers–products), not every entity/relation from the proposal narrative.
- **Training objective:** Link prediction with **BCE** and random product negatives; **no** RelBench task metrics in-repo yet (MRR / Recall@K next).
- **Neighborhoods:** Each step uses **in-batch** edges for **`GraphAttentionEmbedding`** (lightweight; not the full historical neighbor memory of the reference PyG TGN example).

---

## License / data

Follow **RelBench** and **Amazon review data** terms for `rel-amazon`. Cite **TGM**, **RelBench**, and **TGN** in the report (`my_references.bib` with the proposal if you use BibTeX).
