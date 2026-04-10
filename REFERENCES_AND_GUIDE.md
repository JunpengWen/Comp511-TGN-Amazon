# References, stack, and guide (complete)

This document lists **everything this project draws on** (papers, libraries, TGM APIs), **what we implemented**, **limitations**, and **planned future work**. It extends the short **`README.md`** with detail for readers who want one place to look.

---

## 1. Papers

### 1.1 Primary references (directly tied to running code)

| Topic | Reference | How we use it |
|--------|-----------|----------------|
| **TGN** | Rossi et al., *Temporal Graph Networks for Deep Learning on Dynamic Graphs* ([arXiv:2006.10637](https://arxiv.org/abs/2006.10637)) | Architecture: memory, messages, aggregation, temporal attention. Our model uses TGM’s **`TGNMemory`**, **`GraphAttentionEmbedding`** (`TransformerConv`-based), **`IdentityMessage`**, **`LastAggregator`** / **`MeanAggregator`**. |
| **RelBench** | Robinson et al., RelBench ([arXiv:2407.20060](https://arxiv.org/abs/2407.20060)) | Dataset **`rel-amazon`** via **`relbench.datasets.get_dataset`** → **`get_db()`** in **`adapter.py`**; **train** reviews are strictly **before** **`val_timestamp`**; val uses **`[val_timestamp, test_timestamp)`**; test uses **`review_time >= test_timestamp`**. |
| **TGM** | Chmura et al., *Temporal Graph Modelling* ([arXiv:2502.07341](https://arxiv.org/abs/2502.07341); `chmura2025tgm` in `project_proposal.tex`) | **`DGData`**, **`DGraph`**, **`DGBatch`**, **`DGDataLoader`**, **`HookManager`**, **`LinkPredictor`**, and TGN encoder code under **`tgm.nn`**. |

### 1.2 Related work (cited in `project_proposal.tex`, not reimplemented here)

These inform **motivation and RQs** (ablations, heterogeneity, etc.) but are **not** separate code paths in this repo:

- **JODIE**, **TGAT**, **Time2Vec**, **HGT**, **LightGCN**, and other citations in the proposal’s Related Work — background reading only.

### 1.3 Course context

- Use your course’s posted deadlines, peer-review process, and report structure (not part of this repository).

---

## 2. Python dependencies (`requirements.txt`)

| Package | Role in this project |
|---------|----------------------|
| **`torch`** | Tensors, training loop, autograd. Training/eval select **`cuda`** when **`torch.cuda.is_available()`** else **`cpu`** (no CLI device flag). |
| **`torch-geometric`** | **`TransformerConv`** (used *inside* TGM’s `GraphAttentionEmbedding`); PyG is a hard dependency of **`tgm-lib`**. |
| **`tgm-lib`** | TGM: dynamic graph data structures, loader, hooks, TGN modules, link predictor. |
| **`relbench`** | Load **`rel-amazon`**, access **`Database`**, **`val_timestamp`** / **`test_timestamp`**. |
| **`numpy`**, **`pandas`** | RelBench and **`adapter.py`**; **`numpy`** also used in **`evaluation.py`** for large-pool negative sampling (`numpy.random.Generator.choice` without materializing a full **`arange(lo, hi)`**). |
| **`py-tgb`** | Listed for compatibility with TGM’s **TGB** examples and optional loaders; **our Amazon pipeline does not import `py-tgb`** — RelBench only. |

---

## 3. Upstream codebases (where to read source)

| Codebase | URL | Relevance |
|----------|-----|-----------|
| **This repo** | `tgn_amazon/`, `scripts/` | Adapter, configs, hook, training, evaluation. |
| **TGM** | [github.com/tgm-team/tgm](https://github.com/tgm-team/tgm) (PyPI: **`tgm-lib`**) | `tgm/data/`, `tgm/hooks/`, `tgm/nn/encoder/tgn.py`. |
| **PyTorch Geometric** | [github.com/pyg-team/pytorch_geometric](https://github.com/pyg-team/pytorch_geometric) | **`torch_geometric/nn/models/tgn.py`** and **`examples/tgn.py`** — TGM’s TGN closely follows this reference (see TGM docstrings). |
| **RelBench** | [github.com/snap-stanford/relbench](https://github.com/snap-stanford/relbench) | Dataset construction and tasks for **`rel-amazon`**. |

After `pip install tgm-lib`, you can inspect **`site-packages/tgm/`** (e.g. **`hook_manager.py`**, **`loader.py`**).

---

## 4. TGM / PyTorch API surface **used in our code**

Roughly in **data → batch → model** order:

| Component | Module / symbol | Where in our repo |
|-----------|-----------------|-------------------|
| Graph container | **`DGData.from_raw`** | `tgn_amazon/adapter.py` |
| Dynamic graph view | **`DGraph`** | `tgn_amazon/training.py`, `tgn_amazon/evaluation.py` |
| Batching | **`DGDataLoader`** (`batch_unit='r'`) | `tgn_amazon/training.py` (`make_train_loader`) |
| Batch type | **`DGBatch`** | Produced by loader; hooks add **`neg`**. |
| Hook registry | **`HookManager`**, **`set_active_hooks('train')`** | `tgn_amazon/training.py` |
| Memory | **`TGNMemory`**, **`IdentityMessage`** | `tgn_amazon/tgn_model.py` |
| Aggregators | **`LastAggregator`**, **`MeanAggregator`** | `tgn_amazon/tgn_model.py` (RQ4: `--mean-agg` → `MeanAggregator`) |
| Time encoding | **`Time2Vec`** (inside **`TGNMemory`**) | Used by TGM; shared with **`GraphAttentionEmbedding`** |
| GNN block | **`GraphAttentionEmbedding`** | `tgn_amazon/tgn_model.py` |
| Link scores | **`LinkPredictor`** | `tgn_amazon/tgn_model.py` |
| Optimizer | **`torch.optim.Adam`** | `tgn_amazon/training.py` (parameters deduplicated when **`time_enc`** is shared) |

**Not used in our scripts (but exist in TGM):** e.g. **`RecipeRegistry`** / **`RECIPE_TGB_LINK_PRED`** (`tgm/hooks/recipe.py`) — tailored to **TGB** datasets with prebuilt negative samplers; we use a **custom** hook for bipartite Amazon instead.

---

## 5. What this repository **implements** (done)

| Piece | Location | Description |
|-------|----------|-------------|
| RelBench → TGM | **`tgn_amazon/adapter.py`** | Loads **`rel-amazon`**, optional **`max_review_edges`** on train builds, **`review_time < until_timestamp`** for train, **`review_time >= from_timestamp`** for val/test starts, **`reuse_node_maps`** for consistent ids; bipartite **`[0, n_c)`** / **`[n_c, n_c+n_p)`**; **`build_dgdata`** raises if **`use_memory=False`** (guard only). |
| Ablations (data) | **`tgn_amazon/config.py`** **`AblationConfig`** | **`static_graph`**, **`homogeneous`**, **`use_features`**, **`max_review_edges`**, **`use_memory`** (slug + guard). |
| Training hyperparameters | **`tgn_amazon/config.py`** **`TrainingConfig`** | **`learning_rate`**, **`batch_size`**, **`epochs`**, **`memory_dim`**, **`time_dim`**, **`embedding_dim`**, **`seed`**, plus optional **early stopping**: **`early_stop_patience`** (None = off), **`early_stop_min_delta`**, **`early_stop_val_max_edges`** (cap val edges for monitoring only). |
| Bipartite negatives | **`tgn_amazon/hooks.py`** | **`BipartiteProductNegativeHook`**: random product negatives; optional **`torch.Generator`**. If resampling still clashes with **`dst`**, fallback picks a **uniform** valid product id (index sample; **O(1)** memory). |
| Model assembly | **`tgn_amazon/tgn_model.py`** | **`build_tgn_stack`**: static fusion **`nn.Linear`** when features are on. |
| Training | **`tgn_amazon/training.py`** | **`train_epoch`**: BCE sum over valid pos/neg logits, mean for reporting; raises if **no logits** accumulated in an epoch; **`assoc`** length **`memory.num_nodes`**; **`validation_epoch`**: same BCE in **`no_grad`** over the val window for early-stop monitoring; **`run_training_job`**: optional early stopping (best val loss → CPU snapshot → restore; **`RunLogger.log_early_stop_summary`**); **`replay_train_loader_for_memory`** (optional eval warm-up). |
| Evaluation | **`tgn_amazon/evaluation.py`** | **`_eval_ranking_metrics`**: one pass for **MRR** and optional **Recall@K** (tie-aware rank, same 1-pos + random-negs pool). **`run_eval_job`**: **`recall_ks`**, **`cached_train_meta`** (skip second train **`build_dgdata`** when no replay), **`eval_max_edges`** (cap val/test eval stream; **`None`** = full split). **`eval_mrr`**, **`eval_recall_at_k`**; **`_validate_num_negatives_for_eval`**, optional replay; **`RunLogger`** + **`recalls_json`**. |
| Run logging | **`tgn_amazon/RunLogger.py`** | Append-only CSVs under **`logs/`** (directory **gitignored** in this repo). **`_append`** warns if existing file header **column count** ≠ current schema. **`log_early_stop_summary`** → **`early_stop.csv`**; **`log_eval`** + **`recalls_json`** when Recall@K runs. |
| CLI | **`scripts/train_tgn_baseline.py`** | Training + MRR: **`--split`**, **`--num-negatives`**, **`--recall-ks`**, **`--eval-max-edges`**, **`--replay-train-eval`**, ablation flags; **`--early-stop-*`**, **`--seed`**; **`--load-checkpoint`** / **`--checkpoint-dir`**; passes **`cached_train_meta`** into **`run_eval_job`** when not replaying (avoids duplicate train build). |
| Smoke / invariants | **`scripts/run_adapter_smoke.py`**, **`run_training_smoke.py`**, **`verify_adapter_invariants.py`** | As in **`README.md`**. |
| Package exports | **`tgn_amazon/__init__.py`** | **`AblationConfig`**, **`TrainingConfig`**, **`RelbenchAmazonAdapter`**. |
| Checkpoints | **`tgn_amazon/checkpointing.py`** | **`save_training_checkpoint`** / **`load_training_checkpoint_dict`** (**`weights_only=False`**: load **trusted** files only). **`configs_from_checkpoint`** merges saved dicts with current dataclass defaults. Checkpoints do **not** embed id maps; eval still needs RelBench + **`build_dgdata`**. |

**RQ4-style comparison** is **`LastAggregator`** vs **`MeanAggregator`** inside **`TGNMemory`** (CLI **`--mean-agg`**). A true “no memory” baseline is **not** implemented; **`use_memory=False`** fails fast in **`build_dgdata`**.

**CSV outputs (default paths):**

| File | Columns (high level) |
|------|----------------------|
| **`logs/training.csv`** | **`run_id`**, **`label`**, **`config`**, **`epoch`**, **`mean_loss`** (train only), **`timestamp`** — one row per completed training epoch. Early-stop **`val_loss`** is not a column (printed in the console only). |
| **`logs/eval.csv`** | **`run_id`**, **`label`**, **`config`**, **`split`**, **`num_negatives`**, **`mrr`**, **`n_queries`**, skip counts, **`recalls_json`** (JSON string **`{"10": 0.35, ...}`** when **`--recall-ks`** is used; else empty), **`timestamp`** — one row per **`run_eval_job`** call. |
| **`logs/early_stop.csv`** | **`run_id`**, **`label`**, **`config`**, **`best_epoch`**, **`best_val_loss`**, **`epochs_completed`**, **`stopped_early`**, **`timestamp`** — one row per training run that used early stopping. |

**`scripts/run_training_smoke.py`** does not instantiate **`RunLogger`**; use **`train_tgn_baseline.py`** (or pass a logger from your own script) for CSV logs.

---

## 6. `HookManager` (why it exists here)

TGM attaches **hooks** to the dataloader so each **`DGBatch`** can be augmented (negatives, etc.) **before** the training loop. **`HookManager`** registers hooks per **key** (we use **`'train'`**), resolves **`requires` / `produces`**, sorts execution order, and **`DGDataLoader`** calls **`execute_active_hooks`** when a hook manager is passed. You must **`set_active_hooks('train')`** before iterating.

**Conceptually:** hooks are **not** layers of the neural network; they are **data pipeline** steps shared with TGM’s TGB-oriented recipes.

*(Full behavioral detail is in **`site-packages/tgm/hooks/hook_manager.py`**.)*

---

## 7. Training loop (high level)

1. **Loader** yields temporal batches; **hook** adds **`neg`** (same length as positive **`dst`**).
2. **Local** **`edge_index`** uses **`assoc[src]`**, **`assoc[dst]`** with **`assoc`** sized to **`memory.num_nodes`** (global bipartite ids; negatives may reference products not present in the current batch’s edge list).
3. **Loss:** BCE with logits on masked valid rows (**`neg != dst`**); mean over **all** pos/neg logits in the batch for the epoch (comparable across batch sizes).
4. **After step:** **`memory.detach()`**, **`memory.update_state(src, dst, t, raw_msg)`**.

### Early stopping (when enabled)

1. **`run_training_job`** builds a **val-window** **`DGData`** with **`reuse_node_maps`** from the train build (same id space as **`run_eval_job`** val split). Optional **`early_stop_val_max_edges`** caps the **monitoring** graph only (first *N* val-window edges after adapter filters).
2. After each **train** epoch, **`validation_epoch`** resets memory, runs the val loader once (with **`BipartiteProductNegativeHook`** and a separate **`torch.Generator`** seed), and reports **mean val BCE** (identical masking and pos/neg construction as training, no backward).
3. **Improvement** means **`val_loss < best_val - early_stop_min_delta`**. If there is no improvement for **`early_stop_patience`** consecutive epochs, training stops early.
4. **Returned modules and saved checkpoints** (from **`train_tgn_baseline.py`**) use the **best val-loss** weights, not the last epoch—even if all **`epochs`** complete without triggering patience, weights are restored to the best snapshot when early stopping is on.
5. **Cost:** Each epoch includes an extra full pass over the (possibly capped) val stream, so per-epoch time increases. **`early_stop_val_max_edges`** trades off monitoring fidelity vs speed.

---

## 8. Evaluation (MRR) (high level)

1. **`run_eval_job`** builds eval **`DGData`** with **`reuse_node_maps`** from train. **`eval_max_edges`** (CLI **`--eval-max-edges`**) sets **`AblationConfig.max_review_edges`** for the **eval** build only (first *N* edges after time filters); **`None`** keeps the **full** val/test stream. It validates **`num_negatives`** vs catalog size when needed. With **`cached_train_meta`** and no replay, it can **skip** rebuilding the train graph (see **`README.md`** limitations).
2. **`_eval_ranking_metrics`** (used by **`eval_mrr`** and **`run_eval_job`**) iterates edges, samples negatives (small pool: **`torch.randperm`**; large pool: **NumPy `choice`**, mixed RNG with **`torch.Generator`**-derived seed), scores **1 positive + sampled negatives** (true item first in the candidate list), and computes **tie-aware average rank** (same rule for MRR and Recall@K). **Recall@K** is the fraction of queries with **`avg_rank <= K`**; it is **not** full-catalog retrieval—only the sampled negatives define the competition set. If **`K ≥`** number of candidates (**`1 + num_negatives`**), **Recall@K** is **1.0** for all non-skipped queries (vacuous). **`memory.update_state`** runs per edge; memory is not reset at entry unless you use replay outside this flow.
3. **`eval_recall_at_k`** is a thin wrapper for scripts/notebooks that only need Recall@K (and optionally **`mrr`** in the same pass).
4. **`_set_tgn_memory_eval_mode`** avoids **`TGNMemory.eval()`** OOM on large graphs.

---

## 9. Design choices and limitations (current)

- **Single relation stream:** **reviews** only → bipartite **customer–product** edges (not every table/relation in the proposal narrative).
- **Neighborhoods:** **In-batch** edges for **`GraphAttentionEmbedding`**, not PyG’s **`LastNeighborLoader`** over full history.
- **Evaluation:** **MRR** and optional **Recall@K** on the same random negative sample as MRR; **not** RelBench task leaderboard wiring, full-catalog **Recall@K**, or **MAP@K** in-repo.
- **Negatives:** Random product ids for train and eval (not hard negatives, not TGB’s precomputed eval sets).
- **TGM:** **`UserWarning`** about **int64 → int32** downcasting in **`dg_data.py`** is common; ids here are below int32 range.
- **Early stopping vs MRR:** The monitored metric is **val BCE** (optionally on a **capped** val prefix). **MRR / Recall@K** use the **full** eval stream by default; **`--eval-max-edges`** caps that stream for smoke tests. The best epoch for BCE is not guaranteed to maximize MRR.
- **GPU:** Same **`torch.cuda.is_available()`** rule as training; install a **CUDA** PyTorch wheel and working NVIDIA drivers for GPU runs.

---

## 10. Future work (planned)

1. **More metrics / RelBench tasks** — **MAP@K**, full-catalog retrieval **Recall@K**, or official **RelBench** task evaluation if required for comparison.
2. **Ablations at scale** — Sweep **`AblationConfig`** × **`TrainingConfig`** after one validation tuning pass.
3. **Optional** — Plots or aggregation from **`logs/*.csv`**; append **`val_loss`** per epoch to **`training.csv`** if you want full spreadsheet parity with the console ( **`early_stop.csv`** already logs best epoch / best val); serialized **`TrainingConfig`** per eval row; stronger neighbor sampling (closer to PyG **`LastNeighborLoader`**); a real **no-memory** baseline if **`use_memory`** is implemented; explicit **`--device`** / **`--cpu`** CLI flags.
4. **Reports** — Per your course’s requirements.

---

## 11. Suggested reading order

1. **TGN paper** — Sections on memory and message passing.
2. **PyG `examples/tgn.py`** — Training *shape* (memory → GNN → predictor → **`update_state`**).
3. **TGM** `DGDataLoader` + **`HookManager`** — How batches are built and hooked.
4. This repo **`adapter.py`**, **`training.py`**, **`evaluation.py`** — RelBench Amazon graph, loss, MRR.

For bibliography entries used in LaTeX, see **`project_proposal.tex`** and whatever `.bib` file you maintain with it.
