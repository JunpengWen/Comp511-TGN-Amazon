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

- **`511Project.txt`** — deadlines, OpenReview, report structure (not a research paper).

---

## 2. Python dependencies (`requirements.txt`)

| Package | Role in this project |
|---------|----------------------|
| **`torch`** | Tensors, training loop, autograd. |
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
| Training hyperparameters | **`tgn_amazon/config.py`** **`TrainingConfig`** | **`lr`**, **`batch_size`**, **`epochs`**, **`memory_dim`**, **`time_dim`**, **`embedding_dim`**, **`seed`**. |
| Bipartite negatives | **`tgn_amazon/hooks.py`** | **`BipartiteProductNegativeHook`**: random product negatives; optional **`torch.Generator`**. |
| Model assembly | **`tgn_amazon/tgn_model.py`** | **`build_tgn_stack`**: static fusion **`nn.Linear`** when features are on. |
| Training | **`tgn_amazon/training.py`** | **`train_epoch`**: BCE sum over valid pos/neg logits, mean for reporting; **`assoc`** length **`memory.num_nodes`**; **`run_training_job`**, **`replay_train_loader_for_memory`** (optional eval warm-up). |
| Evaluation | **`tgn_amazon/evaluation.py`** | **`eval_mrr`**: **MRR** with **`K`** random distinct product negatives; **`run_eval_job`**: uncapped eval graph, **`_validate_num_negatives_for_eval`**, optional replay; metrics include skip counts. |
| CLI | **`scripts/train_tgn_baseline.py`** | Training + MRR: **`--split`**, **`--num-negatives`**, **`--replay-train-eval`**, ablation flags. |
| Smoke / invariants | **`scripts/run_adapter_smoke.py`**, **`run_training_smoke.py`**, **`verify_adapter_invariants.py`** | As in **`README.md`**. |
| Package exports | **`tgn_amazon/__init__.py`** | **`AblationConfig`**, **`TrainingConfig`**, **`RelbenchAmazonAdapter`**. |
| Notes | **`MRR_EVALUATION_REVIEW.md`** | Protocol details and naming (**MRR** vs typo **MMR**). |

**RQ4-style comparison** is **`LastAggregator`** vs **`MeanAggregator`** inside **`TGNMemory`** (CLI **`--mean-agg`**). A true “no memory” baseline is **not** implemented; **`use_memory=False`** fails fast in **`build_dgdata`**.

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

---

## 8. Evaluation (MRR) (high level)

1. **`run_eval_job`** builds eval **`DGData`** with **`max_review_edges=None`** (full val/test stream), **`reuse_node_maps`** from train, and validates **`num_negatives`** vs catalog size when needed.
2. **`eval_mrr`** iterates edges, samples negatives (small pool: **`torch`**; large pool: **NumPy `choice`**, mixed RNG — see **`MRR_EVALUATION_REVIEW.md`**), ranks **1 + K** candidates, tie-aware rank, **`memory.update_state`** per edge; does not reset memory at entry unless you use replay outside this flow.
3. **`_set_tgn_memory_eval_mode`** avoids **`TGNMemory.eval()`** OOM on large graphs.

---

## 9. Design choices and limitations (current)

- **Single relation stream:** **reviews** only → bipartite **customer–product** edges (not every table/relation in the proposal narrative).
- **Neighborhoods:** **In-batch** edges for **`GraphAttentionEmbedding`**, not PyG’s **`LastNeighborLoader`** over full history.
- **Evaluation:** **MRR** with random product negatives on time splits; **not** RelBench task leaderboard wiring or **Recall@K** / **MAP@K** in-repo.
- **Negatives:** Random product ids for train and eval (not hard negatives, not TGB’s precomputed eval sets).
- **TGM:** **`UserWarning`** about **int64 → int32** downcasting in **`dg_data.py`** is common; ids here are below int32 range.

---

## 10. Future work (planned)

1. **Logging** — CSV (and optional plots): config slug, epoch, train/val metrics.
2. **More metrics / RelBench tasks** — **Recall@K**, **MAP@K**, or official **RelBench** task evaluation if required for comparison.
3. **Ablations at scale** — Sweep **`AblationConfig`** × **`TrainingConfig`** after one validation tuning pass.
4. **Optional** — Stronger neighbor sampling (closer to PyG **`LastNeighborLoader`**); a real **no-memory** baseline if **`use_memory`** is implemented.
5. **Reports** — Per **`511Project.txt`**.

---

## 11. Suggested reading order

1. **TGN paper** — Sections on memory and message passing.
2. **PyG `examples/tgn.py`** — Training *shape* (memory → GNN → predictor → **`update_state`**).
3. **TGM** `DGDataLoader` + **`HookManager`** — How batches are built and hooked.
4. This repo **`adapter.py`**, **`training.py`**, **`evaluation.py`** — RelBench Amazon graph, loss, MRR.

For bibliography entries used in LaTeX, see **`project_proposal.tex`** and your **`my_references.bib`** (if present).
