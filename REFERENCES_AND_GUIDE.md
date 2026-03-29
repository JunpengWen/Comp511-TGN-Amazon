# References, stack, and guide (complete)

This document lists **everything this project draws on** (papers, libraries, TGM APIs), **what we implemented**, **limitations**, and **planned future work**. It extends the short **`README.md`** with detail for readers who want one place to look.

---

## 1. Papers

### 1.1 Primary references (directly tied to running code)

| Topic | Reference | How we use it |
|--------|-----------|----------------|
| **TGN** | Rossi et al., *Temporal Graph Networks for Deep Learning on Dynamic Graphs* ([arXiv:2006.10637](https://arxiv.org/abs/2006.10637)) | Architecture: memory, messages, aggregation, temporal attention. Our model uses TGM’s **`TGNMemory`**, **`GraphAttentionEmbedding`** (`TransformerConv`-based), **`IdentityMessage`**, **`LastAggregator`** / **`MeanAggregator`**. |
| **RelBench** | Robinson et al., RelBench ([arXiv:2407.20060](https://arxiv.org/abs/2407.20060)) | Dataset **`rel-amazon`** via **`relbench.datasets.get_dataset`** → **`get_db()`** in **`adapter.py`**; **train** reviews are cut at **`val_timestamp`**. |
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
| **`numpy`**, **`pandas`** | Used by RelBench and by **`adapter.py`** (e.g. review table handling). |
| **`py-tgb`** | Listed for compatibility with TGM’s **TGB** examples and optional loaders; **our Amazon pipeline does not import `py-tgb`** — RelBench only. |

---

## 3. Upstream codebases (where to read source)

| Codebase | URL | Relevance |
|----------|-----|-----------|
| **This repo** | `tgn_amazon/`, `scripts/` | Adapter, configs, custom hook, training. |
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
| Dynamic graph view | **`DGraph`** | `tgn_amazon/training.py` (`run_training_job`) |
| Batching | **`DGDataLoader`** (`batch_unit='r'`) | `tgn_amazon/training.py` (`make_train_loader`) |
| Batch type | **`DGBatch`** | Produced by loader; fields like `edge_src`, `edge_dst`, `edge_time`, `edge_x`; hooks add **`neg`**. |
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
| RelBench → TGM | **`tgn_amazon/adapter.py`** | Loads **`rel-amazon`**, sorts reviews by time, optional **`max_review_edges`**, filters **`review_time < until_timestamp`** (train split), bipartite ids: customers **`[0, n_c)`**, products **`[n_c, n_c+n_p)`**, builds **`DGData`** + **`AdapterMetadata`**. |
| Ablations (data) | **`tgn_amazon/config.py`** **`AblationConfig`** | **`static_graph`** (event index vs Unix time), **`homogeneous`** (drop `node_type` / `edge_type`), **`use_features`** (edge + static node features vs zeros), **`max_review_edges`**. |
| Training hyperparameters | **`tgn_amazon/config.py`** **`TrainingConfig`** | **`lr`**, **`batch_size`**, **`epochs`**, **`memory_dim`**, **`time_dim`**, **`embedding_dim`**, **`seed`**. |
| Bipartite negatives | **`tgn_amazon/hooks.py`** | **`BipartiteProductNegativeHook`**: one random **product** negative per edge (`requires` / `produces` for TGM). |
| Model assembly | **`tgn_amazon/tgn_model.py`** | **`build_tgn_stack`**: optional **`nn.Linear`** to fuse **static node features** with memory before **`GraphAttentionEmbedding`**. |
| Training | **`tgn_amazon/training.py`** | **`train_epoch`** (BCE pos/neg), **`run_training_job`**, **`raw_msg_dim_from_config`**, **`make_train_loader`**. |
| CLI | **`scripts/train_tgn_baseline.py`** | Flags: **`--max-edges`**, **`--epochs`**, **`--mean-agg`**, **`--static`**, **`--homo`**, **`--no-feat`**, etc. |
| Smoke tests | **`scripts/run_adapter_smoke.py`**, **`scripts/run_training_smoke.py`** | Data-only vs tiny training (LastAgg vs MeanAgg). |
| Invariants | **`scripts/verify_adapter_invariants.py`** | Asserts bipartite structure, time order, val cutoff, loader batch. |
| Package exports | **`tgn_amazon/__init__.py`** | **`AblationConfig`**, **`TrainingConfig`**, **`RelbenchAmazonAdapter`**. |

**Note:** **`AblationConfig.use_memory`** is reserved for a future “no persistent memory” variant; **current** RQ4-style comparison is **`LastAggregator`** vs **`MeanAggregator`** inside **`TGNMemory`** (CLI **`--mean-agg`**).

---

## 6. `HookManager` (why it exists here)

TGM attaches **hooks** to the dataloader so each **`DGBatch`** can be augmented (negatives, etc.) **before** the training loop. **`HookManager`** registers hooks per **key** (we use **`'train'`**), resolves **`requires` / `produces`**, sorts execution order, and **`DGDataLoader`** calls **`execute_active_hooks`** when a hook manager is passed. You must **`set_active_hooks('train')`** before iterating.

**Conceptually:** hooks are **not** layers of the neural network; they are **data pipeline** steps shared with TGM’s TGB-oriented recipes.

*(Full behavioral detail is in **`site-packages/tgm/hooks/hook_manager.py`**.)*

---

## 7. Training loop (high level)

1. **Loader** yields temporal batches; **hook** adds **`neg`** (same length as positive **`dst`**).
2. **Local** **`edge_index`** = **`assoc[src]`**, **`assoc[dst]`** so **`GraphAttentionEmbedding`** indexes **`z`** / **`last_update`** correctly.
3. **Loss:** BCE with logits on positive and negative pairs.
4. **After step:** **`memory.detach()`**, **`memory.update_state(src, dst, t, raw_msg)`** (standard TGN training pattern).

---

## 8. Design choices and limitations (current)

- **Single relation stream:** **reviews** only → bipartite **customer–product** edges (not every table/relation in the proposal narrative).
- **Neighborhoods:** **In-batch** edges for **`GraphAttentionEmbedding`**, not PyG’s **`LastNeighborLoader`** over full history (lighter, different from the reference **`examples/tgn.py`** setup).
- **Evaluation:** **Training loss** only in code; **MRR / Recall@K / MAP@K** and RelBench **task** evaluation are **future** (see below).
- **Negatives:** Random **product** ids (not hard negatives, not TGB’s precomputed eval negatives).

---

## 9. Future work (planned)

Aligned with **`README.md`** and the proposal:

1. **Evaluation** — Hook into RelBench **tasks** (e.g. **`user-item-purchase`** or related); report **MRR**, **Recall@K**, optionally **MAP@K** on proper time-based splits (not only BCE on random negatives).
2. **Logging** — CSV (and optional plots): config slug, epoch, train/val metrics for reproducibility and reports.
3. **Ablations at scale** — Sweep **`AblationConfig`** × **`TrainingConfig`** (fixed after one validation tuning pass): static vs temporal, homogeneous vs heterogeneous, no features, **LastAggregator** vs **MeanAggregator** (and eventually **`use_memory`** if a separate architecture is added).
4. **Optional** — Stronger neighbor sampling (closer to PyG **`LastNeighborLoader`**), **`use_memory`** wiring if you replace **`TGNMemory`** with a non-memory baseline for RQ4.
5. **Reports** — Progress (e.g. Apr 3) and final (e.g. Apr 14) per **`511Project.txt`**.

---

## 10. Suggested reading order

1. **TGN paper** — Sections on memory and message passing.
2. **PyG `examples/tgn.py`** — Training *shape* (memory → GNN → predictor → **`update_state`**).
3. **TGM** `DGDataLoader` + **`HookManager`** — How batches are built and hooked.
4. This repo **`adapter.py`** + **`training.py`** — Concrete RelBench Amazon graph and loss.

For bibliography entries used in LaTeX, see **`project_proposal.tex`** and your **`my_references.bib`** (if present).
