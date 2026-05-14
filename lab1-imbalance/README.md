# Lab 1 — Undersampling and Oversampling

Demonstrates how class-imbalance methods affect a binary classifier trained on DINOv2 embeddings of the HAM10000 dermoscopy dataset (`kmader/skin-cancer-mnist-ham10000` on Kaggle), with a **lesion-aware** train/valid/test split and the train split artificially imbalanced to 1:50.

## Deliverables

- `lab1.ipynb` — executed notebook with code, plots, and tables.
- `results.csv` — 18 rows from the main analysis at artificial 1:50 imbalance.
- `results_natural.csv` — 18 rows from Section 10's bonus comparison at HAM10000's natural ~1:8 imbalance.
- `CELL_NOTES.md` — cell-by-cell walkthrough explaining constants, choices, and terminology.

## Launch

### Locally (Linux + NVIDIA GPU)

```bash
# from repo root
uv sync
uv run jupyter lab lab1-imbalance/lab1.ipynb
```

JupyterLab opens in your default browser. Run cells top-to-bottom (or **Kernel → Restart Kernel and Run All Cells**).

### Locally (macOS / Windows / CPU-only)

See the top-level `README.md` § *Run anywhere – B*. The notebook's setup cell auto-detects `cuda → mps → cpu`. Expect:

- **Apple-Silicon MPS:** DINOv2 forward pass ≈ 20–30 min for HAM10000's ~10k images.
- **CPU only:** ≈ 1–2 hours for the forward pass.
- All other cells (PCA / UMAP / t-SNE / model fits) take a few minutes total regardless of device.

### Remote (GCP VM with port-forwarded JupyterLab)

See the top-level `README.md` § *Run anywhere – C*.

## Data

Downloaded automatically on first run via `kagglehub.dataset_download("kmader/skin-cancer-mnist-ham10000")` (~5.2 GB; cached afterwards). Requires Kaggle API credentials at `~/.kaggle/kaggle.json` (or `~/.kaggle/access_token` from `kagglehub login`).

The dataset ships as two image directories (`HAM10000_images_part_1`, `HAM10000_images_part_2`) plus `HAM10000_metadata.csv`. The notebook reads the CSV's `lesion_id` column and uses `GroupShuffleSplit` to build train/valid/test (70/15/15) such that **every capture of one lesion lives in exactly one split** — no leakage from HAM10000's multi-capture entries.

### Why this dataset and not `drscarlat/melanoma`

The DermMel mirror (`drscarlat/melanoma`) is HAM10000 with melanoma artificially augmented ~8× (zoom / rotation / shift / flips) to balance the dataset, and the augmentation timing relative to its train/valid/test split is undocumented. That setup is **leaky two ways**: (a) HAM10000's own multi-capture lesions split across train/test, and (b) augmented variants of the same source image split across train/test. Earlier runs on DermMel produced MCC ≈ 0.86 for bare LogReg — that was the leakage talking. Pulling HAM10000 raw and grouping by `lesion_id` gives honest numbers.

## Outputs

- `embeddings/{train,valid,test}.npz` — DINOv2 features for the 1:50 imbalanced main analysis.
- `embeddings/train_natural.npz` — extra cache for Section 10's natural-1:8 sweep (7,002 train samples vs the 6,356 in `train.npz`).
- `results.csv` (1:50) and `results_natural.csv` (1:8) — committed alongside the notebook so reviewers can inspect numbers without re-executing.

> **Cache invalidation:** the embedding cache key is the split name only. If you change `TARGET_RATIO` or `SEED` (which alters the split AND the minority subsample), delete `embeddings/*.npz` before re-running. The notebook's `X_train.shape[0] == (df['split'] == 'train').sum()` assertion will catch a count mismatch but won't flag identity drift.

## Reproducibility

Single seed `42` (torch / numpy / sklearn / `GroupShuffleSplit` / xgboost / samplers). Embedding extraction is deterministic; classifier training uses fixed seeds. No multi-seed averaging — Lab 1 is descriptive, not inferential.

## Notes / known fallbacks

- **Python 3.13 (not 3.14):** PyTorch's `cu124` index has no `cp314` wheels at the time of setup. Revisit if PyTorch publishes them.
- **KMeansSMOTE config:** at 1:50 imbalance (~124 minority samples after the lesion-aware split + imbalance) the default `kmeans_estimator=8` and default `cluster_balance_threshold` reject every cluster. The notebook uses `kmeans_estimator=4, cluster_balance_threshold=0.0` to keep it usable.
- **NearMiss-1 is actively harmful in this run:** all three classifiers + NearMiss-1 land at MCC 0.15–0.20, well below `cw_logreg` (0.32) and below most SMOTE / RandomUnder variants. `NearMiss-1_xgb` (0.15) even underperforms bare LogReg. This is the lab's interesting negative result, not a bug — see the conclusions section.
- **Bare MLP / XGBoost collapse at 1:50:** with no rebalancing, both predict the majority class for essentially every test sample. Recall_pos = 0.000 (MLP), 0.011 (XGB). PR-AUC stays informative (0.22–0.45), confirming the *ranking* signal is in the embeddings — it's the 0.5 threshold that's wrong without intervention.
- **Section 10 (natural 1:8) shows the inverse story:** at HAM10000's actual prevalence, bare MLP MCC jumps from 0.000 → 0.432 and **becomes the global winner** — no imbalance method beats it. Best resampler is `KMeansSMOTE_logreg` (0.417). Several sampling methods (SMOTE_xgb, NearMiss-1 all flavors, RandomUnder_mlp) are *worse* than bare baselines. Imbalance-method utility scales with imbalance severity, not vice-versa.
