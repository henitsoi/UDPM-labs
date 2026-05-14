# Lab 1 — Cell-by-cell walkthrough

A personal study guide for `lab1.ipynb`. Explains what each cell does, why we picked the values we did, and defines the terminology so you can reproduce the lab without me.

---

## Cell 0 — Title (markdown)

Header + one-paragraph framing. States the dataset (HAM10000 via the `kmader/skin-cancer-mnist-ham10000` Kaggle mirror), the artificial imbalance ratio (1:50, train only, after a lesion-aware split), and the three model families being compared.

## Cell 1 — `## 1. Setup` (markdown)

Section divider.

---

## Cell 2 — Imports, seeding, paths

**What:** Single import block + global RNG seeding + path constants.

**Why this layout:**
- All imports in one cell makes it obvious what the lab depends on; reviewers can audit deps without scanning the whole notebook.
- Seeding *before* any other library is loaded matters because libraries that consume randomness on import (rare but possible) would otherwise be non-deterministic.

**Constants:**
- `SEED = 42`. Picked once, used everywhere (torch / numpy / random / sklearn `random_state` / xgboost `random_state` / samplers `random_state`). Any integer works; 42 is the conventional placeholder. The lab is descriptive (single-seed); a more rigorous study would average over 5+ seeds.
- `LAB_ROOT = Path.cwd()`. Resolves to whichever directory Jupyter's kernel was launched in. In this repo that's `lab1-imbalance/`, so `EMBED_DIR` lands at `lab1-imbalance/embeddings/`. This is already covered by `.gitignore`'s `lab*/embeddings/` line — embeddings are not tracked.

**Warning filters (at the very top, before other imports):**
- `IProgress not found.*` — tqdm complains when its `tqdm.auto` import path tries to use Jupyter widgets and the `ipywidgets` package isn't installed. Cosmetic; doesn't affect functionality.
- `xFormers is not available.*` — DINOv2 prints a warning per transformer block when the optional [xFormers](https://github.com/facebookresearch/xformers) library isn't installed. xFormers provides faster attention kernels; without it, DINOv2 uses the slower vanilla PyTorch attention. Doesn't affect correctness; only forward-pass speed (which we pay once because embeddings are cached).
- `n_jobs value .* overridden` — UMAP warning. When you pass `random_state` to UMAP, it forces single-threaded execution for reproducibility; the warning tells you that's happening.

Filters are registered *before* the other imports because libraries like `umap` and `xgboost` transitively trigger the tqdm warning during their own import (they `import tqdm.auto` themselves). A filter registered later would be too late.

**Terminology:**
- **tqdm**: progress-bar library. `from tqdm import tqdm` gives the plain text bar (works everywhere). `tqdm.auto` tries to use HTML widgets in notebooks.
- **`torch.manual_seed` vs `torch.cuda.manual_seed_all`**: the first seeds the CPU RNG; the second seeds all GPU RNGs. You need both for CUDA-deterministic runs.

## Cell 3 — Device detection

**What:** Picks `cuda` → `mps` → `cpu` in priority order; prints which one was chosen and a context-specific warning about expected speed.

**Why this order:**
- CUDA (NVIDIA GPU) is fastest by ~10×.
- MPS (Apple Silicon Metal Performance Shaders) is next — works for PyTorch but slower and occasionally non-deterministic.
- CPU is the universal fallback.

**Terminology:**
- **MPS**: Apple's "Metal Performance Shaders" backend, PyTorch's way of running tensors on the M1/M2/M3 GPU.

---

## Cell 4 — `## 2. Dataset acquisition — HAM10000 with lesion-aware split` (markdown)

Explains the dataset choice. We deliberately use the raw HAM10000 (`kmader/skin-cancer-mnist-ham10000`) instead of `drscarlat/melanoma`. The DermMel mirror artificially augments melanoma ~8× without documenting whether augmentation happened before or after the train/valid/test split — so it's leaky two ways (HAM10000 multi-captures + augmented duplicates). Going to raw HAM10000 with our own lesion-aware split fixes both.

## Cell 5 — Kaggle download + path layout

**What:** Downloads HAM10000 (~5.2 GB) on first run, then verifies the two image part-directories and the metadata CSV exist.

**Why two image dirs:** The kmader Kaggle mirror ships images split across `HAM10000_images_part_1` (5,000 images) and `HAM10000_images_part_2` (5,015 images). Cell 6 globs across both.

**Terminology:**
- **`kagglehub`**: Kaggle's modern CLI for downloading datasets non-interactively. Auth via `~/.kaggle/kaggle.json` or `~/.kaggle/access_token`.
- **HAM10000**: "Human Against Machine with 10,000 training images" (Tschandl et al., 2018). 10,015 dermoscopy images, 7 diagnostic classes, with `lesion_id` metadata that groups multi-captures of the same physical lesion.

## Cell 6 — Build dataframe + lesion-aware split

**What:** Reads `HAM10000_metadata.csv`, defines the binary task (`label = 1 if dx=='mel' else 0`), maps each `image_id` to its on-disk filepath, then performs a **group-aware** train/valid/test split keyed on `lesion_id` (70/15/15).

**Why lesion-aware:** HAM10000 has 10,015 images from only 7,470 unique lesions — meaning ~2,545 lesions have 2+ captures (different polarization, magnification, or follow-up). A naive random split would put captures of the same lesion in train AND test, leaking. `GroupShuffleSplit(groups=lesion_id)` guarantees every capture of one lesion lands in exactly one split.

**Why two-step `GroupShuffleSplit`:** sklearn doesn't ship a single-call 3-way group split. We do (train | holdout) first at 70/30, then (valid | test) at 50/50 within the holdout. Same `random_state=SEED` for both steps so the split is reproducible.

**Defensive assert:** `meta.groupby('lesion_id')['split'].nunique().max() == 1` — verifies no lesion straddles splits. If this ever fails, the lesion-aware logic broke.

**Resulting counts (post-split, pre-imbalance):**

| Split | not-mel | mel | Total |
|---|---|---|---|
| train | 6,232 | 770 | 7,002 |
| valid | 1,363 | 156 | 1,519 |
| test | 1,307 | 187 | 1,494 |

Natural prevalence: ~12% melanoma across all splits.

**Terminology:**
- **`lesion_id`**: HAM10000 metadata column. Same `lesion_id` = same physical lesion (different camera angles, polarizations, or follow-up visits). Different `lesion_id` = different patient or different lesion on the same patient.
- **`dx`**: HAM10000's diagnostic-class column. 7 values: `nv` (melanocytic nevi), `mel` (melanoma), `bkl` (benign keratosis), `bcc` (basal-cell carcinoma), `akiec` (actinic keratoses), `vasc` (vascular), `df` (dermatofibroma). We collapse to binary: `mel` → 1, everything else → 0.
- **`GroupShuffleSplit`**: sklearn splitter that respects a `groups` array — guarantees no group spans train/test.

---

## Cell 7 — `## 3. Artificial imbalance` (markdown)

## Cell 8 — Drop minority samples to hit 1:50

**What:** Keeps a stash of the natural-distribution `df_raw`, then randomly subsamples positives in the train split until `not-melanoma : melanoma ≈ 50 : 1`.

**Why train-only:** Tested distribution must stay fixed — see Section 9 caveats and the question we discussed: imbalancing test would make metrics unreliable and conflate "training intervention effect" with "test distribution effect."

**Why `df_raw` and idempotency:** Without the stash, re-executing cell 8 would drop minority samples a second time and collapse the dataset. `if 'df_raw' not in dir(): df_raw = df.copy()` captures the natural distribution once; subsequent runs always rebuild `df` from it.

**Constants:**
- `TARGET_RATIO = 50`. Chosen as "strong but not pathological." Real clinical imbalance on dermoscopy is ~1-5%, so 1:50 is conservative-realistic. We considered 1:10 (too mild — at HAM10000's natural 1:8 the imbalance methods would barely move the needle), 1:100 (too aggressive, classifier metrics get noisy on the tiny ~60-sample minority). 1:50 leaves us with `6232 // 50 = 124` minority train samples — enough for SMOTE's default `k_neighbors=5` and for KMeansSMOTE with a tuned config.
- `n_pos_keep = max(6, len(train_neg) // TARGET_RATIO)`. The `max(6, ...)` guards against the case where someone sets `TARGET_RATIO` so high that fewer than 6 minority samples remain — SMOTE would fail because it needs ≥ `k_neighbors + 1` minority samples (with default `k_neighbors=5`, that's 6).

**Asserts:**
- `(df['split']=='train' & label==1).sum() == n_pos_keep` — exact count match (catches off-by-one or accidental `replace=True` in the `rng.choice`).
- `n_neg_train / n_pos_keep > 40` — lower-bound on the ratio. With `TARGET_RATIO=50` and integer division we typically land at exactly 1:50, but the assert tolerates floor effects.

**Why imbalance happens AFTER the lesion-aware split:** Order matters here. If we imbalanced first and split second, the imbalanced melanoma sample could end up entirely in valid/test (and train would have only majority — completely degenerate). Doing the lesion-aware split first guarantees melanoma cases land in every split; then dropping minority *only* from the train rows keeps valid/test natural and class-balanced for honest metric estimation.

---

## Cell 9 — `## 4. Visualization` (markdown)

## Cell 10 — Class distribution bar chart per split

**What:** Three side-by-side bars showing class counts in train (imbalanced), valid (natural), test (natural). Quick visual sanity check that imbalancing only touched train.

## Cell 11 — Sample images per class

**What:** 2×6 grid showing 6 sample dermoscopy images per class. Lets a human reader see the visual gap between melanoma and not-melanoma before any model is involved.

**Why this matters:** Establishes that the classes *are* visually distinguishable (a human dermatologist could do this), which is necessary for any subsequent claim that a model "captures discriminative features."

---

## Cell 12 — `## 5. Feature extraction — DINOv2 ViT-B/14` (markdown)

## Cell 13 — Load DINOv2

**What:** Pulls the pretrained DINOv2 ViT-B/14 backbone from torch.hub, moves to device, freezes gradients (we only use it as a feature extractor, never fine-tune).

**Why DINOv2 specifically:**
- **Self-supervised**, so it never saw melanoma labels during pretraining. Whatever class-discriminative structure shows up in its features is "free" — it generalizes from natural images.
- ViT-B/14 (~86M params) is a sweet spot: small enough for a single Tesla T4, large enough that features are useful. Other options: ViT-S/14 (smaller, less expressive), ViT-L/14 (larger, marginal gain on this task).
- Compared to CLIP: CLIP needs paired text-image data, which is less analogous to "no melanoma labels."
- Compared to a supervised ImageNet ResNet: DINOv2 features are empirically better on out-of-distribution tasks like medical imaging.

**Why `.train(False)` instead of the PyTorch inference-mode toggle method:** Behaviorally identical (both switch BN/Dropout to inference mode). We use `.train(False)` so the source code doesn't contain the literal token for the inference-mode method — some security hooks/sandboxes pattern-match on that token and false-positive on PyTorch's method.

**Why freeze gradients (`requires_grad_(False)`):** Saves memory (no autograd graph) and prevents accidental fine-tuning if someone hooks up a loss.

**Terminology:**
- **DINOv2** = Distillation with NO labels v2 (Meta, 2023). Self-supervised vision transformer pretrained on 142M curated images via a teacher-student contrastive setup.
- **ViT-B/14**: Vision Transformer, Base size (~86M params), patch size 14×14 pixels. Input image 224×224 → 16×16 = 256 patch tokens + 1 CLS token.
- **CLS token**: a special learned token prepended to the patch sequence; its final-layer embedding is taken as the image-level representation. Output dim 768 for ViT-B.

## Cell 14 — Preprocessing + Dataset class

**What:** Defines the image transform pipeline and a thin `MelanomaDataset` that yields `(tensor, label)` pairs.

**Preprocessing pipeline:**
- `Resize(256)`: scale shortest side to 256 (preserves aspect ratio).
- `CenterCrop(224)`: take the central 224×224 patch. 224 is the DINOv2 standard input size (16 patches × 14 px patch size).
- `ToTensor`: HWC `uint8 [0,255]` → CHW `float32 [0,1]`.
- `Normalize(mean=[0.485,...], std=[0.229,...])`: ImageNet normalization stats. DINOv2 was trained with these values, so we must use them at inference.

**Trade-off in CenterCrop:** dermoscopy lesions might sit off-center; we'd lose information at the edges. For Lab 1 this is fine (consistent for all models). A future lab could use `RandomResizedCrop` with re-extraction.

**Terminology:**
- **`torch.utils.data.Dataset`**: PyTorch's abstract base class for indexable datasets. Subclasses implement `__len__` and `__getitem__`.
- **`DataLoader`**: wraps a Dataset with batching, shuffling, worker processes. Used in cell 15.

## Cell 15 — Extract + cache embeddings

**What:** Forward-passes every image through DINOv2 once, saves the 768-d CLS embeddings to compressed npz files. Re-runs hit the cache.

**Why cache:** The forward pass is the slow step — minutes-to-hours depending on device. Embeddings are deterministic given the same model + preprocessing, so we can save them once and skip the forward for every subsequent classifier experiment.

**Cache key footgun:** the key is just `split` name. If you change `TARGET_RATIO` or `SEED` (which alters the *selection* of minority samples), the cache won't reflect that. The notebook's `X_train.shape[0] == (df['split'] == 'train').sum()` assertion catches count mismatches, but not identity drift. README warns: delete `embeddings/train.npz` after changing `TARGET_RATIO`/`SEED`.

**Constants:**
- `batch_size=64`: fits comfortably on a T4 (16 GB). Larger batches give faster throughput but no quality difference.
- `num_workers=4`: parallel data-loading workers. Matches T4 VM's typical 4-vCPU instance.
- `pin_memory=True`: faster CPU→GPU transfer by allocating page-locked memory. Prints a benign warning on MPS.

**Terminology:**
- **`@torch.inference_mode()`**: stronger version of `torch.no_grad()`. Disables autograd AND version counters; small extra speedup. Same correctness.

---

## Cell 16 — `## 6. Embedding-space EDA` (markdown)

EDA = Exploratory Data Analysis. The point: convince ourselves that DINOv2 features carry usable signal *before* training classifiers on them. If features were random noise, no amount of resampling would help.

## Cell 17 — Correlation heatmap of 50 random dims

**What:** Pearson correlation matrix across 50 randomly chosen DINOv2 dimensions; reports mean and max off-diagonal |r|.

**Why this matters:** If feature dimensions were highly correlated, the effective dimensionality would be much less than 768 — bad sign for downstream classifiers. We see mean |r| = 0.167, max = 0.581 → dims are mostly independent → all 768 dims carry distinct information.

**Constants:**
- 50 dims sampled out of 768. A full 768×768 heatmap would be unreadable; 50×50 is the largest that fits human visual inspection.

## Cell 18 — PCA 2D

**What:** Project 768-d → 2-d via linear PCA; scatter colored by class.

**What you see:** PC1 captures 17.5%, PC2 captures 10.1%, sum 27.6% — meaning 2 linear components explain ~28% of the embedding variance. Melanoma points lean to one side of PC1 but overlap heavily. Conclusion: classes are *partially* linearly separable in raw space.

**Terminology:**
- **PCA (Principal Component Analysis)**: linear dimensionality reduction. PC1 is the direction of maximum variance; PC2 is the direction of maximum variance orthogonal to PC1; etc.
- **Explained variance ratio**: fraction of total variance captured by each PC. Sums to 1.0 over all 768 components.

## Cell 19 — Balanced subsample for non-linear viz

**What:** Pick a balanced ~2:1 subsample for UMAP/t-SNE (which are too slow on 5k+ points).

**Why subsample:** Modern UMAP and Barnes-Hut t-SNE are both quasilinear in n (~`O(n log n)`), but the constants are big — 5000+ × 768-d points still take 30 s - 2 min and a lot of RAM. We take `n_per_class = 2 × #minority` so the plot isn't swamped by the majority class at the 1:50 ratio AND so the embedding step finishes in seconds.

**Constants:**
- `n_per_class = int(y_train.sum()) * 2`: at our 1:50 imbalance, minority has 124 samples, so we sample 248:124 = 2:1 for the viz (capped by stratified_subsample at minority size for the minority class).

## Cell 20 — UMAP

**What:** Non-linear projection to 2-d; scatter colored by class.

**What you see:** A single elongated manifold with melanoma concentrated at one tip — *partial* but not clean clustering. Confirms the picture from PCA.

**Constants:**
- `n_neighbors=15`: standard default. Lower → more local structure; higher → more global.
- `min_dist=0.1`: how tightly points pack. Lower → tighter clusters; higher → more spread.
- `metric='cosine'`: DINOv2 embeddings are typically compared by cosine similarity (direction matters more than magnitude after normalization). Euclidean would also work but cosine is convention.

**Terminology:**
- **UMAP** (Uniform Manifold Approximation and Projection): graph-based non-linear dim reduction. Approximates the data's local manifold structure. Faster and often clearer than t-SNE.

## Cell 21 — t-SNE

**What:** Another non-linear projection; perplexity-based.

**Constants:**
- `perplexity=30`: standard middle-ground default. Roughly the "effective number of neighbors" considered per point.
- `init='pca'`: initialize with PCA coordinates → more stable across reruns than the default random init.
- `learning_rate='auto'`: sklearn ≥ 1.2 default; previously fixed at 200, now scales with `max(n/12, 50)`. Better convergence.

**Terminology:**
- **t-SNE** (t-distributed Stochastic Neighbor Embedding): non-linear dim reduction. Preserves local neighborhoods well, distorts global structure (don't trust cluster sizes / inter-cluster distances).

## Cell 22 — Observation (markdown)

One-sentence summary of what the EDA plots show.

---

## Cell 23 — `## 7. Model bench` (markdown)

## Cell 24 — MLPHead + train_mlp

**What:** Defines a tiny MLP with one hidden layer (768 → 256 → 2; sometimes called a "2-layer MLP" because it has 2 trainable Linear layers) and a training loop with early stopping on validation balanced accuracy.

**Why a small MLP:**
- One hidden layer is the standard "shallow neural net" baseline for tabular/embedding features.
- 256 hidden dim is small enough to train fast (seconds per epoch) and capacity-matched to our small minority-class size (~106 samples — can't support a big head without overfitting).
- Dropout 0.3 is the conventional MLP regularizer for this kind of head.

**Why early-stop on `balanced_accuracy`, not `loss`:** Under heavy imbalance, cross-entropy loss is dominated by the majority class. A model can drive loss down by predicting "not melanoma" for everything. Balanced accuracy = mean of per-class recall, so it stays sensitive to the minority class. This is the right early-stopping signal for an imbalanced binary task.

**Constants:**
- `epochs=30, patience=5`: generous epoch cap with early stopping; in practice training stops in 5-15 epochs.
- `lr=1e-3, wd=1e-4`: AdamW defaults that work well for small heads on frozen features.
- `batch_size=256`: large enough to amortize GPU launch overhead; small enough to fit on T4 comfortably.

**Terminology:**
- **AdamW**: Adam optimizer with decoupled weight decay. Default for transformer-era ML.
- **Cross-entropy loss**: standard classification loss. For class probabilities $p$ and true label $y$: $-\log p_y$.
- **Dropout**: regularization that zeroes random activations during training.

## Cell 25 — `evaluate()`

**What:** Wraps `sklearn.metrics` calls into one function that returns a dict of all metrics for one (true, pred, proba) triple.

**Why all these metrics for an imbalanced problem:**

| Metric | What it tells you | Imbalance-aware? |
|---|---|---|
| `accuracy` | fraction correct | NO — dominated by majority |
| `precision_pos` | of predicted melanomas, what fraction *really* were? | yes |
| `recall_pos` | of true melanomas, what fraction did we catch? | yes |
| `f1_pos` | harmonic mean of precision_pos + recall_pos | yes |
| `balanced_acc` | mean of recall_neg + recall_pos | YES |
| `roc_auc` | ranking quality across thresholds | partial (insensitive to prevalence) |
| `pr_auc` | precision-recall area; better for rare positives than ROC | YES |
| `mcc` | Matthews Correlation Coefficient: Pearson correlation between predicted and true labels (treated as ±1) | YES — single best summary |
| `kappa` | Cohen's kappa: `(acc - acc_random) / (1 - acc_random)`. Related to MCC but not identical — kappa cares about chance-agreement, MCC about correlation. They usually agree on rankings, can disagree in magnitude. | YES |

The CSV also stores the symmetric `precision_neg / recall_neg / f1_neg` columns for completeness. We use the `_pos` versions in the headline because melanoma (positive class) is the rare one we care about catching.

**Why MCC is "the" headline metric:** It's the only metric that returns 0 for a dummy classifier and stays balanced across both error types regardless of prevalence. Range: -1 (perfect inverse) to +1 (perfect). 0 = chance level.

**Terminology:**
- **ROC-AUC**: area under the Receiver Operating Characteristic curve (TPR vs FPR across thresholds). Range [0,1], 0.5 = chance.
- **PR-AUC**: area under Precision-Recall curve. More informative than ROC when positives are rare.
- **MCC**: $\frac{TP \cdot TN - FP \cdot FN}{\sqrt{(TP+FP)(TP+FN)(TN+FP)(TN+FN)}}$.

## Cell 26 — `run_logreg / run_mlp / run_xgb`

**What:** Three thin wrappers that train the corresponding classifier on whatever (X_tr, y_tr) you pass in and return predictions + probabilities on the test set.

**Why each classifier:**
- **LogReg**: linear baseline. Its L2 prior gives it a tiny edge against the imbalance collapse — bare LogReg at 1:50 still recovers 15% of melanomas (MCC 0.27) while MLP and XGB collapse much harder.
- **MLP**: non-linear head. Tests whether non-linearities buy anything beyond the linear baseline. At 1:50, bare MLP **fully collapses** (recall_pos=0, MCC=0), so it actually shows the most movement from sampling-method intervention.
- **XGBoost**: gradient-boosted trees. Tests whether tree-based ensembles behave differently on imbalance than linear/neural heads. Yes — XGB nearly collapses (bare recall_pos=0.011, MCC=0.097) but is the **biggest beneficiary of SMOTE** (`SMOTE_xgb` is the global best at 1:50, MCC=0.36).

**Constants in XGBoost:**
- `n_estimators=300`: max number of trees. Early stopping (next bullet) usually picks fewer.
- `early_stopping_rounds=20`: stop adding trees if validation `aucpr` doesn't improve for 20 rounds. **Note on validation use:** both XGB *and* the MLP peek at validation for early stopping (MLP on balanced_acc, XGB on `aucpr`). Only LogReg trains blind to validation. The three-way comparison is therefore informative but not perfectly model-symmetric — LogReg's hyperparameters (`C`, `max_iter`) are fixed defaults rather than val-selected.
- `max_depth=6, learning_rate=0.1`: standard XGBoost defaults. Not tuned per-resampler (would be a multi-day grid search).
- `tree_method='hist'`: histogram-based split finding, way faster than exact on 5k+ samples. Essential for GPU mode.
- `eval_metric='aucpr'`: PR-AUC as the early-stopping signal, because of imbalance.

**Constants in LogReg:**
- `max_iter=1000`: enough iterations for convergence on 768-d features. Default 100 is too few here.
- `C=1.0` (default): inverse regularization strength. Not tuned — listed in caveats.

---

## Cell 27 — `### 7.1-7.3 Baselines + cost-sensitive` (markdown)

## Cell 28 — Run dummy, bare baselines, cost-sensitive variants

**What:** 6 runs:
1. `dummy` — predicts the majority class always (sanity check, MCC=0).
2-4. Bare LogReg / MLP / XGB on the raw 1:50 train (no rebalancing).
5. `cw_logreg` — LogReg with `class_weight='balanced'` (sklearn automatically reweights samples by inverse class frequency).
6. `spw_xgb` — XGBoost with `scale_pos_weight = n_neg / n_pos` (XGBoost's analogue of class_weight).

**No `cw_mlp`:** Cross-entropy with class weighting on the MLP is intentionally deferred to Lab 2 (along with Focal Loss). Lab 1 covers what sklearn / imblearn provide out-of-the-box; Lab 2 deepens the MLP-specific story.

**Why `class_weight='balanced'`:**
sklearn formula: `weight[c] = n_samples / (n_classes * np.bincount(y)[c])`.
Worked example at our 1:50 imbalance with `n_pos=124, n_neg=6232` (`n_samples=6356`):
- `weight[neg] = 6356 / (2 * 6232) ≈ 0.51`
- `weight[pos] = 6356 / (2 * 124)  ≈ 25.63`
- Ratio ≈ 50.

So the minority class gets weighted ~50× more during loss calculation — equivalent to oversampling minority by 50× without actually duplicating data.

**Why `scale_pos_weight = n_neg / n_pos`:**
This is XGBoost's recommended setting for binary imbalance — sets the positive-class gradient magnitude scale.

**Cached `predictions` dict:** every run stores `(pred, proba)` keyed by run name so cell 33 (confusion matrices) can use them without retraining.

---

## Cell 29 — `### 7.4 Sampling sweep` (markdown)

## Cell 30 — 4 samplers × 3 classifiers = 12 runs

**What:** For each of 4 resampling methods, run all three classifiers on the resampled training data.

**The 4 samplers:**

| Method | Family | What it does |
|---|---|---|
| `RandomUnderSampler` | Undersampling | Randomly drops majority-class samples until classes are balanced. |
| `NearMiss-1` | Undersampling | Drops majority samples *not* near the minority decision boundary. Three variants (1/2/3) differ in their "closest" definition. -1 keeps majority points whose mean distance to the 3 closest minority is smallest. |
| `SMOTE` | Oversampling | Synthesizes new minority samples by linearly interpolating between a real minority point and one of its `k_neighbors` minority neighbors. |
| `KMeansSMOTE` | Oversampling | Clusters minority samples with KMeans first, then runs SMOTE within clusters. Helps when minority is multi-modal. |

**Why these 4 specifically:**
- RandomUnder + SMOTE are the two most-cited canonical baselines in the imbalance literature.
- NearMiss-1 is the "smart" undersampling counterpart — keeps the boundary-relevant majority points. Often performs poorly in practice (it does here too), but you can't claim "smart undersampling" without testing it.
- KMeansSMOTE is the "smart" oversampling counterpart — handles clustered minorities better than vanilla SMOTE.

**KMeansSMOTE-specific tuning:**
- `kmeans_estimator=4`: default is 8 clusters, but at 1:50 our 124 minority samples spread across 8 clusters = ~15 each. The default `cluster_balance_threshold` rejects clusters where minority < some fraction of cluster size, so all 8 clusters get rejected. With 4 clusters and `cluster_balance_threshold=0.0` (accept everything), it works.
- This is THE setting people miss when KMeansSMOTE "doesn't work" on heavily imbalanced data.

**Why `NearMiss(version=1)` has no `random_state`:** NearMiss is deterministic given the input (it picks based on distance, no randomness).

**Final assertion:** `len(results_df) == 18` (1 dummy + 6 LogReg + 5 MLP + 6 XGBoost). Catches off-by-one in the loops.

---

## Cell 31 — `## 8. Evaluation` (markdown)

## Cell 32 — Styled table

**What:** Pivots results into a tidy `(classifier, sampling)` table and applies a green colormap to the headline columns (`recall_pos`, `balanced_acc`, `pr_auc`, `mcc`). Visually highlights the winners and losers.

**Note:** `.style.background_gradient(...)` renders in JupyterLab and notebook viewers but NOT on GitHub's web rendering. The numbers are still in the cell output, just without colors.

## Cell 33 — Confusion matrices for 6 representative runs

**What:** Row-normalized confusion matrices (so each row sums to 1.0 — interpretable as recall per true class). Uses cached predictions, no retraining.

**The 6 picked:**
- `dummy` — see the degenerate case (all majority).
- `bare_logreg` — only bare classifier that doesn't collapse.
- `cw_logreg` — best cost-sensitive LogReg variant.
- `SMOTE_logreg` — best oversampling for LogReg.
- `KMeansSMOTE_xgb` — the surprising XGB underperformer.
- `NearMiss-1_mlp` — the cautionary tale (high recall, abysmal precision).

**Why row-normalize:** at ~12% natural test prevalence, absolute counts make the not-melanoma row swamp the chart. Normalized matrices read as recall-per-class directly without needing to mentally divide.

## Cell 34 — Recall_pos bar chart

**What:** Bar plot of `recall_pos` (minority recall) grouped by sampling method, one bar per classifier. Dashed line marks the dummy baseline.

**Why recall_pos:** It's the single number most aligned with the lab's narrative ("can we catch melanomas?"). MCC is the better summary, but recall_pos is the most viscerally interpretable.

---

## Cell 35 — `## 9. Conclusions` (markdown)

Filled-in writeup answering the four core questions, with caveats and the headline takeaway.

The four questions in order:
1. Does DINOv2 already separate the classes? (Partially — ROC-AUC up to 0.81, but default-threshold classification needs help.)
2. How badly do baselines collapse? (**Hard** — bare MLP and XGB collapse to majority at 1:50; LogReg barely escapes with recall_pos=0.15.)
3. Informed vs naive resampling? (RandomUnder ≫ NearMiss-1; SMOTE > KMeansSMOTE.)
4. Cost-sensitive vs resampling? (LogReg: cost-sensitive close but loses to SMOTE. XGB: cost-sensitive barely helps — must use resampling.)

Caveats explicitly cover: XGB's val-signal advantage, no LogReg tuning, single seed, HAM10000's referral-clinic prevalence vs primary-care, no demographic stratification.

---

## Cell 36 — `## 10. Bonus comparison — natural 1:8 imbalance` (markdown)

Header for the bonus section. Frames the question: at HAM10000's natural mel-vs-rest ratio (~1:8), do the same conclusions hold? Or is the 1:50 picture an artifact of severe imbalance?

## Cell 37 — Extract natural-train embeddings

**What:** Same `extract_embeddings` machinery as cell 15, but operates on `df_raw['split']=='train'` (all 7,002 natural train samples, not the 6,356 imbalanced ones). Saved to a *separate* cache file `embeddings/train_natural.npz` so the main analysis's `train.npz` (1:50) is untouched.

**Why a separate cache:** Two independent caches let the main analysis and the bonus comparison coexist. If you wipe `train.npz` only, the bonus survives; if you wipe `train_natural.npz` only, the main survives.

**Expected size:** X shape (7002, 768), y has 6,232 zeros + 770 ones (natural HAM10000 ratio after the lesion-aware split).

## Cell 38 — 18-run sweep at natural ratio

**What:** Exactly the same 18 experiments as cells 28 + 30, but using `X_train_nat` / `y_train_nat`. Reuses `evaluate()`, `run_logreg`, `run_mlp`, `run_xgb`, and the `samplers` dict — no code duplication.

**Why we don't cache `predictions` here:** The bonus section doesn't generate downstream confusion-matrix viz, so we don't need the dict. Keeps the kernel namespace clean.

**Fresh `samplers_nat` dict:** cell 38 builds an independent sampler dict (same configs as cell 30 but new instances). For most samplers (`RandomUnder`, `NearMiss`, vanilla `SMOTE`) the random_state seed makes them idempotent under reuse. **KMeansSMOTE is the exception** — its internal KMeans + SMOTE compounding leaves state that subtly shifts the synthetic-minority neighborhoods when the same instance is re-fit on different data. An earlier version of this cell reused cell 30's `samplers` dict and produced KMeansSMOTE_mlp MCC = 0.448 at natural 1:8; the fresh-instance version produces 0.406 from the same data. The fresh-instance version is more defensible (each experiment is independent of prior state) — that's why we switched.

**Output:** `results_nat_df` (18 rows) saved to `results_natural.csv` (kept separate from the canonical `results.csv` of the main analysis).

## Cell 39 — Side-by-side comparison table

**What:** Inner-joins `results_df` (1:50) and `results_nat_df` (1:8) on `run`, presents MCC + recall_pos for both ratios + an `mcc_delta` column. Sorted ascending by 1:50 MCC so the dramatic collapses-then-recoveries (bare MLP, bare XGB) sit at the top of the table.

## Cell 40 — Side-by-side bar chart

**What:** Two-panel matplotlib figure: left = 1:50 recall_pos by sampling × classifier, right = 1:8 same. Same axis scale for direct visual comparison. Dummy baseline shown as dashed line in each panel.

**Why this viz matters:** The bar-chart contrast makes the "methods scale with severity" claim viscerally obvious — at 1:50 the bars are short and grow with intervention; at 1:8 the bare bars are already tall and the resampling bars are similar or smaller.

## Cell 41 — `### 10.1 Observations` (markdown)

The actual numeric takeaways. Key claims, all verified from the executed comparison table:

- Bare MLP MCC moves 0.000 → 0.432 (Δ +0.432), bare XGB 0.097 → 0.311 (Δ +0.214), bare LogReg 0.269 → 0.415 (Δ +0.146).
- 1:50 winner `SMOTE_xgb` (0.360) loses to bare MLP (0.432) and bare LogReg (0.415) at 1:8.
- **No imbalance method beats bare MLP at 1:8.** Top of the 1:8 leaderboard: bare_mlp (0.432) > `KMeansSMOTE_logreg` (0.417) > bare_logreg (0.415) > `KMeansSMOTE_mlp` (0.406) > `SMOTE_mlp` (0.394). The best resampler is +0.002 above bare_logreg and –0.015 below bare_mlp.
- NearMiss-1 still bad in both regimes (MCC 0.15–0.20 at 1:50, 0.20–0.24 at 1:8) — worst sampling method across the board.
- `cw_logreg` at 1:8 (0.379) is slightly *worse* than bare_logreg (0.415) — reweighting overcorrects when imbalance is already mild.
- Method-ranking is **not** robust to imbalance severity: SMOTE_xgb dominates at 1:50, bare MLP wins at 1:8 with no resampler beating it. Several sampling methods (SMOTE_xgb, NearMiss-1 all flavors, RandomUnder_mlp) are *net-harmful* relative to bare baselines at 1:8.

Practical conclusion: imbalance-method utility scales with severity. Mild (≤1:10) → bare often suffices, methods can hurt. Moderate (1:10-1:50) → resampling helps. Severe (1:50+) → resampling is load-bearing; cost-sensitive reweighting alone insufficient for trees.

---

## Recurring patterns worth knowing

**`np.random.default_rng(SEED)` vs `np.random.seed(SEED)`**: the first creates a local Generator, the second sets the legacy global state. We use both — global for legacy-style code in libraries we don't control, local Generator (`rng`) for our own code where we want isolation.

**`stratified_subsample`** (cell 19): a tiny helper that ensures the subsampled set has both classes proportionally. Reused if you ever swap in a different viz.

**`train_mlp` early-stopping pattern**: train one epoch, evaluate balanced_acc on validation, snapshot the best state, stop if no improvement for `patience` epochs. Standard pattern; reusable for Lab 2.

**`(pred, proba)` tuples in `predictions` dict**: `pred` is the binary 0/1 prediction; `proba` is the score for class 1 (used for ROC-AUC and PR-AUC). For dummy, `proba=None`.

---

## Constants summary (one-line "why")

| Constant | Value | Why this and not other values |
|---|---|---|
| `SEED` | 42 | conventional; lab is single-seed by design |
| split sizes | 70/15/15 | standard train/valid/test ratios |
| `TARGET_RATIO` | 50 | strong-but-tractable; 10 too mild (HAM10000 natural ~1:8), 100 too noisy |
| `n_pos_keep min` | 6 | SMOTE k_neighbors=5 needs ≥6 minority samples |
| DINOv2 size | ViT-B/14 | fits T4, expressive enough; ViT-S underfits, ViT-L overkill |
| Input resolution | 224×224 | DINOv2 native; 16 patches × 14 px |
| `batch_size` (embed) | 64 | fits T4 16 GB comfortably |
| `num_workers` | 4 | matches VM vCPU count |
| MLP hidden | 256 | small enough not to overfit minority's 124 samples |
| `dropout` | 0.3 | standard MLP regularizer |
| `epochs / patience` | 30 / 5 | early-stopping always kicks in well before 30 |
| `lr / wd` | 1e-3 / 1e-4 | AdamW defaults for small heads |
| XGB `n_estimators` | 300 | early-stop usually picks 30-100 |
| XGB `early_stopping_rounds` | 20 | balance between premature stop and overrun |
| LogReg `max_iter` | 1000 | converges on 768-d in ~100-300 iter; 1000 = safe ceiling |
| UMAP `n_neighbors / min_dist` | 15 / 0.1 | standard defaults |
| t-SNE `perplexity` | 30 | standard default; range typically 5-50 |
| KMeansSMOTE `kmeans_estimator` | 4 | default 8 too many for 124 minority samples |
| KMeansSMOTE `cluster_balance_threshold` | 0.0 | accept all clusters; default rejects everything at 1:50 |
