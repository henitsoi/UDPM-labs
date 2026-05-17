# Lab 3 — cell-by-cell walkthrough

This file documents each of the 40 cells in `lab3.ipynb` in execution order.
Cell indices are 0-based notebook position (not the `[N]:` execution counter
shown by JupyterLab).

## §0 — Title (cell 0)

**Cell 0 (markdown):** Title + 1-paragraph project description.

## §1 — Setup (cells 1-4)

**Cell 1 (markdown):** `## §1 Setup` header.

**Cell 2 (code, §1.1):** Imports — torch, medmnist, sklearn, umap, matplotlib.

**Cell 3 (code, §1.2):** Device check — picks cuda → mps → cpu. On the T4 VM
prints `cuda: Tesla T4`.

**Cell 4 (code, §1.3):** Sets `SEED=42`, seeds torch + numpy. Creates output
directories: `data/`, `gan/`, `gan/samples/`, `synth/`, `dinov2_cache/`.

## §2 — Data (cells 5-9)

**Cell 5 (markdown):** §2 header.

**Cell 6 (code, §2.1):** DermaMNIST download + binarize to mel vs not-mel.
Natural train: **779 mel / 6228 not-mel** ≈ 1:8 ratio.

**Cell 7 (code, §2.2):** Natural ~1:8 variant — uses binarized train as-is.

**Cell 8 (code, §2.3):** Artificial 1:50 variant — subsamples mel to
`n_neg // 50 = 124` images. Ratio = 1:50.23.

**Cell 9 (code, §2.4):** Saves all four splits to `splits_cache.pt` (~154 MB).

## §3 — EDA (cells 10-15)

**Cell 10 (markdown):** §3 header.

**Cell 11 (code, §3.1):** Class-counts table + horizontal bar chart.

**Cell 12 (code, §3.2):** 4×8 grid of mel and not-mel real samples. Defines
`to_display(x)` for `[-1,1] → [0,1]` matplotlib conversion.

**Cell 13 (code, §3.3):** PCA + t-SNE + UMAP on flattened raw pixels (balanced
500/500 subset). Three-panel scatter.

**Cell 14 (code, §3.4):** Per-class mean image + pixel-wise label correlation.
Max abs pixel-corr ≈ 0.125.

**Cell 15 (code, §3.5):** Loads DINOv2 ViT-B/14 from torch.hub. Defines
`dinov2_extract(x)` (resizes 28→224, ImageNet-normalizes, forwards). Caches
7007 natural-train embeddings to `dinov2_cache/train_real.npz` (~22 MB).
Computes real-mel centroid (L2-normalized).

## §4 — Generator + Discriminator (cells 16-19)

**Cell 16 (markdown):** §4 header.

**Cell 17 (code, §4.1):** Generator (~454k params).

**Cell 18 (code, §4.2):** Discriminator (~433k params).

**Cell 19 (code, §4.3):** Forward sanity tests with local `*_sanity` names.

## §5 — GAN training + best-ckpt selection (cells 20-25)

**Cell 20 (markdown):** §5 header.

**Cell 21 (code, §5.1):** `ema_update(...)` and `class_balanced_loader(...)`.

**Cell 22 (code, §5.2):** `train_gan(...)`. Reseeds torch + cuda + numpy at
entry. AdamW G+D, BCE loss, EMA after each step. **Stores snapshots on CPU**
(to support `snapshot_every=1` with 200 snapshots × 3 sd × ~3 MB ≈ 1.8 GB).
Per-epoch logging of G/D losses + dinov2_cos. PNG sample-grid every 10 epochs
(decoupled from snapshot frequency).

**Cell 23 (code, §5.5):** `save_best_ckpt(snapshots, log_rows, gan_dir,
metric='margin', ...)`. For `metric='margin'` (default): per-snapshot
DINOv2 eval, picks max `cos(synth, real_mel) − cos(synth, real_notmel)`.
Writes `gan/margin_log.csv` (per-snapshot epoch/cos_mel/cos_notmel/margin).
Saves `ckpt_G.pt`, `ckpt_G_ema.pt`, `ckpt_D.pt`. Handles CPU state_dicts
by moving to DEVICE on load.

**Cell 24 (code, §5.6):** Executes training on natural train with
`snapshot_every=1`. **20 min wallclock for training + 10 min for margin
eval over 200 snapshots = ~30 min total.** Calls save_best_ckpt with
metric='margin'. Output: `best ckpt by margin: epoch 195 (score=+0.0382)`.

**Cell 25 (code, §5.7):** `select_by_val_mcc(snapshots, ...)` — the
task-aligned alternative to margin selection. For each snapshot (sampled
every 5), generates synth pool, trains a quick CNN (10 epochs) on
real+synth, measures val MCC. Then explicitly evaluates the margin-selected
epoch (195) so the two metrics can be compared head-to-head at the same
snapshot. Writes `gan/valmcc_log.csv` (41 rows). **~4.5 min wallclock**
for 41 snapshots. Output: `best by val_mcc: epoch 171 (val_mcc=+0.4044)`,
`best by margin: epoch 195 (margin=+0.0382, val_mcc=+0.3761)` — margin
ckpt's val_mcc is within 0.028 of the peak, inside the ±0.03 CUDA noise
floor. Keeps the margin-selected ckpt on disk; this cell is for §9's
analysis comparison only.

## §6 — GAN evaluation (cells 26-30)

**Cell 26 (markdown):** §6 header.

**Cell 27 (code, §6.1):** Loss + D-accuracy curves from `gan/train_log.csv`.

**Cell 28 (code, §6.2):** Sample-grid montage at landmark epochs 10, 50,
100, 150, 200.

**Cell 29 (code, §6.3):** Two figures and three scalar margins:
- **§6.3a-left:** per-epoch dinov2_cos trajectory over 200 epochs.
- **§6.3a-right:** per-snapshot margin trajectory (200 points from
  `gan/margin_log.csv`) — **the metric used for selection**. Peaks at
  epoch 195.
- **§6.3b:** PCA + UMAP overlay of (real_mel + real_notmel + synth_mel)
  DINOv2 embeddings, 500 samples each.
- **Scalar margins (printed):** synth_mel cos to real_mel centroid ≈ 0.77,
  to real_notmel centroid ≈ 0.74, margin ≈ +0.03.

**Cell 30 (code, §6.4):** Loads `ckpt_G_ema.pt` (margin-selected, epoch 195),
generates **6204 synth-mel** images in 256-image chunks. Saves to
`synth/mel_synth.pt` (~58 MB). Visualizes 32 samples.

## §7 — Downstream CNN (cells 31-34)

**Cell 31 (markdown):** §7 header.

**Cell 32 (code, §7.1):** `build_small_cnn(num_classes)` — **68,098 params.**

**Cell 33 (code, §7.2):** `train_cnn(...)` — reseeds torch + cuda + numpy.
AdamW lr=1e-3, batch=64, 30 epochs max, early-stop on val MCC patience 5.

**Cell 34 (code, §7.3):** `evaluate_on_test(model, X_test, y_test)` returns
column-schema dict with test_mcc, test_acc, test_bal_acc, test_rec_pos,
test_prec_pos, test_f1_pos.

## §8 — Sweep (cells 35-38)

**Cell 35 (markdown):** §8 header.

**Cell 36 (code, §8.1):** Defensively reloads splits + synth pool.

**Cell 37 (code, §8.2):** Runs **8 experiments** — 2 imbalances × 4 conditions
(real_only, +synth_50pct, +synth_100pct, +synth_balanced). Stores trained
models in `trained_models` dict for §8.3. Writes `results.csv`.

**Cell 38 (code, §8.3):** Confusion matrix grid for all 8 conditions. Title
MCC from `results.csv`. Per-class summary printed (rec_notmel, rec_mel,
overall_acc).

## §9 — Conclusions (cell 39)

**Cell 39 (markdown):** Q1-Q5 written analysis grounded in evidence:
- **Q1 (GAN convergence):** Yes. Three selection metrics compared:
  cos → epoch 70 (max cos=0.788 but max margin only +0.032);
  margin → epoch 195 (margin=+0.0382, canonical, val_mcc=+0.3761);
  val-MCC (stride=5 + epoch 195) → epoch 171 (val_mcc=+0.4044, tied with
  epoch 196). **Margin-selected ckpt's val_mcc sits 0.028 below the
  val-MCC peak — inside the ±0.03 CUDA noise floor.** The val-MCC peak
  epoch itself is unstable across re-runs (111 → 191 → 171), which is part
  of why margin selection is preferred for the canonical sweep.
- **Q2 (synth-mel in-distribution?):** Modestly — DINOv2 margin ≈ +0.03.
- **Q3 (help at 1:50?):** Yes at the balanced dose — MCC jumps from 0.000 to
  **0.363** (rec_mel 0.740).
- **Q4 (help at 1:8?):** Yes — +synth_100pct wins by **+0.055 MCC** over
  real_only (0.395 vs 0.340).
- **Reproducibility caveat:** ±0.03 noise from CUDA forward non-determinism
  (no `cudnn.deterministic=True` set). Qualitative findings stable.
- **Q5 (failure mode + fix):** 4 ranked fixes: enable `cudnn.deterministic`
  + multi-seed, combine balanced synth + weighted loss at 1:50, use val-MCC
  selection on a held-out split (current impl reuses early-stopping val),
  per-image curation by DINOv2 margin.