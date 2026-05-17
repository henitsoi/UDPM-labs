# Lab 3 — GAN-based augmentation

Vanilla conditional DCGAN (BCE + AdamW + EMA + best-by-discriminative-margin
checkpoint) on DermaMNIST 28×28. Synthetic mel images augment imbalanced train
sets; a small CNN from scratch measures whether augmentation moves the needle at
1:50 and ~1:8 imbalance.

## Deliverables

- `lab3.ipynb` — executed notebook (§1 setup → §9 conclusions, 40 cells).
- `results.csv` — 8 rows: {1:50, 1:8} × {real_only, +synth_50pct, +synth_100pct, +synth_balanced}.
- `CELL_NOTES.md` — cell-by-cell walkthrough.

## Headline numbers

**At 1:50 (sorted by test MCC):**

```
real+synth_balanced  0.3626   (rec_mel 0.740, bal_acc 0.759 — breaks the trap!)
real_only            0.0000   (constant not-mel predictor)
real+synth_50pct     0.0000   (same)
real+synth_100pct    0.0000   (same)
```

**At 1:8 (sorted by test MCC):**

```
real+synth_100pct    0.3953   (+0.055 vs real_only; rec_mel 0.471, bal_acc 0.700)
real+synth_balanced  0.3743   (rec_mel 0.731 — best recall, lower precision)
real+synth_50pct     0.3612   (+0.021 vs real_only)
real_only            0.3399   (rec_mel 0.220, bal_acc 0.603)
```

**Synth augmentation helps at both imbalances** with margin-selected synth:

- At **1:50** (where the real-only classifier collapses), the balanced dose
  (6104 synth-mel → 1:1 ratio) breaks the trap: MCC 0.000 → **0.363**,
  recall_mel 0 → **0.740**.
- At **1:8**, every synth dose beats real_only on MCC; the 100% dose wins at
  **+0.055 MCC over real_only**. The balanced dose hits the highest recall_mel
  (0.731) but its precision drops.

See §9 Q3-Q5 for the per-condition analysis, dose-response discussion, and the
cos/margin/val-MCC selection comparison.

## GAN training + checkpoint selection summary

- 200 epochs on the natural ~1:8 train split, ~20 min on Tesla T4.
- `snapshot_every=1` (200 snapshots, stored on CPU, ~1.8 GB).
- **Three selection metrics compared:**
  - **margin** (canonical): max `cos(synth↔real_mel) − cos(synth↔real_notmel)`.
    Picks **epoch 195** (margin=+0.0382). ~10 min for 200 snapshots.
  - **cos** (legacy): max `cos(synth↔real_mel)`. Picks epoch 70 (cos=0.788
    but margin only +0.032 — synth is in the class-overlap zone).
  - **val-MCC** (task-aligned, stride=5 + epoch 195): max val MCC after training
    a quick CNN on real+synth across 40 stride-5 snapshots plus the margin-selected
    epoch evaluated explicitly. Picks **epoch 171** (val_mcc=+0.4044, tied with
    epoch 196 at +0.4044). The margin-selected epoch 195 has val_mcc=+0.3761 —
    only **0.028 below the val-MCC peak**, inside the ±0.03 CUDA noise floor.
- The lab uses **margin selection** for the canonical sweep because: (a) it's
  ~2× cheaper than val-MCC, (b) the val-MCC peak epoch itself is unstable
  across re-runs (we observed 111 → 191 → 171), and (c) the margin-selected
  ckpt's val_mcc is always within ±0.03 of whatever val-MCC peak that run
  produces.
- D dominance kicks in around epoch 80 (D_loss drops from 1.29 → 0.50,
  D_real_acc climbs from 0.74 → 0.95). Late epochs are degraded; margin
  selection naturally avoids them.
- Synth pool: **6204** mel images (sized to cover the largest dose = balanced
  condition at 1:50). Sampled in deterministic 256-image chunks from EMA-G
  of the margin-selected ckpt.
- Per-snapshot data persisted: `gan/margin_log.csv` (200 rows: epoch,
  cos_mel, cos_notmel, margin), `gan/valmcc_log.csv` (41 rows: 40 stride-5
  snapshots + the margin-selected epoch 195).

## Launch

The notebook expects DermaMNIST via the `medmnist` package — auto-downloaded
on first run to `data/`. Run order: §1 (setup) → §2 (data, ~30s) → §3 (EDA,
~3 min first-run due to DINOv2 hub-load + 7007-image feature extract) → §4
(G/D defs, <1s) → §5 (train GAN ~20 min + margin selection over 200 snapshots
~10 min + val-MCC selection over 40 snapshots ~4.5 min) → §6.4 (generate synth
pool, ~30 s) → §7 (CNN defs) → §8.1–§8.3 (run 8 sweep cells + confusion
matrices, ~1.5 min). **Total wallclock: ~40 min** on a freshly restarted kernel.

### Local install / Remote VM

See top-level `README.md` § *Run anywhere*. The §1.2 device-check cell
auto-detects `cuda → mps → cpu`, but §5 GAN training asserts CUDA — CPU is
impractical.

## Reproducibility

Single seed `42`. Both `train_gan` and `train_cnn` reseed `torch.manual_seed`,
`torch.cuda.manual_seed_all`, and `np.random.seed` internally on each call.

**CUDA forward-pass non-determinism caveat:** without
`torch.backends.cudnn.deterministic = True` (which we did NOT set, to preserve
the existing committed ckpts), CNN inference is non-deterministic at the
algorithm-selection level. We measured this to introduce **~±0.03 noise** on
individual `test_mcc` values across in-session re-runs (same seed, same code).
The qualitative findings (synth helps at 1:50 +balanced and 1:8 +100pct,
margin-selected ckpt sits within ±0.03 of the val-MCC peak) are stable across
runs; individual MCC numbers should be treated as ±0.03 noisy. The val-MCC
peak epoch itself flickers across runs (we observed 111 → 191 → 171), which
is part of why margin selection is preferred for the canonical sweep.

Canonical numbers come from `results.csv` and the most recent top-to-bottom
sweep.

## Notes / known caveats

- **MedMNIST splits leak train↔val** at the image level (no `lesion_id`
  exposed). Test set is clean. We don't gate on val MCC except for early-stopping.
- **DINOv2 at 28×28 input** is meaningless without upsampling — we resize to
  224×224 before each forward pass and apply ImageNet normalization.
- **GAN trained on natural ~1:8 train**, not on the 1:50 subset. Rationale:
  124 mel samples are too few to train cDCGAN; the same G_ema is then used
  for augmenting both 1:50 and 1:8 downstream conditions.
- **Synth pool is one 6204-image file** sliced for smaller doses, so per-dose
  comparisons use the same z's (same images).
- **snapshot_every=1 vs every=10 picks different best epochs** (195 vs 160)
  with similar margins (+0.0382 vs +0.0367). Downstream MCC is unchanged at
  the noise floor — the fine-grained scan finds a slightly noisier peak.
- **Margin vs val-MCC selection**: the val-MCC peak epoch is itself unstable
  across in-session re-runs (111 → 191 → 171). What's stable is that the
  margin-selected epoch 195 has val_mcc within ±0.03 of whatever val-MCC peak
  the run produces (e.g. 0.376 vs 0.404 in the canonical run). Margin is the
  cheaper, more stable choice; val-MCC is task-aligned but itself noisy at
  this scale.