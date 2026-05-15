# Lab 2 — Weighted Loss Functions

Hand-implemented class-weighted cross-entropy, Focal Loss, and Class-Balanced
Focal Loss on the same HAM10000 DINOv2 embeddings as Lab 1. Mirrors Lab 1's
two-imbalance structure (artificial 1:50 + natural ~1:8).

## Deliverables

- `lab2.ipynb` — executed notebook with hand-rolled loss code, math, plots, conclusions.
- `results.csv` — 51 rows at artificial 1:50 imbalance (1 bare + 12 WeightedCE + 18 Focal + 20 CB-Focal).
- `results_natural.csv` — 51 rows at natural ~1:8 imbalance, same grid.
- `CELL_NOTES.md` — cell-by-cell walkthrough.

## Headline numbers

**Top 5 at 1:50 (sorted by test MCC):**

```
WeightedCE {"w_neg": 1, "w_pos": 10.0}                                       MCC=0.3837
WeightedCE {"w_neg": 1, "w_pos": 15.0}                                       MCC=0.3702
WeightedCE {"w_neg": 1, "w_pos": 5.0}                                        MCC=0.3628
CBFocal    {"beta": 0.999, "gamma": 0.5, "class_counts": [6232, 124]}        MCC=0.3607
Focal      {"alpha": [1.0, 50.26], "gamma": 0.25}                            MCC=0.3498
```

(The val-MCC-selected WeightedCE winner is `w_pos=50.0` at MCC=0.3480; the val
rule picks a different operating point than the raw-test argmax. See §9 Q2.)

**Top 5 at 1:8 (sorted by test MCC):**

```
CBFocal    {"beta": 0.99, "gamma": 0.5, "class_counts": [6232, 770]}         MCC=0.4595
Focal      {"alpha": [1.0, 1.0], "gamma": 2.0}                               MCC=0.4475
CBFocal    {"beta": 0.9, "gamma": 2.0, "class_counts": [6232, 770]}          MCC=0.4475
BareCE     {"w_neg": 1, "w_pos": 1}                                          MCC=0.4452
WeightedCE {"w_neg": 1, "w_pos": 1.0}                                        MCC=0.4452
```

(BareCE and `WeightedCE w_pos=1.0` are numerically identical — same loss,
just labelled differently in the grid. Both tie at 0.4452.)

## Family winners (val-MCC-selected, not test-cherry-picked)

The per-family winners reported in §8 use `val_mcc_at_stop`, not test MCC, to
avoid test-set selection bias:

| imbalance | family     | hyperparams                  | val_mcc | test MCC |
|-----------|------------|------------------------------|---------|----------|
| 1:50      | BareCE     | w_pos=1                      | 0.0000  | 0.0000   |
| 1:50      | WeightedCE | w_pos=50.0                   | 0.3783  | 0.3480   |
| 1:50      | Focal      | α=[1, 50.26], γ=0.25         | 0.3919  | 0.3498   |
| 1:50      | CBFocal    | β=0.99999, γ=0.5             | 0.3735  | 0.3109   |
| 1:8       | BareCE     | w_pos=1                      | 0.4332  | 0.4452   |
| 1:8       | WeightedCE | w_pos=1.0                    | 0.4332  | 0.4452   |
| 1:8       | Focal      | α=[1, 1], γ=1.0              | 0.4386  | 0.4310   |
| 1:8       | CBFocal    | β=0.9, γ=1.0                 | 0.4386  | 0.4310   |

**Best Lab 2 @ 1:50 (val-selected):** Focal α=[1, 50.26] γ=0.25 → test MCC **0.3498**.
**Best Lab 2 @ 1:8 (val-selected):** Focal α=[1,1] γ=1 (≡ CBFocal β=0.9 γ=1) → test MCC **0.4310**.

## Comparison vs Lab 1

### At 1:50

|                     | model                                      | MCC    |
|---------------------|--------------------------------------------|--------|
| Lab 2 best          | Focal α=[1, 50.26] γ=0.25 (val-selected)   | 0.3498 |
| Lab 1 best MLP      | SMOTE_mlp                                  | 0.3447 |
| Lab 1 best overall  | SMOTE_xgb                                  | 0.3600 |

Lab 2's val-selected loss-reweighted MLP narrowly beats Lab 1's MLP+resampling
(+0.0051) but loses to Lab 1's best overall (XGBoost+SMOTE) by 0.0102 MCC.

### At 1:8

|                     | model                                              | MCC    |
|---------------------|----------------------------------------------------|--------|
| Lab 2 best          | Focal α=[1,1] γ=1 (≡ CBFocal β=0.9 γ=1)            | 0.4310 |
| Lab 1 best MLP      | bare_mlp                                           | 0.4322 |
| Lab 1 best overall  | bare_mlp                                           | 0.4322 |

At 1:8, Lab 2's val-selected MLP+loss loses to Lab 1's bare MLP by 0.0012 —
within descriptive-comparison noise. (Lab 2's BareCE row at 1:8 reaches test
MCC=0.4452, which would beat Lab 1's bare MLP by +0.0130; the gap is plausibly
small DataLoader-pipeline differences.)

## Launch

The notebook reads `../lab1-imbalance/embeddings/*.npz` — Lab 1's cache must
exist. If missing, run Lab 1's §5 first or copy the four `.npz` files from
someone else's run (they're not in git — gitignored).

### Local install (Linux + GPU, macOS, Windows)

See top-level `README.md` § *Run anywhere – A/B*. The §1.2 device-check cell
auto-detects `cuda → mps → cpu`.

Lab 2's per-sweep wallclock is fast (~1–2 min on Tesla T4, ~30–60 min on CPU)
because training is just an MLP head on cached embeddings — no heavy DINOv2
forward pass, unlike Lab 1.

**One caveat from this lab's development sessions:** `uv sync --no-sources`
rewrites `uv.lock` to a different torch version (it bypasses the `cu124` index
pin and resolves a newer torch from PyPI, which then pulls in CUDA-13-generation
deps). If you need a Mac env, `--no-sources` is the available workaround, but
**do NOT commit the resulting lock change** — it would replace the canonical
cu124 pin used by the VM. The repo's intended workflow is "heavy compute on
the VM, CSV inspection / doc edits on the laptop".

### Remote (GCP VM, recommended for any heavy compute)

See top-level `README.md` § *Run anywhere – C*. Same SSH-tunnel + Jupyter MCP
pattern as Lab 1.

## Reproducibility

Single seed `42`. `train_mlp` reseeds internally on each call, so the notebook
IS reproducible across fresh kernels. NOT bit-reproducible across in-session
re-runs of the same sweep cell (same reason as Lab 1 — DataLoader generator
state). Canonical numbers come from a top-to-bottom run on a freshly restarted
kernel.

## Notes / known fallbacks

- **No `losses.py` module.** Hand-rolled losses live inline in §4 of the
  notebook — the centerpiece reviewers should read directly.
- **Embeddings reused from Lab 1.** Notebook reads
  `../lab1-imbalance/embeddings/*.npz` with no symlink. If Lab 1 is regenerated
  with a different `TARGET_RATIO` or `SEED`, delete those `.npz` files before
  re-running Lab 2 (Lab 1's cache invalidation rule applies transitively).
- **Focal Loss at gamma=5 may not converge in 30 epochs.** Rows where
  `epoch_stopped == 30` are flagged with yellow background in §8.1 and
  excluded from family-winner selection unless every row in the family hit
  the cap. The cap is inherited from Lab 1 for comparability — do not raise.
  In this run: 2 rows at 1:8 hit the cap (`WeightedCE w_pos=100.0` and
  `Focal α=[1,1] γ=3.0`).
- **NaN-pr_auc guard.** `evaluate_on_test` writes `pr_auc=NaN` when
  `probs[:, 1]` contains non-finite values. Discovered during the expanded
  sweep: Focal α=[1,1] γ=0.25 at 1:50 produces NaN logits (the model fails to
  escape the all-negative trap and softmax saturates). MCC and per-class
  metrics use argmax and are unaffected; only the threshold-independent PR-AUC
  column is invalidated for those rows.
- **Pandas StringDtype gotcha.** The §8.6 / §11 helper detects Lab 1's
  experiment-name column with `pd.api.types.is_string_dtype(col)` rather than
  `col.dtype == object`. Recent pandas infers `StringDtype` for string columns
  by default, which fails the `==object` check. If you regenerate Lab 1's CSV
  with an older pandas, both forms will work; the new form is forward-compatible.
