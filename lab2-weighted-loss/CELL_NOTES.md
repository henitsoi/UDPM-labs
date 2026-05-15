# Lab 2 cell-by-cell notes

Cell indices are 0–27 (notebook position), not the `[N]:` execution count
displayed by JupyterLab.

## §1 Setup (cells 0–2)

**Cell 0 — markdown title.** Header + one-paragraph framing that names the
three hand-rolled losses (WeightedCE, Focal, ClassBalancedFocal), the two
imbalance regimes (1:50 artificial, ~1:8 natural), and confirms we reuse
Lab 1's DINOv2 embeddings rather than re-extracting.

**Cell 1 — imports + global seed = 42.** The seed is set on `np.random`,
`torch.manual_seed`, and `torch.cuda.manual_seed_all`. Python stdlib `random`
is never used by this lab so it's not seeded. `train_mlp` reseeds internally
on each call, so the global seed here is mostly for the EDA cells (§3) and
any one-off sanity checks. Note: `import os` lives in this cell even though
it's only used in §2 — keeping every import at the top is the style choice.

**Cell 2 — device check.** On the GCP VM this prints `Device: cuda`,
`GPU: Tesla T4`, `CUDA: 12.4`. On a laptop without CUDA it falls back to MPS
or CPU; both work but training will be ~10× slower on CPU.

## §2 Data loading (cells 3–4)

**Cell 3 — `load_split` reads NPZ files from `../lab1-imbalance/embeddings/`.**
The path is relative; no symlink, no copy. If Lab 1's `.npz` cache is missing,
`FileNotFoundError` is raised with a hint to run Lab 1 §5 first.

A CWD safeguard at the top of the cell auto-chdirs into `lab2-weighted-loss/`
if the kernel was launched from repo root (e.g. `uv run jupyter lab` without
a path argument). The check is:

```python
if Path('lab1-imbalance/embeddings').exists() and not Path('../lab1-imbalance/embeddings').exists():
    os.chdir('lab2-weighted-loss')
```

i.e. "Lab 1 is reachable from CWD but not via `..`" → we must be at repo root,
so chdir into the lab dir. The fallback is a no-op on the VM (kernel CWD is
already `lab2-weighted-loss/`) and only fires for repo-root launches. The
final `assert EMBED_DIR.exists()` line catches the case where neither location
resolves (e.g. kernel started in `~` with nothing in scope).

**Cell 4 — sanity counts.** This is the canonical source of class counts used
by ClassBalancedFocalLoss in §6 and §7. For our run:
- train_50: n_neg=6232, n_pos=124 (ratio 50.3)
- train_natural: n_neg=6232, n_pos=770 (ratio 8.1)
- valid: n_neg=1363, n_pos=156 (ratio 8.7)
- test: n_neg=1307, n_pos=187 (ratio 7.0)

These numbers are reproduced verbatim in the README's "Class counts" line and
again in §9's conclusions. If you regenerate Lab 1 with a different
`TARGET_RATIO`, the train_50 counts will change and every CB-Focal row in
`results.csv` becomes stale — delete `embeddings/*.npz` first.

## §3 EDA (cells 5–6)

**Cell 5 — class-distribution bars side-by-side for both train splits.** The
only EDA plot that's actually different from Lab 1; the rest of the
embedding-space EDA is unchanged because the embeddings are byte-identical.
The y-axis is linear (the 124 positives at 1:50 are dwarfed by 6232 negatives),
but every bar has a numeric label above it via `ax.text` so the minority count
is still readable at a glance.

**Cell 6 — markdown pointer to Lab 1 §6** for the embedding-space EDA
(correlation summary, PCA, UMAP, t-SNE). Not repeated to save kernel time and
to avoid drift between two notebooks computing the same statistics.

## §4 Hand-rolled losses (cells 7–13)

§4 spans 7 cells because §4.5 is split into three sanity-check cells (one per
loss). Layout:
- cell 7: §4.1 math markdown
- cell 8: §4.2 `WeightedCrossEntropy` class
- cell 9: §4.5(a) WCE sanity check
- cell 10: §4.3 `FocalLoss` class
- cell 11: §4.5(b) Focal sanity check
- cell 12: §4.4 `ClassBalancedFocalLoss` class
- cell 13: §4.5(c) CB-Focal sanity check

This is the centerpiece of the lab. We wrote the three losses from scratch
rather than calling `nn.CrossEntropyLoss(weight=...)`.

**Reduction conventions to remember:**
- **WeightedCE** divides by sum-of-weights-over-batch — matches PyTorch's
  `nn.CrossEntropyLoss(weight=w, reduction='mean')` exactly. The sanity check
  in cell 9 verifies this at `w=[1,1]` and `w=[1,50]` with `atol=1e-6`.
- **FocalLoss** divides by batch size N — the focusing term `(1-p)^gamma` is
  non-linear in alpha, so weight-normalized averaging is not meaningful.
- **CBFocal** inherits FocalLoss's reduction. The novelty is the constructor,
  which computes effective-number weights from `class_counts` per
  Cui et al. (CVPR 2019): `w_c = (1 - β) / (1 - β^n_c)`, then normalises so
  weights sum to `n_classes`.

**Numerical stability:** all three losses use `F.log_softmax(logits).exp()`
for `p_true`, never `softmax(logits).log()`. The `(1-p)^gamma` focusing term
in Focal/CBFocal underflows to 0 at gamma=5 when computed naively from
post-softmax probabilities; routing through `log_softmax` keeps the dynamic
range stable.

**Alpha-parameter semantics:** Lin et al. (2017) define alpha as a SCALAR in
[0,1] paired with (1-alpha) for the other class. We use the per-class VECTOR
form, which is more general and more PyTorch-natural. To map: our
`alpha=[1, 50.26]` corresponds to Lin's `alpha ≈ 0.98`. The factor 50.26 is
`n_neg / n_pos = 6232 / 124` — the inverse-frequency weight at 1:50.

**Sanity-check cells (9, 11, 13)** are self-contained: each re-seeds `_logits`
and `_y` via `torch.Generator().manual_seed(0)` so the cells can be re-run in
any order without state leakage. Each one prints `OK: <claim>` lines that
spell out the invariant being verified (e.g. "WCE@[1,1] == nn.CE
within atol=1e-6").

## §5 Training helper (cells 14–16)

**Cell 14 — `MLP` class + `train_mlp` helper**, with a leading comment
confirming the hyperparameters are verified identical to Lab 1's `train_mlp`:
lr=1e-3, wd=1e-4, max_epochs=30, patience=5, batch=256, hidden=256, dropout=0.3.

The only differences from Lab 1 are structural, not behavioral:
- Lab 1 kept tensors on CPU and sliced batches per-iteration; Lab 2 moves
  them to device once and uses a seeded `DataLoader` generator
  (`torch.Generator().manual_seed(SEED)`) so shuffle order is reproducible.
- Lab 1 hard-coded `nn.CrossEntropyLoss()`; Lab 2 accepts `loss_fn` as a
  parameter (the entire point of Lab 2).
- Lab 2 returns a richer dict (`epoch_stopped`, `val_balacc_at_stop`,
  `val_mcc_at_stop`, `history`) needed for §6/§7 sweep reporting.

**Cell 15 — `evaluate_on_test`.** Returns a dict matching the CSV schema.
Computes discrete-prediction metrics (accuracy, per-class precision/recall/F1,
confusion matrix, MCC, balanced accuracy) AND continuous-score PR-AUC
(needs probabilities, which we get from `F.softmax`).

**NaN-pr_auc guard.** `evaluate_on_test` writes `pr_auc=NaN` when `probs[:, 1]`
contains non-finite values. This was discovered during the expanded sweep —
Focal α=[1,1] γ=0.25 at 1:50 produces NaN logits (the model fails to escape
the all-negative trap and softmax saturates). MCC and per-class metrics use
argmax and are unaffected by the NaN; only the threshold-independent PR-AUC
column is invalidated for those rows.

**Cell 16 — `run_sweep`.** Iterates a grid of `{'loss_family', 'hyperparams',
'loss_fn'}` entries, calls `train_mlp` + `evaluate_on_test` per row, prints
progress, returns a DataFrame matching the CSV schema. The `loss_fn` is a
zero-arg callable that constructs a *fresh* loss instance per run — important
because some losses (CBFocal) cache device-moved tensors and would carry state
across runs if reused.

## §6 Train at 1:50 (cells 17–18)

**Cell 17 — grid construction.** Collapses spec's §6.1–§6.4 into a single
cell; sub-sections are comment-delimited. 51 rows total: 1 BareCE +
12 WeightedCE + 18 Focal + 20 CB-Focal. The `assert len(grid_50) == 51` guard
catches grid-mismatch (off-by-one in a nested loop is the typical failure
mode).

WeightedCE w_pos values: `{1, 2, 5, 10, 15, 20, 25, 35, 50, 75, 100, balanced}`
(12 rows; "balanced" maps to `n_neg/n_pos`, which at 1:50 is ≈50.26 — close to
but not exactly the explicit `w_pos=50` row). Focal sweeps `(alpha ∈ {[1,1],
[1, n_neg/(2·n_pos)], [1, n_neg/n_pos]}, gamma ∈ {0.25, 0.5, 1, 2, 3, 5})`
(18 rows; the middle alpha gives half-balanced weighting). CB-Focal sweeps
`(beta ∈ {0.9, 0.99, 0.999, 0.9999, 0.99999}, gamma ∈ {0.5, 1, 2, 5})` (20
rows). At low β (β=0.9 with these class counts) the effective-number weights
are nearly uniform, so `CBFocal(β=0.9, γ=g)` produces numerically the same
loss as `Focal(α=[1,1], γ=g)` — that's why those pairs appear as duplicates
in the leaderboard.

**Cell 18 — execute sweep, write `results.csv`.** 51 sequential training runs
(~6–10 min on Tesla T4 — most rows early-stop quickly because bare CE / low
weighting collapses at epoch 1 to "predict negative always" and the patience-5
val-MCC plateau triggers immediately).

## §7 Train at 1:8 (cells 19–20)

Same shape as §6 but training on `y_train_natural` with `W_BALANCED_8 ≈ 8.09`.

**Cell 19 — grid construction (51 rows).** The grid is *structurally
identical* to §6's; only the underlying training labels differ. This is on
purpose — comparing the same sweep across two imbalance levels is the whole
"1:50 vs 1:8 inversion" question.

**Cell 20 — execute sweep, write `results_natural.csv`.** Two rows in this
sweep (`WeightedCE w_pos=100.0` and `Focal α=[1,1] γ=3.0`) hit the 30-epoch
cap — flagged yellow in §8.1 and excluded from family-winner selection.

## §8 Unified visualizations (cells 21–26)

**Cell 21 — combined leaderboard styled DataFrame.** 102 rows (51+51), sorted
by test MCC, RdYlGn background gradient on `mcc` and `balanced_acc`. Rows where
`epoch_stopped == 30` get yellow background — undertrained, excluded from
family-winner selection in cell 22 unless every row in their family hit cap.

**Cell 22 — `select_family_winners` applies the spec §8.2 rule:** prefer
`epoch_stopped < 30`; among those, sort by `val_mcc_at_stop` descending and
take the top. Returns 8 winner rows (4 families × 2 imbalances). Using
val-MCC instead of test-MCC avoids test-set selection bias — a tenet borrowed
from Lab 1.

**Cell 23 — grouped bar chart** of per-family winner test MCC at both
imbalances. The 1:50 vs 1:8 contrast is the takeaway: at 1:50, Focal (with
balanced alpha) narrowly tops WeightedCE on val-selection; at 1:8, BareCE and
WeightedCE w_pos=1 are tied for best.

**Cell 24 — 8-panel confusion-matrix grid** (2 imbalances × 4 families).
Visualises that BareCE @ 1:50 collapses entirely (all-negative prediction)
while every reweighted loss recovers some positives.

**Cell 25 — CB-Focal two-panel β×γ heatmap, one per imbalance.** Spec §8.4.
After grid expansion this is a 5×4 (β × γ) heatmap per panel. Reveals that
high-β with low-γ (β=0.99999, γ=0.5) is the val-selected 1:50 winner while
β=0.9 + γ=1 is the val-selected 1:8 winner — the effective-number reweighting
tracks the actual class ratio.

**Cell 26 — apples-to-apples Lab 1 panel at BOTH imbalances.** Loads
`../lab1-imbalance/results.csv` AND `../lab1-imbalance/results_natural.csv`,
detects the experiment-name column with `pd.api.types.is_string_dtype(col)`
(handles both legacy `object` and pandas 2.x `StringDtype`). Renders a 2-panel
bar chart (1:50 on the left, 1:8 on the right) showing three bars per panel:
Lab 1 best MLP / Lab 2 best (val-selected) / Lab 1 best overall.

## §9 Conclusions (cell 27)

Markdown answering all 6 spec questions with real numbers. Top-level takeaways:

- **Focal-with-balanced-alpha wins at 1:50** by val-MCC selection (Focal α=[1,
  50.26] γ=0.25, test MCC=0.3498), narrowly edging WeightedCE w_pos=50.0
  (0.3480). The simple inverse-frequency direction is what wins across both
  families; the choice between Focal and WeightedCE is within descriptive
  noise. Note: by raw test MCC, `WeightedCE w_pos=10` scores higher at 0.3837,
  but it's not the val-selected winner — val-MCC overshoots the actual test
  optimum on this small validation set (N_pos=156). Using val-MCC for
  selection is the methodologically correct choice; the val→test divergence
  is documented as an honest caveat in §9 Q2.
- **The 1:50 → 1:8 inversion reproduces.** At natural prevalence, BareCE
  reaches MCC=0.4452 — top of the val-selected leaderboard. Loss reweighting
  is unnecessary overhead at moderate imbalance for val-selected operating
  points. (Caveat: raw-test-MCC top at 1:8 IS a CB-Focal row — β=0.99 γ=0.5
  → 0.4595 — so "weighted losses *can* help at 1:8" is true; val-MCC
  selection just doesn't recover that operating point.)
- **Lab 2 narrowly beats Lab 1's best MLP at 1:50** (+0.005 MCC) but **loses
  to Lab 1's best overall** (SMOTE_xgb at 0.3600) by 0.010. At 1:8 the
  val-selected Lab-2 MLP ties Lab 1's bare MLP within noise (−0.001).
  Conclusion: at 1:50 imbalance, classifier choice (tree ensemble vs MLP)
  matters more than the loss-vs-resample distinction; at 1:8, neither matters
  much because both reduce to plain MLP on natural-distribution embeddings.
