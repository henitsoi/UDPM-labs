# Lab 1 — Undersampling and Oversampling

Demonstrates how class-imbalance methods affect a binary classifier trained on DINOv2 embeddings of the `drscarlat/melanoma` Kaggle dataset, artificially imbalanced to 1:50.

## Launch

On the GCP VM (after `uv sync`):

```bash
uv run jupyter lab --no-browser --ip=127.0.0.1 --port=8888 --ServerApp.token=<TOKEN>
```

Tunnel from laptop:

```bash
gcloud compute ssh <vm-name> -- -L 8888:localhost:8888
```

Open the notebook `lab1.ipynb` in JupyterLab or via Jupyter MCP. Run cells top-to-bottom.

## Data

Downloaded automatically by the first data-acquisition cell via `kagglehub.dataset_download("drscarlat/melanoma")`. Requires Kaggle API credentials on the VM (`~/.kaggle/kaggle.json`).

## Outputs

- `embeddings/{train,valid,test}.npz` — DINOv2 features, cached after first run (~few minutes on a single GPU).
- `results.csv` — 18-row metric table (1 dummy + 6 LogReg + 5 MLP + 6 XGBoost runs).

## Reproducibility

Single seed 42 (torch, numpy, sklearn). All non-determinism documented in the notebook's setup cell.

## Notes / fallbacks

**Python 3.14 → 3.13 fallback applied (2026-05-13).** PyTorch's `cu124` index ships only `cp39`–`cp313` wheels at the time of setup, so the project pins `requires-python = ">=3.13,<3.14"`. Revisit if PyTorch publishes `cp314` wheels.
