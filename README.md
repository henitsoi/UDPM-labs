# UDPM Labs

PhD coursework — three labs exploring class imbalance on the Kaggle `drscarlat/melanoma` dataset:

- **`lab1-imbalance/`** — Undersampling, oversampling, and cost-sensitive learning on DINOv2 embeddings.
- **`lab2-weighted-loss/`** — Weighted cross-entropy and Focal Loss on the same embeddings (in progress).
- **`lab3-gan-augmentation/`** — Conditional GAN for minority-class synthesis (in progress).

## Setup (run on the GCP VM, not the laptop)

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
git clone git@github.com:henitsoi/UDPM-labs.git
cd UDPM-labs
uv sync
uv run jupyter lab --no-browser --ip=127.0.0.1 --port=8888 --ServerApp.token=<TOKEN>
```

Then from your laptop, SSH-tunnel `localhost:8888` to the VM and point Jupyter MCP at it.

## Stack

Python 3.14.5 via `uv`. PyTorch + CUDA, DINOv2 (torch.hub), scikit-learn, imbalanced-learn, xgboost, umap-learn.
