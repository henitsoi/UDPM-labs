# UDPM Labs

PhD coursework — three labs exploring class imbalance on the Kaggle `kmader/skin-cancer-mnist-ham10000` dataset (raw HAM10000 with lesion-aware splits):

- **`lab1-imbalance/`** — Undersampling, oversampling, and cost-sensitive learning on DINOv2 embeddings.
- **`lab2-weighted-loss/`** — Weighted cross-entropy, Focal Loss, and Class-Balanced Focal Loss on the same embeddings.
- **`lab3-gan-augmentation/`** — Conditional DCGAN for minority-class synthesis on DermaMNIST 28×28, with best-by-DINOv2-margin checkpoint selection.

## Stack

Python 3.13 via `uv`. PyTorch + CUDA, DINOv2 (torch.hub), scikit-learn, imbalanced-learn, xgboost, umap-learn, JupyterLab.

## Run anywhere

There are three supported ways to run this repo, ordered by friction:

### A. Locally on a Linux machine with NVIDIA GPU (recommended for clones)

```bash
git clone https://github.com/henitsoi/UDPM-labs.git
cd UDPM-labs
curl -LsSf https://astral.sh/uv/install.sh | sh           # install uv if missing
uv sync                                                    # provisions Python 3.13 + CUDA wheels
uv run jupyter lab lab1-imbalance/lab1.ipynb
```

`pyproject.toml` pins `torch` to the PyTorch `cu124` index (CUDA 12.4). Verify your driver supports it (`nvidia-smi` → CUDA Version ≥ 12.4). If it doesn't, change `cu124` to `cu121` / `cu126` / `cu128` to match.

Kaggle dataset needs API credentials at `~/.kaggle/kaggle.json` or `~/.kaggle/access_token`. Get one from kaggle.com → Account → Create New API Token.

### B. Locally on macOS / Windows / Linux without GPU (CPU + slow)

The pinned CUDA torch index has no macOS or CPU-only wheels, so a fresh `uv sync` will fail on these platforms. Two options:

1. **One-shot override** — sync without the explicit CUDA source:

   ```bash
   uv sync --no-sources
   ```

   This pulls CPU torch wheels from default PyPI. DINOv2 forward pass is much slower (≈20–30 min for HAM10000's 10k images on Apple-Silicon MPS, hours on plain CPU); the rest of the notebook is fast.

2. **Permanent edit** — open `pyproject.toml` and change

   ```toml
   torch = [{ index = "pytorch-cu124" }]
   torchvision = [{ index = "pytorch-cu124" }]
   ```

   to

   ```toml
   torch = [{ index = "pytorch-cu124", marker = "sys_platform == 'linux'" }]
   torchvision = [{ index = "pytorch-cu124", marker = "sys_platform == 'linux'" }]
   ```

   Then `uv sync` resolves CUDA on Linux and CPU/MPS elsewhere.

The notebook auto-detects available hardware (CUDA → MPS → CPU) so no notebook edits are needed once `uv sync` succeeds.

### C. On a remote GPU VM (e.g. the GCP box used for this coursework)

```bash
# On the VM
curl -LsSf https://astral.sh/uv/install.sh | sh
git clone https://github.com/henitsoi/UDPM-labs.git
cd UDPM-labs
uv sync
uv run jupyter lab --no-browser --ip=127.0.0.1 --port=8888 --IdentityProvider.token=<TOKEN>

# On your laptop — tunnel and open in browser
gcloud compute ssh <vm> --zone <zone> --project <project> --tunnel-through-iap -- -N -L 8888:127.0.0.1:8888
open "http://127.0.0.1:8888/lab?token=<TOKEN>"
```

## Reproducibility

Single seed `42` (torch, numpy, sklearn). `uv.lock` pins exact dependency versions. `requirements.txt` is an `uv export` snapshot for reviewers who don't use uv.

## Notes / known fallbacks

- **Python 3.13 (not 3.14):** PyTorch's `cu124` index has no `cp314` wheels at the time of writing; revisit if PyTorch ships 3.14 support.
