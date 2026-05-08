# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

L1AD (Level 1 Anomaly Detection) is a machine learning framework for anomaly detection in particle physics detector data. It trains autoencoders on background (normal) events; anomalous signal events produce higher reconstruction errors, enabling AUC-ROC-based discrimination.

## Common Commands

**Quick training (standalone script):**
```bash
python train_axo.py
```

**Config-driven training:**
```bash
python src/models/wnae/wasserstein_normalized_autoencoder.py \
    --config config/config.yaml \
    --override training.batch_size=512 data.filepath="path/to/data.h5"
```

**Multi-stage pipeline:**
```bash
python src/run_pipeline.py  # uses configs/pipeline.yaml
```

**Kubernetes GPU job (NRP-Nautilus cluster):**
```bash
kubectl apply -f nrp/wnae_job.yaml
```

## Architecture

### Two Model Approaches

**WNAE (Wasserstein Normalized AutoEncoder)** — primary model:
- Uses dual Langevin MCMC sampling: one chain in data space (x), one in latent space (z)
- Persistent Contrastive Divergence (PCD) with a replay buffer (`_sample_buffer.py`) for stable negative examples
- Loss combines reconstruction + Wasserstein distance between real and sampled distributions
- Anomaly score = reconstruction error at inference time

**VICReg (Variance-Invariance-Covariance Regularization)** — secondary/experimental:
- Self-supervised contrastive approach; uses encoder + projection head
- `src/stages/vicreg_stage.py` is currently a stub

### Key Files

| File | Role |
|------|------|
| `src/models/wnae/wasserstein_normalized_autoencoder.py` | Core WNAE model + CLI entry point |
| `src/models/wnae/_mcmc_utils.py` | Langevin dynamics sampling for x and z |
| `src/models/wnae/_sample_buffer.py` | Replay buffer for PCD training |
| `src/trainer/wnae_trainer.py` | Training loop, validation, AUC evaluation |
| `src/stages/wnae_stage.py` | Pipeline stage wrapper for WNAE |
| `src/run_pipeline.py` | Orchestrates multi-stage pipeline |
| `config/config.yaml` | All hyperparameters (data paths, MCMC, model arch) |

### Data Format

HDF5 input files with structure:
- `data/Background_data/Train/DATA` — training (normal) events
- `data/Background_data/Test/DATA` — test background
- `data/Signal_data/GluGluHToBB_M-125/DATA` — signal/anomaly events
- `data/Normalisation/norm_scale`, `norm_bias` — normalization parameters

### Training Loop

1. Train on background data with WNAE loss (reconstruction + MCMC-based Wasserstein term)
2. Validate on held-out background; checkpoint best model by validation loss
3. Evaluate AUC-ROC by comparing background vs. signal reconstruction errors
4. Output: timestamped directory with checkpoints, CSV metrics, and 1D feature histograms

## Environment

Conda environment `wnae_env`:
- Python 3.10, PyTorch 2.5.1, CUDA 12.1
- Key packages: `pot` (Python Optimal Transport), `numpy`, `pandas`, `scikit-learn`, `rich`, `typer`

## Notes

- There are two model directories: `src/models/` (current) and `src/model/` (older, being phased out). Prefer `src/models/`.
- Both `train_axo.py` and `train_axo.ipynb` exist; the `.py` script is the production entry point, the notebook has more inline debugging.
- GPU jobs on NRP-Nautilus mount data at `/axovol`; local runs expect HDF5 path from config.
