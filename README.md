# L1AD — Level 1 Anomaly Detection

Anomaly detection for particle physics trigger-level data. The system trains autoencoders on background (normal) events; anomalous signal events produce higher reconstruction errors, enabling discrimination via AUC-ROC scoring.

The current pipeline is a two-stage approach:
1. **VICReg** — self-supervised pretraining of an encoder using physics-inspired augmentations
2. **WNAE** — a Wasserstein Normalized AutoEncoder trained in the VICReg embedding space

---

## Table of Contents

- [Architecture](#architecture)
- [Data Format](#data-format)
- [Environment Setup](#environment-setup)
- [Scripts](#scripts)
- [Config Files](#config-files)
- [Running the Pipeline](#running-the-pipeline)
- [NRP / Kubernetes GPU Jobs](#nrp--kubernetes-gpu-jobs)
- [Source Code Layout](#source-code-layout)
- [Branches](#branches)

---

## Architecture

### Stage 1 — VICReg Encoder

VICReg (Variance-Invariance-Covariance Regularization) is a self-supervised contrastive method. Two augmented views of each event are passed through an encoder + projection head; the loss enforces that the embeddings are similar, have non-collapsed variance, and are decorrelated.

- **Input:** 57 raw detector features per event
- **Encoder architecture:** `57 → [29] → 10`
- **Projection head:** 3-layer MLP to projection dim 128
- **Loss coefficients:** sim=50, std=50, cov=1
- **Augmentations:** feature blurring, object masking, Lorentz rotation (all differentiable, applied per-batch on GPU)

The encoder is trained on background data only. Once trained, its weights are frozen and used to embed all data (train/val/signal) into a 10-dimensional space.

### Stage 2 — WNAE (Wasserstein Normalized AutoEncoder)

WNAE is trained on the VICReg embeddings. It learns to reconstruct background events; anomalous events have higher reconstruction error at inference.

- **Input:** standardized 10-dim VICReg embeddings
- **Encoder:** `10 → [9, 6] → 4`
- **Decoder:** `4 → [6, 9] → 10`
- **Loss:** reconstruction + Wasserstein distance between real and MCMC-sampled distributions
- **MCMC:** Persistent Contrastive Divergence (PCD) with Langevin dynamics in both data space (x) and latent space (z), using a replay buffer for stable negative examples
- **Anomaly score at inference:** per-event reconstruction error

### Why this two-stage design?

Training WNAE directly on the 57 raw features is difficult because the energy-based model has to learn both the signal structure and the Wasserstein landscape. Passing through a pretrained VICReg encoder reduces dimensionality and enforces a structured, decorrelated embedding space, which makes the WNAE training more stable.

---

## Data Format

The data is stored in a single HDF5 file (currently `conditionsupdate_apr25.h5`):

```
data/
  Background_data/
    Train/DATA    — training events (background only)
    Test/DATA     — validation/test events (background)
  Signal_data/
    GluGluHToBB_M-125/DATA  — signal events for AUC evaluation
  Normalisation/
    norm_scale    — per-feature scale
    norm_bias     — per-feature bias
```

Each event is a 2D array that gets flattened to 57 features.

- **Local path:** `../training/v5/conditionsupdate_apr25.h5`
- **NRP path:** `/axovol/training/v5/conditionsupdate_apr25.h5`

---

## Environment Setup

### Conda (local or NRP interactive pod)

```bash
conda env create -f environment.yml
conda activate wnae_env
```

The environment (`wnae_env`) includes:
- Python 3.10
- PyTorch 2.5.1 with CUDA 12.1
- `pot` (Python Optimal Transport) 0.9.1
- `numpy`, `pandas`, `matplotlib`, `scikit-learn`
- `rich`, `typer`

> **Note:** On the NRP cluster the environment is pre-installed at `/axovol/conda_envs/wnae_env`. The job containers (`axol1tl-container`) also have most dependencies pre-installed; the job scripts only need to `pip install pot reportlab` on top.

### NRP container

The production container image is:
```
gitlab-registry.nrp-nautilus.io/mquinnan/axol1tl-hub:axol1tl-container
```

---

## Scripts

All root-level scripts add `src/` to the Python path automatically.

### Training

| Script | Purpose |
|--------|---------|
| `train_vicreg.py` | Train the VICReg encoder from scratch. Saves checkpoints to `checkpoint_dir` defined in the config (default `/axovol/l1ad/checkpoints/vicreg`). Logs per-epoch loss components to `metrics.csv`. |
| `train_vicreg_wnae.py` | **Main pipeline script.** Loads a pretrained VICReg checkpoint, embeds and standardizes all data, then trains WNAE in embedding space. Saves `vicreg_wnae_final.pt` containing model weights + embedding stats. |
| `sweep_train_wnae.py` | Single WNAE training run with MCMC hyperparameter overrides (`--x-step`, `--x-step-size`). Calls `train_vicreg_wnae` internally, then generates a PDF report via `generate_report.py`. Used by `launch_sweep.py`. |

### Sweeps

| Script | Purpose |
|--------|---------|
| `launch_sweep.py` | Generates and optionally submits a 4×4 grid of NRP Kubernetes jobs sweeping `x_step` ∈ {5,10,20,50} × `x_step_size` ∈ {0.01,0.05,0.1,0.2}. Each job runs `sweep_train_wnae.py` and saves output to `/axovol/l1ad/checkpoints/sweep/step{N}_size{S}/`. |

```bash
python launch_sweep.py              # preview job list
python launch_sweep.py --dry-run    # print first job YAML
python launch_sweep.py --apply      # submit all 16 jobs to NRP
```

### Evaluation

| Script | Purpose |
|--------|---------|
| `eval_vicreg.py` | Evaluate a trained VICReg encoder. Loads a checkpoint, extracts embeddings, and produces PCA and t-SNE scatter plots plus per-feature distribution histograms. Optionally runs with `--random-init` to compare against an untrained baseline. |
| `plot_vicreg_metrics.py` | Compare VICReg training loss curves across multiple runs. Takes one or more `metrics.csv` paths (optionally labelled `path:label`). |
| `generate_report.py` | Generate a formatted PDF report from a `train_vicreg_wnae.py` output directory. Reads `training.csv`, `config.yaml`, and any generated plots. Called automatically by `sweep_train_wnae.py`. |

```bash
# Evaluate a VICReg checkpoint
python eval_vicreg.py \
    --checkpoint checkpoints/vicreg_fixed/checkpoint_epoch1000.pt \
    --data ../training/v5/conditionsupdate_apr25.h5 \
    --outdir eval_plots/

# Compare two training runs
python plot_vicreg_metrics.py \
    --metrics checkpoints/vicreg_fixed/metrics.csv:fixed \
               checkpoints/vicreg_bug/metrics.csv:buggy \
    --outdir metric_comparison/
```

---

## Config Files

### `config/vicreg_config.yaml`

Used by `train_vicreg.py`. Controls VICReg training.

```yaml
training:
  batch_size: 2048
  n_epochs: 1000
  learning_rate: 0.00005
  checkpoint_dir: "/axovol/l1ad/checkpoints/vicreg"
  checkpoint_interval: 50        # save every N epochs

model:
  encoder:
    intermediate_architecture: [29]
    bottleneck_size: 10          # embedding dimension

augmentation:
  blur_p: 0.90   blur_magnitude: 0.93   blur_strength: 0.75
  mask_p: 0.57
  rotation_p: 0.5
```

### `config/vicreg_wnae_config.yaml` — **main config**

Used by `train_vicreg_wnae.py` and `sweep_train_wnae.py`. All hyperparameters for the full two-stage pipeline in one file.

```yaml
vicreg:
  checkpoint: "/axovol/l1ad/checkpoints/vicreg_fixed/checkpoint_epoch1000.pt"
  encoder:
    input_size: 57
    intermediate_architecture: [29]
    bottleneck_size: 10          # must match the trained VICReg checkpoint

model:                           # WNAE architecture (operates on embeddings)
  encoder:
    intermediate_architecture: [9, 6]
    bottleneck_size: 4
  decoder:
    intermediate_architecture: [9, 6]
    bottleneck_size: 4

training:
  batch_size: 2048
  n_epochs: 100
  es_patience: 5000              # early stopping patience (epochs)
  optimizer: "AdamW"
  learning_rate: 0.005
  lr_scheduler: "ReduceLROnPlateau"

wnae:
  sampling: "pcd"                # Persistent Contrastive Divergence
  x_step: 5                      # Langevin steps in data space
  x_step_size: 0.05              # step size (noise magnitude)
  x_temperature: 0.063
  x_bound: [-3, 3]               # clip MCMC samples to this range
  z_step: 5                      # Langevin steps in latent space
  replay: true                   # use replay buffer
  replay_ratio: 0.95             # fraction of buffer samples vs fresh
  buffer_size: 10000
```

### `config/config.yaml`

Standalone WNAE config (without VICReg). Used when training WNAE directly on raw 57-dim features. Less commonly used now that the two-stage pipeline is established.

---

## Running the Pipeline

### Step 1 — Train VICReg

```bash
python train_vicreg.py \
    --config config/vicreg_config.yaml \
    --data /path/to/data.h5
```

Checkpoints are saved every 50 epochs to the `checkpoint_dir` in the config.

### Step 2 — Train WNAE on VICReg embeddings

```bash
python train_vicreg_wnae.py \
    --config config/vicreg_wnae_config.yaml \
    --checkpoint /path/to/vicreg/checkpoint_epoch1000.pt \
    --data /path/to/data.h5 \
    --output output_vicreg_wnae/
```

Outputs saved to `--output`:
- `vicreg_wnae_final.pt` — model weights + embedding normalization stats
- `best.pt` — best checkpoint by validation loss
- `training.csv` — epoch-by-epoch loss and AUC
- `train_history.png` — loss and AUC curves
- `sample_feature_1D_hist/` — per-feature MCMC vs real distribution plots

### Step 3 — MCMC hyperparameter sweep (optional)

```bash
# Run 16 jobs on NRP covering x_step ∈ {5,10,20,50} × x_step_size ∈ {0.01,0.05,0.1,0.2}
python launch_sweep.py --apply
```

Each job writes results to `/axovol/l1ad/checkpoints/sweep/step{N}_size{S}/` and generates a PDF report.

---

## NRP / Kubernetes GPU Jobs

All job files are in `nrp/`. The cluster namespace is `axol1tl` and data is on the `axovol` PersistentVolumeClaim.

| File | Purpose |
|------|---------|
| `nrp/vicreg_job.yaml` | Train VICReg. Uses `python:3.10-slim` and installs dependencies at runtime. Runs `train_vicreg.py`. |
| `nrp/vicreg_wnae_job.yaml` | **Primary job.** Runs the full VICReg→WNAE pipeline via `train_vicreg_wnae.py`. Uses the pre-built `axol1tl-container` image. Does `git pull` to pick up latest code before running. |
| `nrp/wnae_pod.yaml` | Interactive GPU pod for development and debugging. Installs `pot`, then sleeps for 1 day. Exec into it to run commands manually. |

> `nrp/wnae_job.yaml` is an older job that runs the now-defunct `train_axo.py`. It should not be used.

### Submitting a job

```bash
# Submit
kubectl apply -f nrp/vicreg_wnae_job.yaml

# Monitor
kubectl logs -f job/vicreg-wnae-job -n axol1tl

# Delete when done
kubectl delete job vicreg-wnae-job -n axol1tl
```

### Interactive pod

```bash
kubectl apply -f nrp/wnae_pod.yaml
kubectl exec -it wnae-pod -n axol1tl -- /bin/bash
# Inside the pod:
cd /axovol/l1ad
python train_vicreg_wnae.py --config config/vicreg_wnae_config.yaml
```

---

## Source Code Layout

```
src/
├── model/
│   ├── encoder.py              Shared MLP encoder (ReLU activations, optional dropout)
│   ├── decoder.py              Shared MLP decoder
│   ├── vicreg.py               VICReg model: encoder + projection head + loss
│   ├── augmentations.py        Physics augmentations: feature blur, object mask, Lorentz rotation
│   ├── vae.py                  VAE (experimental, not part of main pipeline)
│   └── wnae/
│       ├── wasserstein_normalized_autoencoder.py   WNAE model: train_step, validation_step, evaluate
│       ├── _mcmc_utils.py      Langevin dynamics for x-space and z-space sampling
│       ├── _sample_buffer.py   Replay buffer for Persistent Contrastive Divergence
│       └── _logger.py          Logging utilities (rich-based)
├── trainer/
│   └── wnae_trainer.py         Training loop: epoch iteration, AUC evaluation, checkpointing,
│                               per-feature MCMC histogram plots, CSV metrics logging
└── stages/
    ├── wnae_stage.py           Pipeline stage wrapper for WNAE (used by run_pipeline.py)
    └── vicreg_stage.py         Stub — not yet implemented
```

### Key model methods

**`WNAE`** (`src/model/wnae/wasserstein_normalized_autoencoder.py`):
- `train_step(x)` — forward pass + MCMC negative sampling + Wasserstein loss
- `validation_step(x)` — computes reconstruction errors (requires autograd; do not wrap in `torch.no_grad`)
- `evaluate(x)` — returns per-event reconstruction errors for AUC computation

**`TrainerWassersteinNormalizedAutoEncoder`** (`src/trainer/wnae_trainer.py`):
- `train()` — full training loop with early stopping, LR scheduling, and checkpointing
- `save_train_plot()` — saves `train_history.png` with dual-axis loss+AUC plot

---

## Branches

| Branch | Status |
|--------|--------|
| `main` | Current production branch. Contains the full VICReg→WNAE pipeline. |
| `study/vicreg-training-bug` | Preserved for reference. Documents the investigation of augmentation bugs in VICReg training — compares the fixed augmentation pipeline against the original buggy version. |
