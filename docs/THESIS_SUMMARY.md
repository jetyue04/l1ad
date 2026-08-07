# Thesis Project Summary — L1AD (Level 1 Anomaly Detection)

> This file summarises the full scope of the project for use in writing an undergraduate thesis.
> It covers motivation, data, models, training pipelines, infrastructure, and engineering decisions.

---

## 1. Project Motivation

The project targets **model-agnostic anomaly detection at the LHC Level-1 trigger** (L1T). The L1T must make accept/reject decisions in ~4 µs; most physics-motivated triggers look for specific pre-defined signatures and therefore miss Beyond-Standard-Model (BSM) signals that do not match known patterns.

The goal is to train an **unsupervised autoencoder** exclusively on ordinary QCD background events. At inference, BSM ("signal") events produce higher reconstruction errors than background events, enabling discrimination without any prior assumption about the signal topology. Performance is measured by **AUC-ROC**: the area under the ROC curve when background (label 0) and signal (label 1) reconstruction errors are used as scores.

---

## 2. Data

**Source file:** HDF5 (v5 conditions, `conditionsupdate_apr25.h5`)

**Structure inside the file:**
| Key | Content |
|-----|---------|
| `data/Background_data/Train/DATA` | Training QCD background events |
| `data/Background_data/Test/DATA` | Held-out background events |
| `data/Signal_data/GluGluHToBB_M-125/DATA` | Signal: Higgs → bb̄ (m = 125 GeV) |
| `data/Normalisation/norm_scale` | Per-feature scale factors |
| `data/Normalisation/norm_bias` | Per-feature bias values |

**Event representation:** Each event is a flat vector of **57 features** (19 objects × 3 features each: pT, η, φ). The phi features carry L1 hardware granularity (144 or 576 bins per 2π depending on object type).

**Normalisation:** Min-max normalisation (controlled by `data.min_max` config flag); an optional z-score standardisation path also exists (`data.standardize`).

**Typical dataset sizes used:**
- Train: 50 000–100 000 background events
- Validation/test: 10 000–50 000 background events
- Signal: 10 000–50 000 signal events

---

## 3. Models

### 3.1 WNAE — Wasserstein Normalised AutoEncoder

The primary model. Combines ideas from **Normalised AutoEncoders (NAE)** and **Optimal Transport / Wasserstein distances**.

#### Architecture

| Component | Topology (default config) |
|-----------|--------------------------|
| Encoder | 57 → [28, 15] → 8 (bottleneck) |
| Decoder | 8 → [57, 128, 64, 32, 24] → 57 |

Both encoder and decoder are fully-connected MLPs with ReLU activations. Dropout is optionally applied per layer. The bottleneck is 8-dimensional by default.

#### Energy function

The model defines an **energy** as the per-sample reconstruction MSE (or MAE):

```
E(x) = (1/d) * ||x - Decoder(Encoder(x))||²
```

This energy is low for background events the model has learned to reconstruct, and high for anomalous events.

#### MCMC sampling — Langevin dynamics

To obtain "negative samples" (samples from the model distribution), the WNAE runs **Langevin Monte Carlo** in data space:

```
x_{t+1} = x_t  -  η · ∇_x E(x_t) / T  +  σ · ε,    ε ~ N(0, I)
```

where η is the step size, T is temperature, and σ = √(2η) at equilibrium. Optional Metropolis-Hastings (MH) correction can be applied for detailed balance, though it was not used in the main runs.

**Three sampling modes:**
- `cd` — Contrastive Divergence: initialise chain from data points, run a few steps
- `pcd` — **Persistent CD (used by default)**: maintain a replay buffer; re-initialise a fraction (1 − replay_ratio) from noise, keep the rest from the previous chain
- `omi` — On-Manifold Initialisation: first run a chain in latent space Z, decode, then refine in data space X

#### Persistent Contrastive Divergence and the Replay Buffer

The `SampleBuffer` stores up to `buffer_size` (default 10 000) MCMC samples from previous iterations. At the start of each sampling call, `replay_ratio` (default 0.95) of the batch is seeded from the buffer (warm-start), and the remaining 5% from the prior (Gaussian or Uniform). After sampling, new samples are pushed into the buffer (FIFO). This prevents the chain from collapsing and keeps negative samples near the model's current energy surface.

#### Loss function

The full WNAE loss is the **Earth Mover's Distance (EMD / Wasserstein-1)** between the data distribution and the MCMC-sampled model distribution, computed via the Python Optimal Transport library (`pot`):

```
L_WNAE = EMD(x_data, x_MCMC)
```

The transport plan is solved with `ot.emd2`. The loss matrix (pairwise squared Euclidean distances) and weights are kept on the same device to avoid CUDA/CPU mismatches. Gradients flow through `ot.emd2` via POT's autograd backend.

The training loop also supports two ablation modes:
- `ae` — pure reconstruction loss (MSE), no MCMC
- `nae` — NAE loss: E(x_data) − E(x_MCMC), no EMD

#### Anomaly score at inference

```
score(x) = E(x) = (1/d) * ||x - Decoder(Encoder(x))||²
```

Higher score → more anomalous.

#### Key hyperparameters (from `config/config.yaml` and `config/vicreg_wnae_config.yaml`)

| Parameter | Default (standalone WNAE) | VICReg-WNAE pipeline |
|-----------|--------------------------|----------------------|
| Batch size | 256 | 2048 |
| Epochs | 2 (debug) / up to 100 | 100 |
| Optimizer | AdamW | AdamW |
| Learning rate | 5e-3 | 5e-3 |
| LR scheduler | ReduceLROnPlateau (factor 0.8, patience 20) | same |
| x_step | 10 | 5 |
| x_step_size | auto from noise_std | 0.05 |
| x_noise_std | 0.22 | 0.22 |
| x_temperature | 0.063 | 0.063 |
| x_bound | [−3, 3] | [−3, 3] |
| z_step | 10 | 5 |
| z_step_size | 1 | 0.05 |
| Replay buffer size | 10 000 | 10 000 |
| Replay ratio | 0.95 | 0.95 |

---

### 3.2 VICReg — Variance-Invariance-Covariance Regularisation

A self-supervised contrastive model used as a **pretrained feature extractor** for the WNAE. VICReg was originally proposed by Bardes et al. (2022).

#### Architecture

| Component | Topology |
|-----------|----------|
| Encoder | 57 → [29] → 10 (bottleneck) |
| Projector | 10 → 128 → 128 → 128 (3-layer MLP, BatchNorm + ReLU) |

The projector head is discarded after training; only the encoder weights are transferred.

#### VICReg loss

Two augmented views x₁, x₂ of each event are passed through the same encoder+projector to produce embeddings z₁, z₂ ∈ ℝ¹²⁸:

```
L = λ · L_sim + μ · L_std + ν · L_cov
```

- **L_sim** (invariance): MSE between z₁ and z₂ — enforces the representation to be stable under augmentation
- **L_std** (variance): hinge loss that keeps the standard deviation of each embedding dimension ≥ 1, preventing collapse
- **L_cov** (covariance): off-diagonal squared covariance, decorrelates embedding dimensions

Default coefficients: λ = μ = 50, ν = 1.

#### Data augmentations (physics-motivated)

Three stochastic augmentations are applied independently to produce each view:

1. **FastFeatureBlur** — randomly replaces feature values with uniform noise. Controlled by `blur_p` (probability of applying to an event), `blur_magnitude` (mixing coefficient), and `blur_strength` (per-feature probability). Simulates detector noise / missing measurements.

2. **FastObjectMask** — zeros out entire particle objects (all 3 features of a particle simultaneously) with probability `mask_p`. Simulates inefficiency where a trigger object is not reconstructed.

3. **FastLorentzRotation** — applies a random azimuthal (φ) rotation to all particles in an event. Properly maps from normalised detector-integer space → radians → adds uniform rotation → maps back. Applied with probability `rotation_p`. Exploits the physical symmetry that QCD background is uniform in φ.

Augmentation is applied in sequence: `blur → mask → rotation`. Bugs in an earlier version of augmentation (wrong phi stride indices, incorrect non-rotation branch) were fixed during development.

Augmentation hyperparameters used for training:
| Parameter | Value |
|-----------|-------|
| blur_p | 0.900 |
| blur_magnitude | 0.927 |
| blur_strength | 0.750 |
| mask_p | 0.571 |
| rotation_p | 0.500 |

---

## 4. Training Pipelines

### 4.1 Standalone WNAE (`train_axo.py`)

Trains WNAE directly on raw 57-dimensional events. Primarily used for early prototyping and debugging. Entry point also exists as `train_axo.ipynb` for interactive exploration.

### 4.2 VICReg only (`train_vicreg.py`)

Trains VICReg on 100 000 background events for up to 1000 epochs (batch size 2048, Adam lr=5e-5, weight decay=1e-6). Saves checkpoints to `/axovol/l1ad/checkpoints/vicreg/` every 50 epochs. Also logs per-epoch metrics (total, repr, std, cov losses, embedding std) to `metrics.csv`.

### 4.3 VICReg → WNAE pipeline (`train_vicreg_wnae.py`)

The main two-stage pipeline:

1. **Load** a frozen VICReg encoder checkpoint (epoch 1000 from the fixed run)
2. **Embed** all data (train / test / signal) through the frozen encoder
3. **Standardise** the embeddings: subtract training-set mean, divide by training-set std (per dimension, clipped to ≥ 1e-6)
4. **Train WNAE** on the 10-dimensional standardised embedding space using the WNAE loss

The WNAE in this pipeline has a much smaller architecture (10 → [9, 6] → 4 → [9, 6] → 10) since the encoder has already extracted structured features.

At the end of training, the final checkpoint includes the WNAE weights and the embedding mean/std needed to transform new events at inference time.

### 4.4 Sweep training (`sweep_train_wnae.py`)

Wraps the VICReg → WNAE pipeline with CLI overrides for MCMC hyperparameters (`--x-step`, `--x-step-size`). After training, automatically calls `generate_report.py` to produce a PDF summary of results. Used by the NRP sweep launcher.

---

## 5. Evaluation

After each epoch of WNAE training, the trainer:
1. Computes reconstruction errors for all validation background events
2. Computes reconstruction errors for all signal events (OOD loader)
3. Computes **AUC-ROC** using `sklearn.metrics.roc_auc_score`
4. Saves `training.csv` with columns: `epoch, training_loss, validation_loss, auc`
5. Generates 1D feature histograms comparing data vs. MCMC samples (saved under `sample_feature_1D_hist/feature_N/epoch_M.png`)

Model checkpointing: saves `best.pt` whenever validation loss improves, and `last_epoch.pt` at the end. Early stopping with configurable patience (`es_patience`).

---

## 6. Infrastructure — NRP Nautilus (Kubernetes)

Jobs run on the **National Research Platform (NRP) Nautilus** GPU cluster in the `axol1tl` namespace.

**Container image:** `gitlab-registry.nrp-nautilus.io/mquinnan/axol1tl-hub:axol1tl-container`

**Persistent storage:** PVC `axovol` mounted at `/axovol`; data and checkpoints live under `/axovol/l1ad/`.

**Job YAML files:**
| File | Purpose |
|------|---------|
| `nrp/wnae_job.yaml` | Standalone WNAE training |
| `nrp/vicreg_job.yaml` | VICReg pretraining |
| `nrp/vicreg_wnae_job.yaml` | VICReg → WNAE pipeline |

Each job runs on 1 GPU with 4 CPUs and 16 GiB RAM.

**Hyperparameter sweep launcher (`launch_sweep.py`):** Generates and optionally submits one Kubernetes Job per `(x_step, x_step_size)` combination. The sweep grid is 4 × 4 = 16 jobs:
- `x_step` ∈ {5, 10, 20, 50}
- `x_step_size` ∈ {0.01, 0.05, 0.1, 0.2}

Each job outputs to `/axovol/l1ad/checkpoints/sweep/step{N}_size{S}/` and generates a PDF report after training.

---

## 7. Repository Structure

```
l1ad/
├── config/
│   ├── config.yaml               # Standalone WNAE config
│   ├── pipeline.yaml             # Multi-stage pipeline config
│   ├── vicreg_config.yaml        # VICReg training config
│   └── vicreg_wnae_config.yaml   # VICReg → WNAE pipeline config
├── nrp/                          # Kubernetes job/pod YAMLs
├── src/
│   ├── model/
│   │   ├── encoder.py            # Shared MLP encoder
│   │   ├── decoder.py            # Shared MLP decoder
│   │   ├── vicreg.py             # VICReg model + projector
│   │   ├── augmentations.py      # Feature blur, object mask, phi rotation
│   │   └── wnae/
│   │       ├── wasserstein_normalized_autoencoder.py  # WNAE model
│   │       ├── _mcmc_utils.py    # Langevin step + chain sampling
│   │       └── _sample_buffer.py # PCD replay buffer
│   ├── trainer/
│   │   ├── wnae_trainer.py       # WNAE training loop + AUC eval
│   │   └── vicreg_trainer.py     # (stub)
│   ├── stages/                   # Pipeline stage wrappers
│   └── run_pipeline.py           # Multi-stage orchestrator
├── train_vicreg.py               # VICReg training entry point
├── train_vicreg_wnae.py          # VICReg → WNAE pipeline entry point
├── sweep_train_wnae.py           # Sweep-aware training wrapper
├── launch_sweep.py               # Kubernetes sweep job launcher
├── eval_vicreg.py                # VICReg evaluation (PCA, t-SNE, per-feature plots)
├── plot_vicreg_metrics.py        # Loss curve comparison across runs
├── generate_report.py            # PDF report generator
└── train_axo.py                  # Standalone WNAE (legacy/debug)
```

---

## 8. Key Engineering Decisions and Bug Fixes

| Issue | Resolution |
|-------|-----------|
| VICReg phi augmentation used wrong stride for phi indices (consecutive rather than every 3rd feature) | Fixed in `FastLorentzRotation`: indices set to `arange(0,19)*3+2` |
| Non-rotation branch in Lorentz rotation was overwriting normalised values with raw radians | Fixed: non-rotation branch now preserves the original normalised `x` values |
| `FastObjectMask` was masking individual features rather than entire 3-feature objects | Rewritten to reshape to `(batch, n_objects, 3)` and mask whole objects |
| WNAE validation loop crashed because `torch.no_grad()` blocked MCMC autograd (Langevin requires gradients through the energy) | Removed `torch.no_grad()` context from validation; MCMC now runs with grad enabled even during eval |
| EMD computation caused device mismatch (CUDA tensors vs CPU weights in `ot.emd2`) | Weights tensor created on the same device as inputs; POT ≥ 0.9 autograd backend used |
| VICReg LR and weight_decay were being parsed as strings from YAML | Fixed YAML types (removed quotes around numeric values) |
| Sweep script used wrong argument names when calling `launch_sweep.py` | Renamed CLI args to match `--x-step` / `--x-step-size` |

---

## 9. Software Environment

- Python 3.10
- PyTorch 2.5.1 (CUDA 12.1)
- `pot` (Python Optimal Transport) — EMD / Wasserstein-1 solver
- `scikit-learn` — AUC-ROC
- `h5py` — data loading
- `numpy`, `pandas`, `matplotlib`
- `rich`, `typer` — CLI / logging
- `reportlab` — PDF report generation
- Conda environment: `wnae_env`

---

## 10. Summary of Research Contribution

The project investigates a **two-stage approach** to L1 trigger anomaly detection:

1. **Stage 1 — VICReg pretraining:** Learn a compact, structured representation of L1 trigger events using self-supervised contrastive learning. The model sees only background events with physics-motivated augmentations (noise, object masking, φ rotation). The resulting 10-dimensional encoder embeds events in a space where background events cluster and physics symmetries are approximately encoded.

2. **Stage 2 — WNAE on embeddings:** Train a Wasserstein Normalised AutoEncoder in the low-dimensional embedding space rather than raw feature space. The WNAE uses Langevin MCMC with PCD to model the background distribution via an energy function, and uses Earth Mover's Distance as the training objective to match the model distribution to the data distribution. At inference, high reconstruction error in the embedding space flags anomalous events.

**Hypothesis:** The VICReg pretraining step provides a better-conditioned latent space for the WNAE, leading to more stable MCMC chains, faster convergence, and improved AUC-ROC on the Higgs → bb̄ signal compared to training WNAE directly on raw 57-dimensional events.

A **hyperparameter sweep** over MCMC step count and step size (16 combinations on NRP Nautilus) is used to characterise the sensitivity of anomaly detection performance to the Langevin dynamics configuration.
