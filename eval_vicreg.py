import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), "src"))

import argparse
import numpy as np
import torch
import yaml
import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

import model


def load_config(config_path):
    with open(config_path) as f:
        return yaml.safe_load(f)


def load_data(filepath, device, n_train=10000, n_val=5000, n_sig=5000):
    with h5py.File(filepath, "r") as f:
        x_train = f["data"]["Background_data"]["Train"]["DATA"][:n_train]
        x_val   = f["data"]["Background_data"]["Test"]["DATA"][:n_val]
        x_sig   = f["data"]["Signal_data"]["GluGluHToBB_M-125"]["DATA"][:n_sig]

    to_tensor = lambda x: torch.tensor(
        x.reshape(x.shape[0], -1), dtype=torch.float32, device=device
    )
    return to_tensor(x_train), to_tensor(x_val), to_tensor(x_sig)


def build_model(config, device):
    enc_cfg = config["model"]["encoder"]
    encoder = model.Encoder(
        input_size=57,
        intermediate_architecture=enc_cfg["intermediate_architecture"],
        bottleneck_size=enc_cfg["bottleneck_size"],
        drop_out=enc_cfg["drop_out"],
    )
    vic_cfg = config["model"]["vicreg"]
    vicreg = model.VICReg(
        encoder=encoder,
        projection_dim=vic_cfg["projection_dim"],
        projection_layers=vic_cfg["projection_layers"],
        sim_coeff=vic_cfg["sim_coeff"],
        std_coeff=vic_cfg["std_coeff"],
        cov_coeff=vic_cfg["cov_coeff"],
    ).to(device)
    return vicreg


@torch.no_grad()
def extract_embeddings(vicreg, x, batch_size=4096):
    vicreg.eval()
    parts = []
    for i in range(0, len(x), batch_size):
        parts.append(vicreg.encoder(x[i : i + batch_size]).cpu().numpy())
    return np.concatenate(parts, axis=0)


def plot_2d(ax, coords, labels, colors, title, legend=True):
    for label, color, mask in zip(
        ["Train BG", "Val BG", "Signal"],
        ["steelblue", "darkorange", "crimson"],
        labels,
    ):
        if mask is not None:
            ax.scatter(
                coords[mask, 0],
                coords[mask, 1],
                s=4,
                alpha=0.35,
                color=color,
                label=label,
                rasterized=True,
            )
    ax.set_title(title, fontsize=11)
    ax.set_xlabel("Component 1")
    ax.set_ylabel("Component 2")
    if legend:
        ax.legend(markerscale=3, fontsize=8)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config",      default="config/vicreg_config.yaml")
    parser.add_argument("--checkpoint",  default=None, help="Path to .pt checkpoint file")
    parser.add_argument("--random-init", action="store_true", help="Skip checkpoint, use random weights")
    parser.add_argument("--data",        default=None,  help="Override data filepath from config")
    parser.add_argument("--n-train",     type=int, default=10000)
    parser.add_argument("--n-val",       type=int, default=5000)
    parser.add_argument("--n-sig",       type=int, default=5000)
    parser.add_argument("--tsne-perp",   type=float, default=30.0)
    parser.add_argument("--outdir",      default="eval_vicreg_plots")
    args = parser.parse_args()

    if not args.random_init and args.checkpoint is None:
        raise ValueError("Provide --checkpoint or --random-init")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    config = load_config(args.config)
    data_path = args.data or config["data"]["filepath"]

    vicreg = build_model(config, device)

    if args.random_init:
        epoch, loss = "random", float("nan")
        print("Using randomly initialized model (no checkpoint)")
    else:
        print(f"Loading checkpoint: {args.checkpoint}")
        ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
        epoch = ckpt.get("epoch", "?")
        loss  = ckpt.get("loss",  float("nan"))
        print(f"  Epoch {epoch}  |  loss {loss:.4f}")
        vicreg.load_state_dict(ckpt["model_state_dict"])

    print(f"Loading data from: {data_path}")
    x_train, x_val, x_sig = load_data(
        data_path, device, args.n_train, args.n_val, args.n_sig
    )

    print("Extracting encoder embeddings …")
    emb_train = extract_embeddings(vicreg, x_train)
    emb_val   = extract_embeddings(vicreg, x_val)
    emb_sig   = extract_embeddings(vicreg, x_sig)

    # ── PCA ──────────────────────────────────────────────────────────────────
    print("Running PCA …")
    pca = PCA(n_components=2)
    all_emb = np.concatenate([emb_train, emb_val, emb_sig], axis=0)
    pca.fit(all_emb)

    pc_train = pca.transform(emb_train)
    pc_val   = pca.transform(emb_val)
    pc_sig   = pca.transform(emb_sig)

    var = pca.explained_variance_ratio_
    print(f"  PCA variance explained: PC1={var[0]:.2%}  PC2={var[1]:.2%}")

    # ── t-SNE ────────────────────────────────────────────────────────────────
    print(f"Running t-SNE (perplexity={args.tsne_perp}) …")
    # Subsample for speed; keep proportions
    n_sub = min(args.n_train, 5000)
    idx_tr = np.random.choice(len(emb_train), n_sub, replace=False)
    idx_va = np.random.choice(len(emb_val),   min(args.n_val, 2500), replace=False)
    idx_si = np.random.choice(len(emb_sig),   min(args.n_sig, 2500), replace=False)

    sub_train = emb_train[idx_tr]
    sub_val   = emb_val[idx_va]
    sub_sig   = emb_sig[idx_si]

    n0, n1, n2 = len(sub_train), len(sub_val), len(sub_sig)
    tsne_all = TSNE(
        n_components=2,
        perplexity=args.tsne_perp,
        learning_rate="auto",
        init="pca",
        random_state=42,
        n_iter=1000,
    ).fit_transform(np.concatenate([sub_train, sub_val, sub_sig], axis=0))

    ts_train = tsne_all[:n0]
    ts_val   = tsne_all[n0 : n0 + n1]
    ts_sig   = tsne_all[n0 + n1 :]

    # ── Plotting ──────────────────────────────────────────────────────────────
    os.makedirs(args.outdir, exist_ok=True)

    # --- Figure 1: Training embeddings (PCA + t-SNE side by side) ---
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle(f"Training Set Embeddings  (epoch {epoch})", fontsize=13)

    ax = axes[0]
    ax.scatter(pc_train[:, 0], pc_train[:, 1], s=4, alpha=0.3, color="steelblue",
               label="Train BG", rasterized=True)
    ax.scatter(pc_sig[:, 0], pc_sig[:, 1], s=4, alpha=0.3, color="crimson",
               label="Signal", rasterized=True)
    ax.set_title(f"PCA  (var: {var[0]:.1%} + {var[1]:.1%})", fontsize=11)
    ax.set_xlabel("PC 1"); ax.set_ylabel("PC 2")
    ax.legend(markerscale=3, fontsize=8)

    ax = axes[1]
    ax.scatter(ts_train[:, 0], ts_train[:, 1], s=4, alpha=0.3, color="steelblue",
               label="Train BG", rasterized=True)
    ax.scatter(ts_sig[:, 0], ts_sig[:, 1], s=4, alpha=0.3, color="crimson",
               label="Signal", rasterized=True)
    ax.set_title(f"t-SNE (perp={args.tsne_perp})", fontsize=11)
    ax.set_xlabel("t-SNE 1"); ax.set_ylabel("t-SNE 2")
    ax.legend(markerscale=3, fontsize=8)

    plt.tight_layout()
    out_train = os.path.join(args.outdir, f"train_embeddings_epoch{epoch}.png")
    plt.savefig(out_train, dpi=150)
    plt.close(fig)
    print(f"Saved: {out_train}")

    # --- Figure 2: Validation embeddings (PCA + t-SNE side by side) ---
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle(f"Validation Set Embeddings  (epoch {epoch})", fontsize=13)

    ax = axes[0]
    ax.scatter(pc_val[:, 0], pc_val[:, 1], s=4, alpha=0.3, color="darkorange",
               label="Val BG", rasterized=True)
    ax.scatter(pc_sig[:, 0], pc_sig[:, 1], s=4, alpha=0.3, color="crimson",
               label="Signal", rasterized=True)
    ax.set_title(f"PCA  (var: {var[0]:.1%} + {var[1]:.1%})", fontsize=11)
    ax.set_xlabel("PC 1"); ax.set_ylabel("PC 2")
    ax.legend(markerscale=3, fontsize=8)

    ax = axes[1]
    ax.scatter(ts_val[:, 0], ts_val[:, 1], s=4, alpha=0.3, color="darkorange",
               label="Val BG", rasterized=True)
    ax.scatter(ts_sig[:, 0], ts_sig[:, 1], s=4, alpha=0.3, color="crimson",
               label="Signal", rasterized=True)
    ax.set_title(f"t-SNE (perp={args.tsne_perp})", fontsize=11)
    ax.set_xlabel("t-SNE 1"); ax.set_ylabel("t-SNE 2")
    ax.legend(markerscale=3, fontsize=8)

    plt.tight_layout()
    out_val = os.path.join(args.outdir, f"val_embeddings_epoch{epoch}.png")
    plt.savefig(out_val, dpi=150)
    plt.close(fig)
    print(f"Saved: {out_val}")

    # --- Figure 3: Combined 2x2 overview ---
    fig, axes = plt.subplots(2, 2, figsize=(14, 11))
    fig.suptitle(f"VICReg Encoder Embeddings  (epoch {epoch}, loss {loss:.4f})", fontsize=13)

    # Row 0: PCA
    ax = axes[0, 0]
    ax.scatter(pc_train[:, 0], pc_train[:, 1], s=4, alpha=0.3, color="steelblue",
               label="Train BG", rasterized=True)
    ax.scatter(pc_sig[:, 0], pc_sig[:, 1], s=4, alpha=0.3, color="crimson",
               label="Signal", rasterized=True)
    ax.set_title(f"PCA — Train  (var: {var[0]:.1%} + {var[1]:.1%})")
    ax.set_xlabel("PC 1"); ax.set_ylabel("PC 2"); ax.legend(markerscale=3, fontsize=8)

    ax = axes[0, 1]
    ax.scatter(pc_val[:, 0], pc_val[:, 1], s=4, alpha=0.3, color="darkorange",
               label="Val BG", rasterized=True)
    ax.scatter(pc_sig[:, 0], pc_sig[:, 1], s=4, alpha=0.3, color="crimson",
               label="Signal", rasterized=True)
    ax.set_title(f"PCA — Val  (var: {var[0]:.1%} + {var[1]:.1%})")
    ax.set_xlabel("PC 1"); ax.set_ylabel("PC 2"); ax.legend(markerscale=3, fontsize=8)

    # Row 1: t-SNE
    ax = axes[1, 0]
    ax.scatter(ts_train[:, 0], ts_train[:, 1], s=4, alpha=0.3, color="steelblue",
               label="Train BG", rasterized=True)
    ax.scatter(ts_sig[:, 0], ts_sig[:, 1], s=4, alpha=0.3, color="crimson",
               label="Signal", rasterized=True)
    ax.set_title(f"t-SNE — Train  (perp={args.tsne_perp})")
    ax.set_xlabel("t-SNE 1"); ax.set_ylabel("t-SNE 2"); ax.legend(markerscale=3, fontsize=8)

    ax = axes[1, 1]
    ax.scatter(ts_val[:, 0], ts_val[:, 1], s=4, alpha=0.3, color="darkorange",
               label="Val BG", rasterized=True)
    ax.scatter(ts_sig[:, 0], ts_sig[:, 1], s=4, alpha=0.3, color="crimson",
               label="Signal", rasterized=True)
    ax.set_title(f"t-SNE — Val  (perp={args.tsne_perp})")
    ax.set_xlabel("t-SNE 1"); ax.set_ylabel("t-SNE 2"); ax.legend(markerscale=3, fontsize=8)

    plt.tight_layout()
    out_combined = os.path.join(args.outdir, f"combined_epoch{epoch}.png")
    plt.savefig(out_combined, dpi=150)
    plt.close(fig)
    print(f"Saved: {out_combined}")

    # --- Figure 4: Per-feature embedding distributions ---
    n_features = emb_train.shape[1]
    ncols = 4
    nrows = (n_features + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 3.5, nrows * 2.8))
    axes = np.array(axes).reshape(-1)
    fig.suptitle(f"Encoder Output Feature Distributions  (epoch {epoch})", fontsize=13)

    for i in range(n_features):
        ax = axes[i]
        lo = min(emb_train[:, i].min(), emb_val[:, i].min(), emb_sig[:, i].min())
        hi = max(emb_train[:, i].max(), emb_val[:, i].max(), emb_sig[:, i].max())
        bins = np.linspace(lo, hi, 50)
        ax.hist(emb_train[:, i], bins=bins, histtype="step", color="steelblue",
                density=True, label="Train BG", linewidth=1.2)
        ax.hist(emb_val[:, i],   bins=bins, histtype="step", color="darkorange",
                density=True, label="Val BG",   linewidth=1.2)
        ax.hist(emb_sig[:, i],   bins=bins, histtype="step", color="crimson",
                density=True, label="Signal",   linewidth=1.2)
        ax.set_title(f"Feature {i}", fontsize=9)
        ax.tick_params(labelsize=7)
        if i == 0:
            ax.legend(fontsize=7)

    for j in range(n_features, len(axes)):
        axes[j].set_visible(False)

    plt.tight_layout()
    out_features = os.path.join(args.outdir, f"feature_distributions_epoch{epoch}.png")
    plt.savefig(out_features, dpi=150)
    plt.close(fig)
    print(f"Saved: {out_features}")
    print("Done.")


if __name__ == "__main__":
    main()
