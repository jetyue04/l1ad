import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), "src"))

import argparse
import csv
import numpy as np
import torch
import torch.nn.functional as F
import yaml
import h5py
from torch.utils import data

import model
import model.augmentations as aug


def load_config(config_path, overrides=None):
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    if overrides:
        for key, value in overrides.items():
            keys = key.split(".")
            sub = config
            for k in keys[:-1]:
                sub = sub.setdefault(k, {})
            sub[keys[-1]] = value
    return config


def load_data(filepath, device):
    f = h5py.File(filepath, "r")
    x_train = f["data"]["Background_data"]["Train"]["DATA"][:100000]
    x_test  = f["data"]["Background_data"]["Test"]["DATA"][:50000]
    x_sig   = f["data"]["Signal_data"]["GluGluHToBB_M-125"]["DATA"][:50000]
    scale   = f["data"]["Normalisation"]["norm_scale"][:]
    bias    = f["data"]["Normalisation"]["norm_bias"][:]

    to_tensor = lambda x: torch.tensor(x.reshape(x.shape[0], -1), dtype=torch.float32, device=device)
    return to_tensor(x_train), to_tensor(x_test), to_tensor(x_sig), scale, bias


def make_loaders(x_train, x_test, x_sig, batch_size):
    train_loader = data.DataLoader(data.TensorDataset(x_train), batch_size=batch_size, shuffle=True)
    val_loader   = data.DataLoader(data.TensorDataset(x_test),  batch_size=batch_size)
    val_loader_no_batch = data.DataLoader(data.TensorDataset(x_test), batch_size=len(x_test))
    sig_loader   = data.DataLoader(data.TensorDataset(x_sig),   batch_size=batch_size)
    return train_loader, val_loader, val_loader_no_batch, sig_loader


def vicreg_loss(z1, z2, sim_coeff=50, std_coeff=50, cov_coeff=1):
    repr_loss = F.mse_loss(z1, z2)

    z1 = z1 - z1.mean(dim=0, keepdim=True)
    z2 = z2 - z2.mean(dim=0, keepdim=True)

    std_z1 = torch.sqrt(z1.var(dim=0, unbiased=False) + 1e-4)
    std_z2 = torch.sqrt(z2.var(dim=0, unbiased=False) + 1e-4)
    std_loss = (torch.mean(torch.relu(1 - std_z1)) + torch.mean(torch.relu(1 - std_z2))) / 2

    def cov_loss(z):
        cov = (z.T @ z) / (z.shape[0] - 1)
        cov.fill_diagonal_(0)
        return (cov ** 2).sum() / z.shape[1]

    cov = cov_loss(z1) + cov_loss(z2)
    loss = sim_coeff * repr_loss + std_coeff * std_loss + cov_coeff * cov
    return repr_loss, std_loss, cov, loss


def save_checkpoint(vicreg, optimizer, epoch, loss, checkpoint_dir):
    os.makedirs(checkpoint_dir, exist_ok=True)
    path = os.path.join(checkpoint_dir, f"checkpoint_epoch{epoch}.pt")
    torch.save({
        "epoch": epoch,
        "model_state_dict": vicreg.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "loss": loss,
    }, path)
    print(f"Checkpoint saved to {path}")


def train(config, device):
    x_train, x_test, x_sig, scale, bias = load_data(config["data"]["filepath"], device)

    batch_size          = config["training"]["batch_size"]
    n_epochs            = config["training"]["n_epochs"]
    checkpoint_dir      = config["training"]["checkpoint_dir"]
    checkpoint_interval = config["training"]["checkpoint_interval"]

    train_loader, val_loader, val_loader_no_batch, sig_loader = make_loaders(
        x_train, x_test, x_sig, batch_size
    )

    enc_cfg = config["model"]["encoder"]
    encoder = model.Encoder(
        input_size=57,
        intermediate_architecture=enc_cfg["intermediate_architecture"],
        bottleneck_size=enc_cfg["bottleneck_size"],
        drop_out=enc_cfg["drop_out"],
    ).to(device)

    vic_cfg = config["model"]["vicreg"]
    vicreg = model.VICReg(
        encoder=encoder,
        projection_dim=vic_cfg["projection_dim"],
        projection_layers=vic_cfg["projection_layers"],
        sim_coeff=vic_cfg["sim_coeff"],
        std_coeff=vic_cfg["std_coeff"],
        cov_coeff=vic_cfg["cov_coeff"],
    ).to(device)

    aug_cfg = config["augmentation"]
    blur        = aug.FastFeatureBlur(prob=aug_cfg["blur_p"], magnitude=aug_cfg["blur_magnitude"], strength=aug_cfg["blur_strength"]).to(device)
    blur_prime  = aug.FastFeatureBlur(prob=aug_cfg["blur_p"], magnitude=aug_cfg["blur_magnitude"], strength=aug_cfg["blur_strength"]).to(device)
    mask        = aug.FastObjectMask(prob=aug_cfg["mask_p"]).to(device)
    mask_prime  = aug.FastObjectMask(prob=aug_cfg["mask_p"]).to(device)
    rotation        = aug.FastLorentzRotation(prob=aug_cfg["rotation_p"], norm_scale=scale, norm_bias=bias).to(device)
    rotation_prime  = aug.FastLorentzRotation(prob=aug_cfg["rotation_p"], norm_scale=scale, norm_bias=bias).to(device)

    def augment(x):
        x1 = rotation(mask(blur(x.clone())))
        x2 = rotation_prime(mask_prime(blur_prime(x.clone())))
        return x1, x2

    optimizer = torch.optim.Adam(
        vicreg.parameters(),
        lr=config["training"]["learning_rate"],
        weight_decay=config["training"]["weight_decay"],
    )

    metrics_path = os.path.join(checkpoint_dir, "metrics.csv")
    os.makedirs(checkpoint_dir, exist_ok=True)
    with open(metrics_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "total_loss", "repr_loss", "std_loss", "cov_loss", "embed_std"])

    for epoch in range(n_epochs):
        vicreg.train()

        repr_total = std_total = cov_total = loss_total = 0.0
        batch_count = 0
        embedding_stds = []

        for (batch,) in train_loader:
            batch = batch.to(device)
            x1, x2 = augment(batch)

            z1 = vicreg(x1)
            z2 = vicreg(x2)

            repr_loss, std_loss, cov_loss, total_loss = vicreg_loss(z1, z2)

            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

            repr_total += repr_loss.item()
            std_total  += std_loss.item()
            cov_total  += cov_loss.item()
            loss_total += total_loss.item()
            batch_count += 1

            with torch.no_grad():
                embedding_stds.append(vicreg.encoder(x1).std(dim=0).mean().item())

        n = batch_count
        avg_loss   = loss_total / n
        avg_repr   = repr_total / n
        avg_std    = std_total  / n
        avg_cov    = cov_total  / n
        avg_embstd = float(np.mean(embedding_stds))

        print(
            f"Epoch [{epoch+1}/{n_epochs}] | "
            f"Total: {avg_loss:.4f} | "
            f"Repr: {avg_repr:.4f} | "
            f"Std: {avg_std:.4f} | "
            f"Cov: {avg_cov:.4f} | "
            f"EmbedStd: {avg_embstd:.4f}"
        )

        with open(metrics_path, "a", newline="") as f:
            csv.writer(f).writerow([epoch + 1, avg_loss, avg_repr, avg_std, avg_cov, avg_embstd])

        if (epoch + 1) % checkpoint_interval == 0:
            save_checkpoint(vicreg, optimizer, epoch + 1, avg_loss, checkpoint_dir)

    save_checkpoint(vicreg, optimizer, n_epochs, avg_loss, checkpoint_dir)
    return vicreg


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/vicreg_config.yaml")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--data",   default=None, help="Override data filepath")
    args = parser.parse_args()

    config = load_config(args.config)
    if args.epochs:
        config["training"]["n_epochs"] = args.epochs
    if args.data:
        config["data"]["filepath"] = args.data

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    train(config, device)
