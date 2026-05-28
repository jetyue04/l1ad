import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), "src"))

import argparse
import copy
import yaml
import torch
import h5py
import numpy as np
from pathlib import Path
from torch.utils import data
from tqdm import tqdm
import pandas as pd
from sklearn.metrics import roc_auc_score

import model as M


def load_config(path):
    with open(path) as f:
        return yaml.safe_load(f)


def load_data(filepath, device, n_train, n_test):
    with h5py.File(filepath, "r") as f:
        x_train = f["data"]["Background_data"]["Train"]["DATA"][:n_train]
        x_test  = f["data"]["Background_data"]["Test"]["DATA"][:n_test]
        x_sig   = f["data"]["Signal_data"]["GluGluHToBB_M-125"]["DATA"][:n_test]

    to_tensor = lambda x: torch.tensor(x.reshape(x.shape[0], -1), dtype=torch.float32, device=device)
    return to_tensor(x_train), to_tensor(x_test), to_tensor(x_sig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--vicreg-checkpoint", required=True)
    parser.add_argument("--wnae-config",       default="config/vicreg_wnae_config.yaml")
    parser.add_argument("--data",              default=None)
    parser.add_argument("--outdir",            default=None,
                        help="Output directory. Defaults to output_sweep/step<N>_size<S>")
    parser.add_argument("--freeze-encoder",    action="store_true")
    # MCMC overrides
    parser.add_argument("--x-step",      type=int,   default=None, help="Override wnae.x_step")
    parser.add_argument("--x-step-size", type=float, default=None, help="Override wnae.x_step_size")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    cfg = load_config(args.wnae_config)

    # Apply MCMC overrides
    if args.x_step is not None:
        cfg["wnae"]["x_step"] = args.x_step
    if args.x_step_size is not None:
        cfg["wnae"]["x_step_size"] = args.x_step_size
    if args.data:
        cfg["data"]["filepath"] = args.data

    x_step      = cfg["wnae"]["x_step"]
    x_step_size = cfg["wnae"]["x_step_size"]

    outdir = Path(args.outdir) if args.outdir else Path(f"output_sweep/step{x_step}_size{x_step_size}")
    outdir.mkdir(parents=True, exist_ok=True)

    # Save the effective config so this run is reproducible
    with open(outdir / "config.yaml", "w") as f:
        yaml.dump(cfg, f)

    print(f"x_step={x_step}  x_step_size={x_step_size}  outdir={outdir}")

    # ── Load VICReg encoder ───────────────────────────────────────────────────
    print(f"Loading VICReg checkpoint: {args.vicreg_checkpoint}")
    ckpt = torch.load(args.vicreg_checkpoint, map_location=device, weights_only=False)
    print(f"  VICReg epoch: {ckpt.get('epoch', '?')}")

    vic_cfg = load_config("config/vicreg_config.yaml")
    enc_cfg = vic_cfg["model"]["encoder"]
    encoder = M.Encoder(
        input_size=57,
        intermediate_architecture=enc_cfg["intermediate_architecture"],
        bottleneck_size=enc_cfg["bottleneck_size"],
        drop_out=enc_cfg["drop_out"],
    )
    vicreg_state = ckpt["model_state_dict"]
    encoder_state = {
        k[len("encoder."):]: v
        for k, v in vicreg_state.items()
        if k.startswith("encoder.")
    }
    encoder.load_state_dict(encoder_state)
    encoder = encoder.to(device)

    if args.freeze_encoder:
        for p in encoder.parameters():
            p.requires_grad_(False)
        print("  Encoder frozen.")

    bottleneck = enc_cfg["bottleneck_size"]
    decoder = M.Decoder(
        output_size=57,
        intermediate_architecture=list(enc_cfg["intermediate_architecture"]),
        bottleneck_size=bottleneck,
        drop_out=None,
    ).to(device)

    # ── Data ──────────────────────────────────────────────────────────────────
    data_path  = cfg["data"]["filepath"]
    n_train    = cfg["data"].get("n_train_sample", 50000)
    n_test     = cfg["data"].get("n_test_sample",  10000)
    batch_size = cfg["training"]["batch_size"]

    print(f"Loading data from: {data_path}")
    x_train, x_test, x_sig = load_data(data_path, device, n_train, n_test)
    train_loader = data.DataLoader(data.TensorDataset(x_train), batch_size=batch_size, shuffle=True)
    val_loader   = data.DataLoader(data.TensorDataset(x_test),  batch_size=batch_size)

    # ── Model + optimizer ─────────────────────────────────────────────────────
    wnae = M.WNAE(encoder=encoder, decoder=decoder, **cfg["wnae"]).to(device)

    trainable = filter(lambda p: p.requires_grad, wnae.parameters())
    optimizer = getattr(torch.optim, cfg["training"]["optimizer"])(
        trainable, lr=float(cfg["training"]["learning_rate"]),
    )

    if cfg["training"]["lr_scheduler"]:
        lr_scheduler = getattr(torch.optim.lr_scheduler, cfg["training"]["lr_scheduler"])(
            optimizer, **cfg["training"]["lr_scheduler_args"]
        )
    else:
        lr_scheduler = None

    # ── Training loop ─────────────────────────────────────────────────────────
    # Column names match generate_report.py expectations
    metrics = {"epoch": [], "training_loss": [], "validation_loss": [], "auc": []}
    best_val_loss = float("inf")
    best_epoch    = 0
    es_counter    = 0
    es_patience   = cfg["training"]["es_patience"]
    n_epochs      = cfg["training"]["n_epochs"]

    for epoch in range(n_epochs):
        wnae.train()
        train_loss, n_batches = 0., 0
        for (batch,) in tqdm(train_loader, desc=f"Epoch {epoch+1}/{n_epochs} [train]", leave=False):
            loss, _ = wnae.train_step(batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            n_batches  += 1
        train_loss /= n_batches

        if lr_scheduler is not None:
            if isinstance(lr_scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                lr_scheduler.step(train_loss)
            else:
                lr_scheduler.step()

        wnae.eval()
        val_loss, n_batches = 0., 0
        val_reco_errors = []
        for (batch,) in val_loader:
            d = wnae.validation_step(batch)
            val_loss += d["loss"]
            val_reco_errors.append(d["reco_errors"])
            n_batches += 1
        val_loss /= n_batches
        val_reco_errors = torch.cat(val_reco_errors).numpy()

        sig_reco_errors = wnae.evaluate(x_sig)["reco_errors"].numpy()
        y_true = np.concatenate([np.zeros(len(val_reco_errors)), np.ones(len(sig_reco_errors))])
        y_pred = np.concatenate([val_reco_errors, sig_reco_errors])
        auc = roc_auc_score(y_true, y_pred)

        metrics["epoch"].append(epoch + 1)
        metrics["training_loss"].append(train_loss)
        metrics["validation_loss"].append(val_loss)
        metrics["auc"].append(auc)

        print(f"Epoch {epoch+1}/{n_epochs} | train {train_loss:.4f} | val {val_loss:.4f} | AUC {auc:.4f}")
        pd.DataFrame(metrics).to_csv(outdir / "training.csv", index=False)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch    = epoch + 1
            torch.save({"epoch": epoch+1, "model_state_dict": wnae.state_dict()}, outdir / "best.pt")
            es_counter = 0
        else:
            es_counter += 1

        if es_counter > es_patience:
            print(f"Early stopping at epoch {epoch+1}")
            break

    torch.save({"epoch": epoch+1, "model_state_dict": wnae.state_dict()}, outdir / "last.pt")
    (outdir / "info.txt").write_text(f"Best epoch: {best_epoch}\n")
    print(f"Done. Outputs in {outdir}/")


if __name__ == "__main__":
    main()
