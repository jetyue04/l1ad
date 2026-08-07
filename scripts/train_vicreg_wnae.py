import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))

import argparse
import numpy as np
import torch
import yaml
import h5py
from torch.utils import data

import model
import trainer as trainer_module


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


def load_data(config, device):
    f = h5py.File(config["data"]["filepath"], "r")
    n_train = config["data"]["n_train_sample"]
    n_test  = config["data"]["n_test_sample"]
    signal  = config["data"]["signal"]

    x_train = f["data"]["Background_data"]["Train"]["DATA"][:n_train]
    x_test  = f["data"]["Background_data"]["Test"]["DATA"][:n_test]
    x_sig   = f["data"]["Signal_data"][signal]["DATA"][:n_test]

    to_tensor = lambda x: torch.tensor(x.reshape(x.shape[0], -1), dtype=torch.float32, device=device)
    return to_tensor(x_train), to_tensor(x_test), to_tensor(x_sig)


def load_vicreg_encoder(config, device):
    enc_cfg = config["vicreg"]["encoder"]
    enc = model.Encoder(
        input_size=enc_cfg["input_size"],
        intermediate_architecture=enc_cfg["intermediate_architecture"],
        bottleneck_size=enc_cfg["bottleneck_size"],
        drop_out=enc_cfg["drop_out"],
    )
    ckpt = torch.load(config["vicreg"]["checkpoint"], map_location=device, weights_only=False)
    enc_state = {
        k[len("encoder."):]: v
        for k, v in ckpt["model_state_dict"].items()
        if k.startswith("encoder.")
    }
    enc.load_state_dict(enc_state)
    enc = enc.to(device).eval()
    print(f"Loaded VICReg encoder (epoch {ckpt.get('epoch', '?')}): "
          f"{enc_cfg['input_size']} → {enc_cfg['intermediate_architecture']} → {enc_cfg['bottleneck_size']}")
    return enc


class DataLoaders:
    def __init__(self, training_loader, validation_loader, validation_loader_no_batch, ood_loader):
        self.training_loader = training_loader
        self.validation_loader = validation_loader
        self.validation_loader_no_batch = validation_loader_no_batch
        self.ood_loader = ood_loader


def embed_and_standardize(vicreg_enc, x_train, x_test, x_sig, batch_size):
    with torch.no_grad():
        e_train = vicreg_enc(x_train)
        e_test  = vicreg_enc(x_test)
        e_sig   = vicreg_enc(x_sig)

    emb_mean = e_train.mean(dim=0)
    emb_std  = e_train.std(dim=0).clamp(min=1e-6)
    e_train_s = (e_train - emb_mean) / emb_std
    e_test_s  = (e_test  - emb_mean) / emb_std
    e_sig_s   = (e_sig   - emb_mean) / emb_std
    print(f"Embeddings: train={tuple(e_train.shape)} — "
          f"standardized mean: {e_train_s.mean():.4f}, std: {e_train_s.std():.4f}")

    loaders = DataLoaders(
        training_loader=data.DataLoader(data.TensorDataset(e_train_s), batch_size=batch_size, shuffle=True),
        validation_loader=data.DataLoader(data.TensorDataset(e_test_s), batch_size=batch_size),
        validation_loader_no_batch=data.DataLoader(data.TensorDataset(e_test_s), batch_size=len(e_test_s)),
        ood_loader=data.DataLoader(data.TensorDataset(e_sig_s), batch_size=batch_size),
    )
    return loaders, emb_mean, emb_std


def train(config, device):
    x_train, x_test, x_sig = load_data(config, device)

    vicreg_enc = load_vicreg_encoder(config, device)
    embed_dim  = config["vicreg"]["encoder"]["bottleneck_size"]

    loaders, emb_mean, emb_std = embed_and_standardize(
        vicreg_enc, x_train, x_test, x_sig, config["training"]["batch_size"]
    )

    enc_cfg = config["model"]["encoder"]
    wnae_encoder = model.Encoder(
        input_size=embed_dim,
        intermediate_architecture=enc_cfg["intermediate_architecture"],
        bottleneck_size=enc_cfg["bottleneck_size"],
        drop_out=enc_cfg["drop_out"],
    ).to(device)

    dec_cfg = config["model"]["decoder"]
    wnae_decoder = model.Decoder(
        output_size=embed_dim,
        intermediate_architecture=dec_cfg["intermediate_architecture"],
        bottleneck_size=dec_cfg["bottleneck_size"],
        drop_out=dec_cfg["drop_out"],
    ).to(device)

    print(f"WNAE: {embed_dim} → {enc_cfg['intermediate_architecture']} → {enc_cfg['bottleneck_size']} "
          f"→ {dec_cfg['intermediate_architecture'][::-1]} → {embed_dim}")

    output_path = config["data"]["output"]
    wnae_trainer = trainer_module.TrainerWassersteinNormalizedAutoEncoder(
        config=config,
        loader=loaders,
        encoder=wnae_encoder,
        decoder=wnae_decoder,
        device=device,
        output_path=output_path,
        loss_function="wnae",
    )
    wnae_trainer.train()
    wnae_trainer.save_train_plot()

    torch.save({
        "model_state_dict": wnae_trainer.model.state_dict(),
        "emb_mean": emb_mean,
        "emb_std":  emb_std,
        "config":   config,
    }, os.path.join(output_path, "vicreg_wnae_final.pt"))
    print(f"Saved final checkpoint to {output_path}/vicreg_wnae_final.pt")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config",     default="config/vicreg_wnae_config.yaml")
    parser.add_argument("--epochs",     type=int,   default=None)
    parser.add_argument("--data",       default=None, help="Override data filepath")
    parser.add_argument("--output",     default=None, help="Override output directory")
    parser.add_argument("--checkpoint", default=None, help="Override VICReg checkpoint path")
    args = parser.parse_args()

    config = load_config(args.config)
    if args.epochs:
        config["training"]["n_epochs"] = args.epochs
    if args.data:
        config["data"]["filepath"] = args.data
    if args.output:
        config["data"]["output"] = args.output
    if args.checkpoint:
        config["vicreg"]["checkpoint"] = args.checkpoint

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    train(config, device)
