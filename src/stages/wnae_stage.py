import os
import json
import yaml
import torch
import h5py
import shutil
import argparse
import numpy as np
from pathlib import Path
from torch.utils import data
from importlib import import_module
from example.trainer import TrainerWassersteinNormalizedAutoEncoder
from example.pdf_generation import create_report
from example.architectures import Encoder, Decoder
from wnae._logger import log


# ------------------------
# Helper functions
# ------------------------

def load_config(config_path, overrides=None):
    """Load YAML config and apply CLI overrides."""
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    if overrides:
        for key, value in overrides.items():
            # Handle nested override like "train.batch_size=512"
            keys = key.split(".")
            sub = config
            for k in keys[:-1]:
                sub = sub.setdefault(k, {})
            sub[keys[-1]] = value
    return config


def prepare_dataloaders(data_config, device):
    """Load and prepare datasets."""
    f = h5py.File(data_config["filepath"], "r")

    n_train_sample = data_config.get("n_train_sample", 100000)
    n_test_sample = data_config.get("n_test_sample", 20000)
    standardize = data_config.get("standardize", False)
    min_max = data_config.get("min_max", False)

    # Load datasets
    x_train = f["data"]["Background_data"]["Train"]["DATA"][:n_train_sample]
    x_test = f["data"]["Background_data"]["Test"]["DATA"][:n_test_sample]
    x_sig = f["data"]["Signal_data"]["GluGluHToBB_M-125"]["DATA"][:n_test_sample]

    def to_tensor(x):
        return torch.tensor(x.reshape(x.shape[0], -1), dtype=torch.float32, device=device)

    x_train, x_test, x_sig = map(to_tensor, (x_train, x_test, x_sig))

    if standardize:
        mean = x_train.mean(dim=0)
        std = x_train.std(dim=0)
        std[std == 0] = 1.0  # prevent division by zero

        x_train = (x_train - mean) / std
        x_test = (x_test - mean) / std
        x_sig = (x_sig - mean) / std
    
    if min_max:
        data_min = torch.min(x_train, dim=0).values
        data_max = torch.max(x_train, dim=0).values

        x_train = (x_train - data_min) / (data_max - data_min + 1e-8)
        x_test = (x_test - data_min) / (data_max - data_min + 1e-8)
        x_sig = (x_sig - data_min) / (data_max - data_min + 1e-8)

    batch_size = data_config.get("batch_size", 256)

    train_loader = data.DataLoader(data.TensorDataset(x_train), batch_size=batch_size)
    val_loader = data.DataLoader(data.TensorDataset(x_test), batch_size=batch_size)
    val_loader_no_batch = data.DataLoader(data.TensorDataset(x_test), batch_size=len(x_test))
    sig_loader = data.DataLoader(data.TensorDataset(x_sig), batch_size=batch_size)

    class MyLoader:
        def __init__(self, train_loader, val_loader, val_loader_no_batch, ood_loader):
            self.training_loader = train_loader
            self.validation_loader = val_loader
            self.validation_loader_no_batch = val_loader_no_batch
            self.ood_loader = ood_loader

    return MyLoader(train_loader, val_loader, val_loader_no_batch, sig_loader), x_train.shape[-1]


def save_config(output_path, config):
    Path(output_path).mkdir(parents=True, exist_ok=True)
    with open(f"{output_path}/config.json", "w") as file:
        json.dump(config, file, indent=4)


# ------------------------
# Main training function
# ------------------------

def main(args):
    config = load_config(args.config, overrides=args.override)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    data_cfg = config["data"]
    train_cfg = config["training"]
    model_cfg = config["model"]

    loaders, input_size = prepare_dataloaders(
        {**data_cfg, "batch_size": train_cfg["batch_size"]}, device
    )

    # --- FIXED: Ensure Path type & handle output path increment ---
    output_path_base = Path(data_cfg["output"])
    output_path = output_path_base
    counter = 1
    while output_path.exists():
        output_path = Path(f"{output_path_base}_{counter}")
        counter += 1

    Path(output_path).mkdir(parents=True, exist_ok=True)
    print(f"Saving outputs to {output_path}")

    save_config(output_path, config)

    # --- Model setup ---
    encoder = Encoder(
        input_size=input_size,
        intermediate_architecture=tuple(model_cfg["encoder"]["intermediate_architecture"]),
        bottleneck_size=model_cfg["encoder"]["bottleneck_size"],
        drop_out=model_cfg["encoder"].get("drop_out", None),
    )
    decoder = Decoder(
        output_size=input_size,
        intermediate_architecture=tuple(model_cfg["decoder"]["intermediate_architecture"]),
        bottleneck_size=model_cfg["decoder"]["bottleneck_size"],
        drop_out=model_cfg["decoder"].get("drop_out", None),
    )

    # --- Training ---
    trainer = TrainerWassersteinNormalizedAutoEncoder(
        config=config,
        loader=loaders,
        encoder=encoder,
        decoder=decoder,
        device=device,
        output_path=str(output_path),
        loss_function="wnae",
    )

    trainer.train()
    log.info("Saving...")
    trainer.save_train_plot()
    log.info("Done.")

    log.info('Creating PDF Report...')
    create_report(output_path, config)


# ------------------------
# CLI Entry Point
# ------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Wasserstein Normalized AutoEncoder")

    parser.add_argument(
        "--config", type=str, default="config/config.yaml",
        help="Path to YAML config file"
    )
    parser.add_argument(
        "--override", nargs="*", default=[],
        help='Override config, e.g. train.batch_size=1024 data.filepath="data/mydata.h5"'
    )

    args = parser.parse_args()

    # Parse CLI overrides into dict
    override_dict = {}
    for item in args.override:
        if "=" not in item:
            continue
        key, value = item.split("=", 1)
        try:
            value = json.loads(value)
        except json.JSONDecodeError:
            pass
        override_dict[key] = value
    args.override = override_dict

    main(args)
