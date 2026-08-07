import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))

import argparse
import torch
import yaml

import model
import trainer as trainer_module

from train_vicreg_wnae import load_config, load_data, load_vicreg_encoder, embed_and_standardize
from generate_report import create_report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config",      default="config/vicreg_wnae_config.yaml")
    parser.add_argument("--epochs",      type=int,   default=None)
    parser.add_argument("--data",        default=None, help="Override data filepath")
    parser.add_argument("--output",      default=None, help="Override output directory")
    parser.add_argument("--checkpoint",  default=None, help="Override VICReg checkpoint path")
    # MCMC sweep overrides
    parser.add_argument("--x-step",      type=int,   default=None, help="Override wnae.x_step")
    parser.add_argument("--x-step-size", type=float, default=None, help="Override wnae.x_step_size")
    args = parser.parse_args()

    config = load_config(args.config)

    if args.epochs:
        config["training"]["n_epochs"] = args.epochs
    if args.data:
        config["data"]["filepath"] = args.data
    if args.checkpoint:
        config["vicreg"]["checkpoint"] = args.checkpoint
    if args.x_step is not None:
        config["wnae"]["x_step"] = args.x_step
    if args.x_step_size is not None:
        config["wnae"]["x_step_size"] = args.x_step_size

    x_step      = config["wnae"]["x_step"]
    x_step_size = config["wnae"]["x_step_size"]

    if args.output:
        config["data"]["output"] = args.output
    else:
        config["data"]["output"] = f"output_sweep/step{x_step}_size{x_step_size}"

    outdir = config["data"]["output"]
    os.makedirs(outdir, exist_ok=True)

    # Save effective config so generate_report.py and future runs can reference it
    with open(os.path.join(outdir, "config.yaml"), "w") as f:
        yaml.dump(config, f)

    print(f"x_step={x_step}  x_step_size={x_step_size}  outdir={outdir}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

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

    wnae_trainer = trainer_module.TrainerWassersteinNormalizedAutoEncoder(
        config=config,
        loader=loaders,
        encoder=wnae_encoder,
        decoder=wnae_decoder,
        device=device,
        output_path=outdir,
        loss_function="wnae",
    )
    wnae_trainer.train()
    wnae_trainer.save_train_plot()

    torch.save({
        "model_state_dict": wnae_trainer.model.state_dict(),
        "emb_mean": emb_mean,
        "emb_std":  emb_std,
        "config":   config,
    }, os.path.join(outdir, "vicreg_wnae_final.pt"))

    create_report(outdir, config=config)
    print(f"Done. Outputs in {outdir}/")


if __name__ == "__main__":
    main()
