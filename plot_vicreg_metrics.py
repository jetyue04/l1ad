import argparse
import csv
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def read_metrics(path):
    epochs, total, repr_l, std_l, cov_l, embed_std = [], [], [], [], [], []
    with open(path) as f:
        for row in csv.DictReader(f):
            epochs.append(int(row["epoch"]))
            total.append(float(row["total_loss"]))
            repr_l.append(float(row["repr_loss"]))
            std_l.append(float(row["std_loss"]))
            cov_l.append(float(row["cov_loss"]))
            embed_std.append(float(row["embed_std"]))
    return epochs, total, repr_l, std_l, cov_l, embed_std


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--metrics", nargs="+", required=True,
        help="One or more metrics.csv paths. Optionally label as path:label e.g. /path/metrics.csv:fixed"
    )
    parser.add_argument("--outdir", default="metric_plots")
    args = parser.parse_args()

    runs = []
    for entry in args.metrics:
        if ":" in entry:
            path, label = entry.rsplit(":", 1)
        else:
            label = os.path.basename(os.path.dirname(entry))
            path = entry
        epochs, total, repr_l, std_l, cov_l, embed_std = read_metrics(path)
        runs.append(dict(label=label, epochs=epochs, total=total,
                         repr=repr_l, std=std_l, cov=cov_l, embed_std=embed_std))
        print(f"Loaded {len(epochs)} epochs from {path} (label: {label})")

    os.makedirs(args.outdir, exist_ok=True)

    # ── Figure 1: All loss components per run, one subplot each ──────────────
    components = [
        ("total", "Total Loss"),
        ("repr",  "Repr Loss"),
        ("std",   "Std Loss"),
        ("cov",   "Cov Loss"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(13, 8))
    fig.suptitle("VICReg Loss Components", fontsize=13)
    for ax, (key, title) in zip(axes.flatten(), components):
        for run in runs:
            ax.plot(run["epochs"], run[key], label=run["label"], linewidth=1.5)
        ax.set_title(title)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.legend(fontsize=8)
    plt.tight_layout()
    out = os.path.join(args.outdir, "loss_components.png")
    plt.savefig(out, dpi=150)
    plt.close(fig)
    print(f"Saved: {out}")

    # ── Figure 2: Embedding std over training ─────────────────────────────────
    fig, ax = plt.subplots(figsize=(8, 4))
    for run in runs:
        ax.plot(run["epochs"], run["embed_std"], label=run["label"], linewidth=1.5)
    ax.set_title("Mean Encoder Embedding Std")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Embedding std")
    ax.legend(fontsize=8)
    plt.tight_layout()
    out = os.path.join(args.outdir, "embed_std.png")
    plt.savefig(out, dpi=150)
    plt.close(fig)
    print(f"Saved: {out}")

    # ── Figure 3: Total loss overlay (clean comparison) ───────────────────────
    fig, ax = plt.subplots(figsize=(8, 4))
    for run in runs:
        ax.plot(run["epochs"], run["total"], label=run["label"], linewidth=1.5)
    ax.set_title("Total Loss Comparison")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Total Loss")
    ax.legend(fontsize=8)
    plt.tight_layout()
    out = os.path.join(args.outdir, "total_loss_comparison.png")
    plt.savefig(out, dpi=150)
    plt.close(fig)
    print(f"Saved: {out}")

    print("Done.")


if __name__ == "__main__":
    main()
