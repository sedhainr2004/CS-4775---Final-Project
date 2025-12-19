import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt

from pipeline_utils import resolve_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot training loss curves.")
    parser.add_argument("--metrics_path", required=True, help="Path to metrics.json")
    parser.add_argument("--output_path", required=True, help="Output image path (png)")
    parser.add_argument("--title", default="Training Loss")
    return parser.parse_args()


def main():
    args = parse_args()
    metrics_path = resolve_path(args.metrics_path)
    with metrics_path.open("r") as handle:
        payload = json.load(handle)

    history = payload.get("history", [])
    epochs = [entry["epoch"] for entry in history]
    total_loss = [entry["loss"] for entry in history]
    recon_loss = [entry.get("recon_loss") for entry in history]
    cluster_loss = [entry.get("cluster_loss") for entry in history]

    has_recon = all(val is not None for val in recon_loss)
    has_cluster = all(val is not None for val in cluster_loss)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Left plot: Original scale with dual y-axis for cluster loss
    ax1 = axes[0]
    ax1.plot(epochs, total_loss, label="total", color="blue", linestyle="--", alpha=0.7)
    if has_recon:
        ax1.plot(epochs, recon_loss, label="recon", color="orange")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss (total/recon)", color="orange")
    ax1.tick_params(axis="y", labelcolor="orange")
    ax1.set_title(f"{args.title} (Dual Axis)")

    if has_cluster:
        ax1_twin = ax1.twinx()
        ax1_twin.plot(epochs, cluster_loss, label="cluster", color="green")
        ax1_twin.set_ylabel("Cluster Loss (KL-div)", color="green")
        ax1_twin.tick_params(axis="y", labelcolor="green")

    # Combine legends from both axes
    lines1, labels1 = ax1.get_legend_handles_labels()
    if has_cluster:
        lines2, labels2 = ax1_twin.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper right")
    else:
        ax1.legend(loc="upper right")

    # Right plot: Log scale for better visualization of all losses
    ax2 = axes[1]
    ax2.semilogy(epochs, total_loss, label="total", color="blue", linestyle="--", alpha=0.7)
    if has_recon:
        ax2.semilogy(epochs, recon_loss, label="recon", color="orange")
    if has_cluster:
        # Avoid log(0) by using small epsilon
        cluster_safe = [max(c, 1e-10) for c in cluster_loss]
        ax2.semilogy(epochs, cluster_safe, label="cluster", color="green")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Loss (log scale)")
    ax2.set_title(f"{args.title} (Log Scale)")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    fig.tight_layout()

    output_path = resolve_path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200)
    print(f"Saved loss plot to {output_path}")


if __name__ == "__main__":
    main()