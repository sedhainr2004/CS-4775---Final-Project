import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt

from pipeline_utils import resolve_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot clustering metrics over epochs.")
    parser.add_argument("--metrics_path", required=True, help="Path to metrics.json")
    parser.add_argument("--output_path", required=True, help="Output image path (png)")
    parser.add_argument("--title", default="Clustering Metrics")
    return parser.parse_args()


def extract_series(history, key, split):
    epochs = []
    values = []
    for entry in history:
        if split in entry and key in entry[split]:
            epochs.append(entry["epoch"])
            values.append(entry[split][key])
    return epochs, values


def main():
    args = parse_args()
    metrics_path = resolve_path(args.metrics_path)
    with metrics_path.open("r") as handle:
        payload = json.load(handle)

    history = payload.get("history", [])

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for ax, metric in zip(axes, ["ari", "nmi", "silhouette"]):
        for split, color in [("train", "blue"), ("val", "orange"), ("test", "green")]:
            epochs, values = extract_series(history, metric, split)
            if epochs:
                ax.plot(epochs, values, label=split, color=color)
        ax.set_title(metric.upper())
        ax.set_xlabel("Epoch")
        ax.set_ylabel(metric.upper())
        ax.grid(True, alpha=0.3)
        ax.legend()

    fig.suptitle(args.title)
    fig.tight_layout(rect=[0, 0, 1, 0.95])

    output_path = resolve_path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200)
    print(f"Saved clustering plot to {output_path}")


if __name__ == "__main__":
    main()
