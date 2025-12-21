import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, silhouette_score

from scgnn_dataset import load_scgnn_data
from pipeline_utils import load_config, resolve_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate saved scGNN embeddings and cluster labels.")
    parser.add_argument("--config", required=True, help="scGNN YAML config (dataset paths).")
    parser.add_argument("--results_dir", required=True, help="Path to results directory with embeddings/labels.")
    return parser.parse_args()


def evaluate(embeddings: np.ndarray, labels: np.ndarray, true_labels: np.ndarray) -> dict:
    metrics = {}
    if len(np.unique(labels)) > 1 and len(np.unique(true_labels)) > 1:
        metrics["silhouette"] = float(silhouette_score(embeddings, labels))
    metrics["ari"] = float(adjusted_rand_score(true_labels, labels))
    metrics["nmi"] = float(normalized_mutual_info_score(true_labels, labels))
    return metrics


def main():
    args = parse_args()
    cfg = load_config(args.config)
    dataset_cfg = cfg["dataset"]
    data = load_scgnn_data(dataset_cfg)

    results_dir = resolve_path(args.results_dir)
    embeddings = np.load(results_dir / "cell_embeddings.npy")
    cluster_labels = np.load(results_dir / "cluster_labels.npy")

    metrics = {}
    for split_name, indices in data.splits.items():
        idx = indices.numpy()
        metrics[split_name] = evaluate(
            embeddings[idx],
            cluster_labels[idx],
            data.pseudo_labels[idx].numpy(),
        )

    output_path = results_dir / "eval_metrics.json"
    with open(output_path, "w") as handle:
        json.dump(metrics, handle, indent=2)
    print(f"Saved evaluation metrics to {output_path}")
    for split, vals in metrics.items():
        print(split, vals)


if __name__ == "__main__":
    main()
