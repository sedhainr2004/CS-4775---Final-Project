import argparse
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

from data.scgnn_dataset import load_scgnn_data
from pipeline_utils import load_config, resolve_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot scGNN embeddings with labels.")
    parser.add_argument("--config", required=True, help="scGNN YAML config.")
    parser.add_argument("--results_dir", required=True, help="Directory with embeddings/labels.")
    parser.add_argument("--output_path", required=True, help="Output image path (png).")
    parser.add_argument("--method", default="umap", choices=["umap", "tsne"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--labels",
        default="predicted",
        choices=["predicted", "pseudo", "both"],
        help="Which labels to plot.",
    )
    parser.add_argument(
        "--colorbar",
        action="store_true",
        help="Use a discrete colorbar instead of legend-style coloring.",
    )
    parser.add_argument("--title", default=None, help="Override plot title.")
    return parser.parse_args()


def compute_projection(embeddings: np.ndarray, method: str, seed: int) -> np.ndarray:
    if method == "tsne":
        tsne = TSNE(n_components=2, random_state=seed, init="pca", learning_rate="auto")
        return tsne.fit_transform(embeddings)
    try:
        import umap  # type: ignore
    except ImportError as exc:
        raise ImportError("umap-learn is required for UMAP projections.") from exc
    reducer = umap.UMAP(n_components=2, random_state=seed)
    return reducer.fit_transform(embeddings)


def plot_scatter(
    ax,
    coords: np.ndarray,
    labels: np.ndarray,
    title: str,
    method: str,
    colorbar: bool = False,
):
    unique = np.unique(labels)
    if colorbar:
        cmap = plt.cm.get_cmap("tab20", len(unique))
        scatter = ax.scatter(coords[:, 0], coords[:, 1], s=10, c=labels, cmap=cmap, alpha=0.85)
        cbar = plt.colorbar(scatter, ax=ax, fraction=0.045, pad=0.04)
        cbar.set_ticks(unique)
    else:
        cmap = plt.cm.get_cmap("tab20", len(unique))
        for idx, label in enumerate(unique):
            mask = labels == label
            ax.scatter(coords[mask, 0], coords[mask, 1], s=8, color=cmap(idx), alpha=0.85)
    ax.set_title(title)
    if not colorbar:
        ax.set_xticks([])
        ax.set_yticks([])


def main():
    args = parse_args()
    cfg = load_config(args.config)
    data = load_scgnn_data(cfg["dataset"])
    dataset_name = cfg["dataset"].get("name", "dataset")

    results_dir = resolve_path(args.results_dir)
    embeddings = np.load(results_dir / "cell_embeddings.npy")
    cluster_labels = np.load(results_dir / "cluster_labels.npy")

    coords = compute_projection(embeddings, args.method, args.seed)
    pseudo_labels = data.pseudo_labels.numpy()

    title_pred = args.title or f"UMAP by scGNN ({dataset_name})"
    title_pseudo = args.title or f"Pseudo labels ({dataset_name})"

    if args.labels == "both":
        fig, axes = plt.subplots(1, 2, figsize=(10, 5))
        plot_scatter(
            axes[0],
            coords,
            cluster_labels,
            title_pred,
            args.method,
            colorbar=args.colorbar,
        )
        plot_scatter(
            axes[1],
            coords,
            pseudo_labels,
            title_pseudo,
            args.method,
            colorbar=args.colorbar,
        )
        fig.tight_layout()
    else:
        fig, ax = plt.subplots(1, 1, figsize=(6, 5))
        if args.labels == "pseudo":
            plot_scatter(
                ax,
                coords,
                pseudo_labels,
                title_pseudo,
                args.method,
                colorbar=args.colorbar,
            )
        else:
            plot_scatter(
                ax,
                coords,
                cluster_labels,
                title_pred,
                args.method,
                colorbar=args.colorbar,
            )
        fig.tight_layout()

    output_path = resolve_path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200)
    print(f"Saved plot to {output_path}")


if __name__ == "__main__":
    main()
