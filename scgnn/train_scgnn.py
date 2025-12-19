import argparse
import json
import logging
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, silhouette_score

from data.scgnn_dataset import load_scgnn_data
from models.scgnn import ScGNN
from pipeline_utils import load_config, resolve_path


logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train scGNN on preprocessed artifacts.")
    parser.add_argument("--config", required=True, help="Path to YAML configuration file.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing outputs.")
    return parser.parse_args()


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)


def build_model(model_cfg: Dict, n_cells: int, n_genes: int, device: torch.device) -> ScGNN:
    model = ScGNN(
        cell_feat_dim=model_cfg["cell_feature_dim"],
        gene_feat_dim=model_cfg["gene_feature_dim"],
        n_genes=n_genes,
        n_cells=n_cells,
        hidden_dims=model_cfg["hidden_dims"],
        latent_dim=model_cfg["latent_dim"],
        n_clusters=model_cfg["n_clusters"],
        dropout=model_cfg.get("dropout", 0.0),
        activation=model_cfg.get("activation", "relu"),
        cluster_alpha=model_cfg.get("cluster_alpha", 1.0),
    )
    return model.to(device)


def reconstruction_loss(pred: torch.Tensor, target: torch.Tensor, loss_type: str) -> torch.Tensor:
    if loss_type == "l1":
        return F.l1_loss(pred, target)
    return F.mse_loss(pred, target)


def clustering_loss(q: torch.Tensor, p: torch.Tensor) -> torch.Tensor:
    eps = 1e-8
    return torch.sum(p * torch.log((p + eps) / (q + eps))) / q.size(0)


def evaluate_clusters(latent: np.ndarray, preds: np.ndarray, labels: np.ndarray) -> Dict[str, float]:
    metrics = {}
    if len(np.unique(preds)) > 1 and len(np.unique(labels)) > 1:
        metrics["silhouette"] = float(silhouette_score(latent, preds))
    metrics["ari"] = float(adjusted_rand_score(labels, preds))
    metrics["nmi"] = float(normalized_mutual_info_score(labels, preds))
    return metrics


def init_clusters_with_kmeans(model: ScGNN, z_cells: torch.Tensor, n_clusters: int, device: torch.device):
    z_np = z_cells.detach().cpu().numpy()
    km = KMeans(n_clusters=n_clusters, n_init=20, random_state=42)
    km.fit(z_np)
    centers = torch.tensor(km.cluster_centers_, dtype=torch.float32, device=device)
    model.set_cluster_centers(centers)
    return km.labels_


def cluster_entropy(q: torch.Tensor) -> float:
    probs = q.detach().cpu().numpy()
    eps = 1e-8
    ent = -np.sum(probs * np.log(probs + eps), axis=1).mean()
    return float(ent)


def save_outputs(output_dir: Path, outputs: Dict[str, np.ndarray], metrics: Dict[str, List[Dict[str, float]]]):
    output_dir.mkdir(parents=True, exist_ok=True)
    np.save(output_dir / "cell_embeddings.npy", outputs["cell_embeddings"])
    np.save(output_dir / "gene_embeddings.npy", outputs["gene_embeddings"])
    np.save(output_dir / "cluster_labels.npy", outputs["cluster_labels"])
    np.savez_compressed(
        output_dir / "reconstructions.npz",
        cell_recon=outputs["cell_recon"],
        gene_recon=outputs["gene_recon"],
    )
    with open(output_dir / "metrics.json", "w") as handle:
        json.dump(metrics, handle, indent=2)


def main():
    args = parse_args()
    cfg = load_config(args.config)
    training_cfg = cfg["training"]
    model_cfg = cfg["model"]
    dataset_cfg = cfg["dataset"]

    output_dir = resolve_path(training_cfg["output_dir"])
    if output_dir.exists() and not args.overwrite:
        logging.info("Outputs already exist at %s; use --overwrite to retrain.", output_dir)
        return

    device = torch.device(training_cfg.get("device", "cpu"))
    set_seed(training_cfg.get("seed", 42))

    data = load_scgnn_data(dataset_cfg)
    model_cfg["cell_feature_dim"] = data.cell_features.shape[1]
    model_cfg["gene_feature_dim"] = data.gene_features.shape[1]

    model = build_model(model_cfg, n_cells=data.expression.shape[0], n_genes=data.expression.shape[1], device=device)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=training_cfg["learning_rate"],
        weight_decay=training_cfg.get("weight_decay", 0.0),
    )

    cell_features = data.cell_features.to(device)
    gene_features = data.gene_features.to(device)
    adjacency = data.adjacency.coalesce().to(device)
    expression = data.expression.to(device)

    recon_weight = cfg["losses"]["reconstruction_weight"]
    cluster_weight = cfg["losses"]["clustering_weight"]
    loss_type = cfg["losses"].get("reconstruction_type", "mse")

    pretrain_epochs = training_cfg.get("pretrain_epochs", 0)
    cluster_start = training_cfg.get("cluster_start_epoch", pretrain_epochs + 1)
    patience = training_cfg.get("early_stopping_patience", 0)
    min_delta = training_cfg.get("early_stopping_min_delta", 0.0)
    best_val_ari = -np.inf
    best_payload = None
    best_epoch = 0

    history = []
    for epoch in range(1, training_cfg["epochs"] + 1):
        model.train()
        outputs = model(cell_features, gene_features, adjacency)
        recon = reconstruction_loss(outputs["cell_recon"], expression, loss_type)
        q = outputs["cluster_probs"]
        p = model.target_distribution(q.detach())

        use_cluster_loss = epoch >= cluster_start
        clust = clustering_loss(q, p) if use_cluster_loss else torch.tensor(0.0, device=device)
        
        # Store raw cluster loss value for logging before weighting
        clust_raw = float(clust.item())
        clust_weighted = (cluster_weight * clust) if use_cluster_loss else torch.tensor(0.0, device=device)
        
        loss = recon_weight * recon + clust_weighted

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        metrics = {
            "epoch": epoch,
            "loss": float(loss.item()),
            "recon_loss": float(recon.item()),
            "recon_loss_weighted": float((recon_weight * recon).item()),
            "cluster_loss": clust_raw,
            "cluster_loss_weighted": float(clust_weighted.item()) if use_cluster_loss else 0.0,
            "cluster_entropy": cluster_entropy(q),
            "cluster_active": use_cluster_loss,
        }

        if epoch == pretrain_epochs and training_cfg.get("kmeans_init", True):
            model.eval()
            with torch.no_grad():
                pre_outputs = model(cell_features, gene_features, adjacency)
                init_clusters_with_kmeans(model, pre_outputs["cell_latent"], model_cfg["n_clusters"], device)

        eval_now = epoch % training_cfg.get("eval_every", 10) == 0 or epoch == training_cfg["epochs"]
        if eval_now:
            model.eval()
            with torch.no_grad():
                eval_outputs = model(cell_features, gene_features, adjacency)
                cell_latent = eval_outputs["cell_latent"].cpu().numpy()
                cluster_labels = torch.argmax(eval_outputs["cluster_probs"], dim=1).cpu().numpy()
                train_idx = data.splits["train"].cpu().numpy()
                val_idx = data.splits["val"].cpu().numpy()
                test_idx = data.splits["test"].cpu().numpy()
                metrics["train"] = evaluate_clusters(
                    cell_latent[train_idx],
                    cluster_labels[train_idx],
                    data.pseudo_labels[train_idx].cpu().numpy(),
                )
                metrics["val"] = evaluate_clusters(
                    cell_latent[val_idx],
                    cluster_labels[val_idx],
                    data.pseudo_labels[val_idx].cpu().numpy(),
                )
                metrics["test"] = evaluate_clusters(
                    cell_latent[test_idx],
                    cluster_labels[test_idx],
                    data.pseudo_labels[test_idx].cpu().numpy(),
                )

                if metrics["val"].get("ari", -np.inf) > best_val_ari + min_delta:
                    best_val_ari = metrics["val"]["ari"]
                    best_epoch = epoch
                    best_payload = {
                        "cell_embeddings": cell_latent,
                        "gene_embeddings": eval_outputs["gene_latent"].cpu().numpy(),
                        "cluster_labels": cluster_labels,
                        "cell_recon": eval_outputs["cell_recon"].cpu().numpy(),
                        "gene_recon": eval_outputs["gene_recon"].cpu().numpy(),
                        "metrics": metrics,
                    }
        history.append(metrics)
        logging.info(
            "Epoch %d/%d -- loss %.4f (recon %.4f, cluster %.6f) entropy %.4f [cluster %s]",
            epoch,
            training_cfg["epochs"],
            metrics["loss"],
            metrics["recon_loss"],
            metrics["cluster_loss"],
            metrics["cluster_entropy"],
            "ON" if use_cluster_loss else "OFF",
        )
        if patience and (epoch - best_epoch) >= patience and best_epoch > 0:
            logging.info("Early stopping at epoch %d (best val ARI %.4f at epoch %d)", epoch, best_val_ari, best_epoch)
            break

    if best_payload is None:
        model.eval()
        with torch.no_grad():
            final = model(cell_features, gene_features, adjacency)
            best_payload = {
                "cell_embeddings": final["cell_latent"].cpu().numpy(),
                "gene_embeddings": final["gene_latent"].cpu().numpy(),
                "cluster_labels": torch.argmax(final["cluster_probs"], dim=1).cpu().numpy(),
                "cell_recon": final["cell_recon"].cpu().numpy(),
                "gene_recon": final["gene_recon"].cpu().numpy(),
                "metrics": {},
            }

    outputs = {
        "cell_embeddings": best_payload["cell_embeddings"],
        "gene_embeddings": best_payload["gene_embeddings"],
        "cluster_labels": best_payload["cluster_labels"],
        "cell_recon": best_payload["cell_recon"],
        "gene_recon": best_payload["gene_recon"],
    }
    metrics_payload = {"history": history, "best": best_payload.get("metrics", {}), "best_epoch": best_epoch}
    save_outputs(output_dir, outputs, metrics_payload)
    logging.info("Training artifacts saved to %s (best epoch %d)", output_dir, best_epoch)


if __name__ == "__main__":
    main()