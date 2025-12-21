import argparse
import logging
from typing import Dict

import numpy as np
import scanpy as sc
import scipy.sparse as sp

from pipeline_utils import (
    load_config,
    resolve_path,
    save_csr_matrix,
    save_json,
)


logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build graphs for paul15.")
    parser.add_argument("--config", required=True, help="Path to YAML config.")
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Rebuild graph artifacts even if they already exist.",
    )
    return parser.parse_args()


def pca_and_neighbors(adata: sc.AnnData, graph_cfg: Dict) -> None:
    scale_cfg = graph_cfg.get("scale", {})
    if scale_cfg.get("enabled", False):
        logging.info("Scaling HVG matrix before PCA (max_value=%s)", scale_cfg.get("max_value"))
        sc.pp.scale(adata, max_value=scale_cfg.get("max_value"))

    pca_cfg = graph_cfg.get("pca", {})
    n_comps = pca_cfg.get("n_comps", 50)
    logging.info("Running PCA with %d components", n_comps)
    sc.tl.pca(
        adata,
        n_comps=n_comps,
        use_highly_variable=False,
        svd_solver="arpack",
        random_state=pca_cfg.get("random_state", 0),
    )

    neigh_cfg = graph_cfg.get("neighbors", {})
    method = neigh_cfg.get("method", "gauss")
    logging.info("Building kNN graph (k=%s, method=%s)", neigh_cfg.get("n_neighbors", 15), method)
    sc.pp.neighbors(
        adata,
        n_neighbors=neigh_cfg.get("n_neighbors", 15),
        n_pcs=n_comps,
        metric=neigh_cfg.get("metric", "euclidean"),
        method=method,
    )


def main():
    args = parse_args()
    cfg = load_config(args.config)
    graph_cfg = cfg["graph"]
    artifacts = cfg["artifacts"]["graphs"]
    required_outputs = [
        artifacts["cell_gene_adj_path"],
        artifacts["full_adj_path"],
        artifacts["features_path"],
        artifacts["meta_path"],
    ]
    if not args.overwrite and all(resolve_path(p).exists() for p in required_outputs):
        logging.info("Found cached graph artifacts; skipping build.")
        return

    processed_path = resolve_path(cfg["artifacts"]["processed_path"])
    logging.info("Loading preprocessed AnnData from %s", processed_path)
    adata = sc.read(processed_path)

    hvgs = adata[:, adata.var["highly_variable"].values].copy()
    logging.info("Using %d HVGs for graph construction", hvgs.n_vars)

    pca_and_neighbors(hvgs, graph_cfg)

    cell_gene = sp.csr_matrix(hvgs.X)
    save_csr_matrix(cell_gene, artifacts["cell_gene_adj_path"])

    top_block = sp.hstack([sp.csr_matrix((hvgs.n_obs, hvgs.n_obs)), cell_gene])
    bottom_block = sp.hstack([cell_gene.T, sp.csr_matrix((hvgs.n_vars, hvgs.n_vars))])
    full_adj = sp.vstack([top_block, bottom_block]).tocsr()
    save_csr_matrix(full_adj, artifacts["full_adj_path"])

    features_path = resolve_path(artifacts["features_path"])
    features_path.parent.mkdir(parents=True, exist_ok=True)
    cell_features = np.asarray(hvgs.obsm["X_pca"], dtype=np.float32)
    gene_features = np.asarray(hvgs.varm["PCs"], dtype=np.float32)
    np.savez_compressed(
        features_path,
        cell_features=cell_features,
        gene_features=gene_features,
        cell_names=hvgs.obs_names.values,
        gene_names=hvgs.var_names.values,
    )

    meta = {
        "dataset": cfg["dataset"]["name"],
        "n_cells": int(hvgs.n_obs),
        "n_genes": int(hvgs.n_vars),
        "adjacency_shape": [int(full_adj.shape[0]), int(full_adj.shape[1])],
        "pca_components": int(graph_cfg.get("pca", {}).get("n_comps", 50)),
    }
    save_json(meta, artifacts["meta_path"])
    logging.info("Graph artifacts written to %s", resolve_path(artifacts["cell_gene_adj_path"]).parent)


if __name__ == "__main__":
    main()
