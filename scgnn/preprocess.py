import argparse
import logging
from typing import Dict, Optional

import h5py
import numpy as np
import pandas as pd
import scanpy as sc
import scipy.sparse as sp

from pipeline_utils import ensure_parent_dir, load_config, resolve_path


logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Preprocess paul15 AnnData.")
    parser.add_argument("--config", required=True, help="Path to YAML config.")
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Recompute preprocessing even if cached output exists.",
    )
    return parser.parse_args()


def load_raw_adata(dataset_cfg: Dict) -> sc.AnnData:
    raw_path = resolve_path(dataset_cfg["raw_path"])
    fmt = dataset_cfg.get("raw_format", "h5ad")
    if fmt == "h5ad":
        logging.info("Reading AnnData from %s", raw_path)
        return sc.read_h5ad(raw_path)
    if fmt == "paul15_h5":
        return _load_paul15_h5(raw_path, dataset_cfg)
    raise ValueError(f"Unsupported raw format: {fmt}")


def _load_paul15_h5(path, dataset_cfg: Dict) -> sc.AnnData:
    logging.info("Reading custom paul15 .h5 file from %s", path)
    with h5py.File(path, "r") as handle:
        matrix = handle["data"][:].T  # cells x genes
        cell_names = [c.decode("utf-8") for c in handle["data_colnames"][:]]
        gene_names = [g.decode("utf-8") for g in handle["data_rownames"][:]]
        batches = handle["batch.names"][:, 0]
        clusters = handle["cluster.id"][:, 0]

    obs = pd.DataFrame(index=cell_names)
    batch_key = dataset_cfg.get("batch_obs_key", "batch")
    obs[batch_key] = batches.astype(int).astype(str)

    cluster_key = dataset_cfg.get("raw_cluster_obs_key", "raw_cluster")
    obs[cluster_key] = clusters.astype(int).astype(str)

    pseudo_key = dataset_cfg.get("pseudo_label_obs_key")
    if pseudo_key:
        obs[pseudo_key] = obs[cluster_key]

    var = pd.DataFrame(index=gene_names)
    adata = sc.AnnData(X=sp.csr_matrix(matrix), obs=obs, var=var)
    adata.obs_names.name = "cell_id"
    adata.var_names.name = "gene_id"
    return adata


def apply_qc_filters(adata: sc.AnnData, qc_cfg: Dict) -> sc.AnnData:
    if "total_counts" not in adata.obs or "n_genes_by_counts" not in adata.obs:
        matrix = adata.X
        if sp.issparse(matrix):
            total_counts = np.asarray(matrix.sum(axis=1)).ravel()
            n_genes = np.asarray(matrix.getnnz(axis=1)).ravel()
        else:
            total_counts = np.asarray(matrix.sum(axis=1)).ravel()
            n_genes = np.asarray((matrix > 0).sum(axis=1)).ravel()
        adata.obs["total_counts"] = total_counts
        adata.obs["n_genes_by_counts"] = n_genes
    mask = np.ones(adata.n_obs, dtype=bool)
    if qc_cfg.get("min_counts_per_cell") is not None:
        mask &= adata.obs["total_counts"] >= qc_cfg["min_counts_per_cell"]
    if qc_cfg.get("max_counts_per_cell") is not None:
        mask &= adata.obs["total_counts"] <= qc_cfg["max_counts_per_cell"]
    if qc_cfg.get("min_genes_per_cell") is not None:
        mask &= adata.obs["n_genes_by_counts"] >= qc_cfg["min_genes_per_cell"]
    if qc_cfg.get("max_genes_per_cell") is not None:
        mask &= adata.obs["n_genes_by_counts"] <= qc_cfg["max_genes_per_cell"]

    logging.info("QC filtering retained %d / %d cells", mask.sum(), adata.n_obs)
    return adata[mask].copy()


def compute_hvgs(adata: sc.AnnData, hvg_cfg: Dict) -> None:
    sc.pp.highly_variable_genes(
        adata,
        n_top_genes=hvg_cfg.get("n_top_genes", 2000),
        flavor=hvg_cfg.get("flavor", "seurat"),
        batch_key=hvg_cfg.get("batch_key"),
        subset=False,
    )
    selection = hvg_cfg.get("selection", "union")
    mask = adata.var["highly_variable"]
    if hvg_cfg.get("batch_key"):
        if selection == "intersection" and "highly_variable_intersection" in adata.var:
            mask = adata.var["highly_variable_intersection"].astype(bool)
        elif selection == "union":
            mask = adata.var["highly_variable"].astype(bool)
    adata.var["highly_variable"] = mask
    adata.uns.setdefault("pipeline", {})["hvg"] = {
        "n_top_genes": int(hvg_cfg.get("n_top_genes", 2000)),
        "selection": selection,
    }
    logging.info("Marked %d genes as HVGs", int(mask.sum()))


def main():
    args = parse_args()
    cfg = load_config(args.config)
    processed_path = resolve_path(cfg["artifacts"]["processed_path"])
    if processed_path.exists() and not args.overwrite:
        logging.info("Found cached preprocessing at %s; skipping.", processed_path)
        return

    adata = load_raw_adata(cfg["dataset"])
    adata.layers["counts"] = adata.X.copy()

    qc_cfg = cfg["preprocessing"]["qc"]
    adata = apply_qc_filters(adata, qc_cfg)

    norm_cfg = cfg["preprocessing"].get("normalize", {})
    if norm_cfg.get("enabled", True):
        target_sum = norm_cfg.get("target_sum", 1e4)
        logging.info("Normalizing counts to target_sum=%s", target_sum)
        sc.pp.normalize_total(adata, target_sum=target_sum)
    else:
        logging.info("Skipping normalization step per config")

    if cfg["preprocessing"].get("log1p", True):
        logging.info("Applying log1p transform")
        sc.pp.log1p(adata)

    compute_hvgs(adata, cfg["preprocessing"]["hvg"])

    adata.obs[cfg["dataset"]["batch_obs_key"]] = adata.obs[
        cfg["dataset"]["batch_obs_key"]
    ].astype("category")
    pseudo_key = cfg["dataset"].get("pseudo_label_obs_key")
    if pseudo_key and pseudo_key in adata.obs:
        adata.obs[pseudo_key] = adata.obs[pseudo_key].astype("category")

    ensure_parent_dir(processed_path)
    logging.info("Writing preprocessed AnnData to %s", processed_path)
    adata.write(processed_path)


if __name__ == "__main__":
    main()
