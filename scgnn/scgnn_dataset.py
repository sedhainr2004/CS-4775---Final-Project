from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Dict, List

import numpy as np
import scanpy as sc
import scipy.sparse as sp
import torch

from pipeline_utils import resolve_path


@dataclass
class ScGNNData:
    cell_features: torch.Tensor
    gene_features: torch.Tensor
    adjacency: torch.Tensor
    expression: torch.Tensor
    splits: Dict[str, torch.Tensor]
    pseudo_labels: torch.Tensor
    label_names: List[str]
    cell_names: List[str]
    gene_names: List[str]


def _load_csr_matrix(path: str) -> sp.csr_matrix:
    npz = np.load(resolve_path(path))
    return sp.csr_matrix((npz["data"], npz["indices"], npz["indptr"]), shape=tuple(npz["shape"]))


def _csr_to_torch_sparse(matrix: sp.csr_matrix) -> torch.Tensor:
    coo = matrix.tocoo()
    indices = torch.tensor(np.vstack([coo.row, coo.col]), dtype=torch.long)
    values = torch.tensor(coo.data, dtype=torch.float32)
    shape = torch.Size(coo.shape)
    return torch.sparse_coo_tensor(indices, values, shape).coalesce()


def _load_expression_matrix(
    adata_path: str, cell_names: np.ndarray, gene_names: np.ndarray, pseudo_label_key: str
):
    adata = sc.read_h5ad(resolve_path(adata_path))
    adata = adata[cell_names, gene_names]
    matrix = adata.X
    if sp.issparse(matrix):
        matrix = matrix.toarray()
    expr = torch.tensor(matrix, dtype=torch.float32)

    pseudo_labels = adata.obs[pseudo_label_key].astype("category")
    label_names = list(pseudo_labels.cat.categories)
    label_codes = torch.tensor(pseudo_labels.cat.codes.values, dtype=torch.long)
    return expr, label_codes, label_names


def load_scgnn_data(dataset_cfg: Dict) -> ScGNNData:
    features_npz = np.load(resolve_path(dataset_cfg["features_path"]), allow_pickle=True)
    cell_features = torch.tensor(features_npz["cell_features"], dtype=torch.float32)
    gene_features = torch.tensor(features_npz["gene_features"], dtype=torch.float32)
    cell_names = features_npz["cell_names"].astype(str)
    gene_names = features_npz["gene_names"].astype(str)

    adjacency = _csr_to_torch_sparse(_load_csr_matrix(dataset_cfg["cell_gene_adj_path"]))

    expression, pseudo_labels, label_names = _load_expression_matrix(
        dataset_cfg["preprocessed_with_splits_path"],
        cell_names,
        gene_names,
        dataset_cfg["pseudo_label_key"],
    )

    with open(resolve_path(dataset_cfg["splits_index_path"]), "r") as handle:
        split_info = json.load(handle)
    splits = {
        "train": torch.tensor(split_info["train_indices"], dtype=torch.long),
        "val": torch.tensor(split_info["val_indices"], dtype=torch.long),
        "test": torch.tensor(split_info["test_indices"], dtype=torch.long),
    }

    return ScGNNData(
        cell_features=cell_features,
        gene_features=gene_features,
        adjacency=adjacency,
        expression=expression,
        splits=splits,
        pseudo_labels=pseudo_labels,
        label_names=label_names,
        cell_names=cell_names.tolist(),
        gene_names=gene_names.tolist(),
    )
