from __future__ import annotations

from typing import Iterable, Tuple

import torch
from torch import nn
import torch.nn.functional as F


class BipartiteGraphConvolution(nn.Module):
    """Message passing between cell and gene nodes in a bipartite graph."""

    def __init__(
        self,
        cell_in: int,
        gene_in: int,
        cell_out: int,
        gene_out: int,
        dropout: float = 0.0,
        activation: str = "relu",
    ) -> None:
        super().__init__()
        self.cell_self = nn.Linear(cell_in, cell_out)
        self.cell_neigh = nn.Linear(gene_in, cell_out)
        self.gene_self = nn.Linear(gene_in, gene_out)
        self.gene_neigh = nn.Linear(cell_in, gene_out)
        self.dropout = nn.Dropout(dropout)
        self.activation = getattr(F, activation) if activation else None

    def forward(
        self, cell_x: torch.Tensor, gene_x: torch.Tensor, adjacency: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        cell_neighbors = torch.sparse.mm(adjacency, gene_x)
        gene_neighbors = torch.sparse.mm(adjacency.transpose(0, 1), cell_x)

        cell_out = self.cell_self(cell_x) + self.cell_neigh(cell_neighbors)
        gene_out = self.gene_self(gene_x) + self.gene_neigh(gene_neighbors)

        if self.activation is not None:
            cell_out = self.activation(cell_out)
            gene_out = self.activation(gene_out)

        cell_out = self.dropout(cell_out)
        gene_out = self.dropout(gene_out)
        return cell_out, gene_out


class ScGNN(nn.Module):
    """Simplified scGNN model with bipartite convolutions and autoencoder heads."""

    def __init__(
        self,
        cell_feat_dim: int,
        gene_feat_dim: int,
        n_genes: int,
        n_cells: int,
        hidden_dims: Iterable[int],
        latent_dim: int,
        n_clusters: int,
        dropout: float = 0.0,
        activation: str = "relu",
        cluster_alpha: float = 1.0,
    ) -> None:
        super().__init__()
        dims = list(hidden_dims)
        prev_cell, prev_gene = cell_feat_dim, gene_feat_dim
        layers = []
        for dim in dims:
            layers.append(
                BipartiteGraphConvolution(
                    prev_cell,
                    prev_gene,
                    dim,
                    dim,
                    dropout=dropout,
                    activation=activation,
                )
            )
            prev_cell = prev_gene = dim
        self.layers = nn.ModuleList(layers)
        self.cell_latent = nn.Linear(prev_cell, latent_dim)
        self.gene_latent = nn.Linear(prev_gene, latent_dim)
        self.cell_decoder = nn.Linear(latent_dim, n_genes)
        self.gene_decoder = nn.Linear(latent_dim, n_cells)
        self.cluster_centers = nn.Parameter(torch.randn(n_clusters, latent_dim))
        self.cluster_alpha = cluster_alpha

    def encode(self, cell_x: torch.Tensor, gene_x: torch.Tensor, adjacency: torch.Tensor):
        for layer in self.layers:
            cell_x, gene_x = layer(cell_x, gene_x, adjacency)
        z_cells = self.cell_latent(cell_x)
        z_genes = self.gene_latent(gene_x)
        return z_cells, z_genes

    def decode(self, z_cells: torch.Tensor, z_genes: torch.Tensor):
        cell_recon = self.cell_decoder(z_cells)
        gene_recon = self.gene_decoder(z_genes)
        return cell_recon, gene_recon

    def forward(
        self, cell_x: torch.Tensor, gene_x: torch.Tensor, adjacency: torch.Tensor
    ) -> dict:
        z_cells, z_genes = self.encode(cell_x, gene_x, adjacency)
        cell_recon, gene_recon = self.decode(z_cells, z_genes)
        q = self._cluster_probabilities(z_cells)
        return {
            "cell_latent": z_cells,
            "gene_latent": z_genes,
            "cell_recon": cell_recon,
            "gene_recon": gene_recon,
            "cluster_probs": q,
        }

    def _cluster_probabilities(self, embeddings: torch.Tensor) -> torch.Tensor:
        squared_dist = torch.cdist(embeddings, self.cluster_centers) ** 2
        numerator = (1.0 + squared_dist / self.cluster_alpha) ** (-(self.cluster_alpha + 1.0) / 2.0)
        q = numerator / numerator.sum(dim=1, keepdim=True)
        return q

    @torch.no_grad()
    def set_cluster_centers(self, centers: torch.Tensor) -> None:
        """Initialize cluster centers from an external source (e.g., k-means)."""
        if centers.shape != self.cluster_centers.shape:
            raise ValueError(f"Center shape {centers.shape} does not match {self.cluster_centers.shape}")
        self.cluster_centers.data.copy_(centers)

    @staticmethod
    def target_distribution(q: torch.Tensor) -> torch.Tensor:
        weight = (q**2) / q.sum(dim=0, keepdim=True)
        return weight / weight.sum(dim=1, keepdim=True)
