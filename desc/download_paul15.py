import scanpy as sc
import numpy as np
from sklearn.cluster import KMeans
import os

os.makedirs("data", exist_ok=True)

print("Loading paul15 (mouse hematopoiesis) dataset...")
adata = sc.datasets.paul15()     # built-in real scRNA-seq dataset

# --- Preprocessing to build cell_type labels ---
sc.pp.normalize_total(adata, target_sum=1e4)
sc.pp.log1p(adata)
sc.pp.highly_variable_genes(adata, n_top_genes=2000, subset=True)
sc.pp.pca(adata, n_comps=50)

# --- K-means to generate pseudo cell-type labels ---
k = 10   # paul15 has ~19 known subtypes but 10 works well
km = KMeans(n_clusters=k, n_init=20, random_state=42).fit(adata.obsm["X_pca"])
adata.obs["cell_type"] = km.labels_.astype(str)

# --- Add single-batch metadata ---
adata.obs["batch"] = "paul15"

# --- Save final dataset ---
adata.write_h5ad("data/paul15.h5ad")
print("Saved: data/paul15.h5ad")
