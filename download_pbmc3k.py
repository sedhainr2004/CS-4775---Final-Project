import scanpy as sc
from sklearn.cluster import KMeans
import os

os.makedirs("data", exist_ok=True)

adata = sc.datasets.pbmc3k()

sc.pp.normalize_total(adata, target_sum=1e4)
sc.pp.log1p(adata)
sc.pp.highly_variable_genes(adata, n_top_genes=2000, subset=True)
sc.pp.pca(adata, n_comps=50)

k = 8
km = KMeans(n_clusters=k, n_init=20, random_state=42).fit(adata.obsm["X_pca"])
adata.obs["cell_type"] = km.labels_.astype(str)
adata.obs["batch"] = "pbmc3k"

adata.write_h5ad("data/pbmc3k.h5ad")