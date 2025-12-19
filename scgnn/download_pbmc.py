import os

import scanpy as sc


def main():
    os.makedirs("data", exist_ok=True)
    print("Downloading pbmc3k dataset from Scanpy...")
    adata = sc.datasets.pbmc3k()

    cluster_input = adata.copy()
    sc.pp.normalize_total(cluster_input, target_sum=1e4)
    sc.pp.log1p(cluster_input)
    sc.pp.highly_variable_genes(cluster_input, n_top_genes=2000, subset=True)
    sc.pp.pca(cluster_input, n_comps=50)
    sc.pp.neighbors(cluster_input, n_pcs=30)
    sc.tl.leiden(cluster_input, key_added="pbmc3k_leiden")

    adata.obs["cell_type"] = cluster_input.obs["pbmc3k_leiden"].astype(str)
    adata.obs["batch"] = "pbmc3k"

    output_path = "data/pbmc3k_raw.h5ad"
    adata.write_h5ad(output_path)
    print(f"Saved raw dataset with pseudo labels to {output_path}")


if __name__ == "__main__":
    main()
