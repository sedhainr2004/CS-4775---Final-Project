"""
Hyperparameter Exploration for SIMLR
Tests different values of beta (=gamma) and rho to find optimal configuration
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import make_blobs
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, silhouette_score
from itertools import product
import scanpy as sc
from scipy import sparse

from SIMLR_implementation import SIMLR

plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")


def run_parameter_exploration(X, y_true, n_clusters, dataset_name="dataset"):
    """
    Run grid search over beta and rho parameters

    Args:
        X: Data matrix (n_samples x n_features)
        y_true: True labels
        n_clusters: Number of clusters
        dataset_name: Name for output files
    """
    beta_values = [0.5, 0.8, 1.0]
    rho_values = [0.5, 1.0, 2.0]

    print("="*60)
    print("SIMLR Hyperparameter Exploration")
    print("="*60)
    print(f"Dataset: {dataset_name}")
    print(f"Samples: {X.shape[0]}, Features: {X.shape[1]}, Clusters: {n_clusters}")
    print(f"\nParameter Grid:")
    print(f"  Beta (=Gamma): {beta_values}")
    print(f"  Rho: {rho_values}")
    print(f"  Total configurations: {len(beta_values) * len(rho_values)}")
    print("="*60)

    results = []
    for i, (beta, rho) in enumerate(product(beta_values, rho_values), 1):
        print(f"\n[{i}/{len(beta_values) * len(rho_values)}] Testing beta={beta}, gamma={beta}, rho={rho}")

        simlr = SIMLR(
            n_clusters=n_clusters,
            beta=beta,
            gamma=beta,
            rho=rho,
            n_iterations=30,
            logging=False
        )

        embedding = simlr.fit_transform(X, embedding_dim=2)
        y_pred = simlr.predict()

        ari = adjusted_rand_score(y_true, y_pred)
        nmi = normalized_mutual_info_score(y_true, y_pred)
        sil = silhouette_score(embedding, y_pred)

        results.append({
            'beta': beta,
            'gamma': beta,
            'rho': rho,
            'ARI': ari,
            'NMI': nmi,
            'Silhouette': sil
        })

        print(f"  ARI: {ari:.4f}, NMI: {nmi:.4f}, Silhouette: {sil:.4f}")

    results_df = pd.DataFrame(results)

    print("\n" + "="*60)
    print("BEST CONFIGURATIONS")
    print("="*60)

    best_ari_idx = results_df['ARI'].idxmax()
    best_nmi_idx = results_df['NMI'].idxmax()
    best_sil_idx = results_df['Silhouette'].idxmax()

    print(f"\nBest ARI ({results_df.loc[best_ari_idx, 'ARI']:.4f}):")
    print(f"  beta={results_df.loc[best_ari_idx, 'beta']}, "
          f"gamma={results_df.loc[best_ari_idx, 'gamma']}, "
          f"rho={results_df.loc[best_ari_idx, 'rho']}")

    print(f"\nBest NMI ({results_df.loc[best_nmi_idx, 'NMI']:.4f}):")
    print(f"  beta={results_df.loc[best_nmi_idx, 'beta']}, "
          f"gamma={results_df.loc[best_nmi_idx, 'gamma']}, "
          f"rho={results_df.loc[best_nmi_idx, 'rho']}")

    print(f"\nBest Silhouette ({results_df.loc[best_sil_idx, 'Silhouette']:.4f}):")
    print(f"  beta={results_df.loc[best_sil_idx, 'beta']}, "
          f"gamma={results_df.loc[best_sil_idx, 'gamma']}, "
          f"rho={results_df.loc[best_sil_idx, 'rho']}")

    # Overall best (average rank across metrics)
    results_df['ARI_rank'] = results_df['ARI'].rank(ascending=False)
    results_df['NMI_rank'] = results_df['NMI'].rank(ascending=False)
    results_df['Sil_rank'] = results_df['Silhouette'].rank(ascending=False)
    results_df['Avg_rank'] = (results_df['ARI_rank'] + results_df['NMI_rank'] + results_df['Sil_rank']) / 3

    best_overall_idx = results_df['Avg_rank'].idxmin()
    print(f"\nBest Overall (by average rank):")
    print(f"  beta={results_df.loc[best_overall_idx, 'beta']}, "
          f"gamma={results_df.loc[best_overall_idx, 'gamma']}, "
          f"rho={results_df.loc[best_overall_idx, 'rho']}")
    print(f"  ARI={results_df.loc[best_overall_idx, 'ARI']:.4f}, "
          f"NMI={results_df.loc[best_overall_idx, 'NMI']:.4f}, "
          f"Silhouette={results_df.loc[best_overall_idx, 'Silhouette']:.4f}")

    output_file = f'{dataset_name}_param_exploration.csv'
    results_df.to_csv(output_file, index=False)
    print(f"\nResults saved to: {output_file}")

    return results_df


if __name__ == "__main__":
    print("### PAUL15 DATA ###")
    adata = sc.read_h5ad("data/paul15.h5ad")
    X_pbmc = adata.X.toarray() if sparse.issparse(adata.X) else adata.X
    y_pbmc = adata.obs['cell_type'].values.astype(int)
    n_clusters_pbmc = len(np.unique(y_pbmc))
    results_pbmc = run_parameter_exploration(X_pbmc, y_pbmc, n_clusters=n_clusters_pbmc, dataset_name="paul15")