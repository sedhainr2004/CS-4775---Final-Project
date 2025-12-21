import scanpy as sc
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (adjusted_rand_score, normalized_mutual_info_score, 
                            confusion_matrix, silhouette_score)
from sklearn.decomposition import PCA
from scipy import sparse
import os

from SIMLR_implementation import SIMLR

# Set style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")


def create_process_visualizations(simlr, dataset_name, output_dir):
    """
    Figure Set 1: SIMLR Algorithm Process
    """
    fig = plt.figure(figsize=(12, 5))

    # 1. Objective Function Convergence
    ax1 = plt.subplot(1, 2, 1)
    obj_vals = simlr.history_['objective']
    ax1.plot(range(1, len(obj_vals)+1), obj_vals, marker='o', linewidth=2,
            markersize=4, color='#3498db')
    ax1.set_xlabel('Iteration', fontsize=11)
    ax1.set_ylabel('Objective Function Value', fontsize=11)
    ax1.set_title('Objective Function Convergence', fontsize=12, fontweight='bold')
    ax1.grid(True, alpha=0.3)

    # 2. S Matrix Change (Frobenius Norm)
    ax2 = plt.subplot(1, 2, 2)
    S_changes = simlr.history_['S_changes']
    ax2.semilogy(range(1, len(S_changes)+1), S_changes, marker='s', linewidth=2,
                markersize=4, color='#2ecc71')
    ax2.set_xlabel('Iteration', fontsize=11)
    ax2.set_ylabel('||S(t) - S(t-1)||_F', fontsize=11)
    ax2.set_title('Similarity Matrix Convergence', fontsize=12, fontweight='bold')
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    output_file = os.path.join(output_dir, f'{dataset_name}_process.png')
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()


def create_results_visualizations(X, simlr, y_true, y_pred, embedding, dataset_name, output_dir):
    """
    Figure Set 2: Results and Performance
    """

    # Calculate metrics
    ari = adjusted_rand_score(y_true, y_pred)
    nmi = normalized_mutual_info_score(y_true, y_pred)
    silhouette = silhouette_score(embedding, y_pred)

    # Calculate purity for summary
    cluster_purity = []
    for cluster in sorted(np.unique(y_pred)):
        mask = y_pred == cluster
        if np.sum(mask) > 0:
            true_labels_in_cluster = y_true[mask]
            most_common = pd.Series(true_labels_in_cluster).mode()[0]
            purity = np.sum(true_labels_in_cluster == most_common) / len(true_labels_in_cluster)
            cluster_purity.append(purity)
        else:
            cluster_purity.append(0)

    # Print summary statistics
    print(f"""
    SIMLR Results Summary
    {'='*50}

    Dataset: {dataset_name}
    Cells: {X.shape[0]:,}
    Genes: {X.shape[1]:,}
    True Clusters: {len(np.unique(y_true))}
    Predicted Clusters: {len(np.unique(y_pred))}

    Performance Metrics:
    {'─'*50}
    ARI:                {ari:.4f}
    NMI:                {nmi:.4f}
    Silhouette:         {silhouette:.4f}
    Mean Purity:        {np.mean(cluster_purity):.4f}

    Algorithm Parameters:
    {'─'*50}
    Iterations:         {simlr.n_iter}
    Beta (β):           {simlr.beta}
    Gamma (γ):          {simlr.gamma}
    Rho (ρ):            {simlr.rho}
    """)

    fig = plt.figure(figsize=(12, 5))

    # 1. t-SNE: True Labels
    ax1 = plt.subplot(1, 2, 1)
    scatter1 = ax1.scatter(embedding[:, 0], embedding[:, 1], c=y_true,
                          cmap='tab20', s=15, alpha=0.6, edgecolors='none')
    ax1.set_title(f'True Cell Types (n={len(np.unique(y_true))})',
                 fontsize=12, fontweight='bold')
    ax1.set_xlabel('t-SNE 1', fontsize=11)
    ax1.set_ylabel('t-SNE 2', fontsize=11)
    plt.colorbar(scatter1, ax=ax1, label='True Cluster')

    # 2. t-SNE: SIMLR Predictions
    ax2 = plt.subplot(1, 2, 2)
    scatter2 = ax2.scatter(embedding[:, 0], embedding[:, 1], c=y_pred,
                          cmap='tab20', s=15, alpha=0.6, edgecolors='none')
    ax2.set_title(f'SIMLR Predictions (ARI={ari:.3f})',
                 fontsize=12, fontweight='bold')
    ax2.set_xlabel('t-SNE 1', fontsize=11)
    ax2.set_ylabel('t-SNE 2', fontsize=11)
    plt.colorbar(scatter2, ax=ax2, label='Predicted Cluster')

    plt.tight_layout()
    output_file = os.path.join(output_dir, f'{dataset_name}_results.png')
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()

    return ari, nmi, silhouette


def process_dataset(dataset_path, dataset_name, output_dir):
    """
    Complete pipeline for one dataset
    """
    print(f"\n{'='*60}")
    print(f"Processing: {dataset_name}")
    print('='*60)
    
    # Load data
    adata = sc.read_h5ad(dataset_path)
    print(f"Loaded: {adata.n_obs} cells × {adata.n_vars} genes")
    
    if sparse.issparse(adata.X):
        X = adata.X.toarray()
    else:
        X = adata.X
    
    y_true = adata.obs['cell_type'].values.astype(int)
    n_clusters = len(np.unique(y_true))
    
    best_beta = 0.8
    best_gamma = 0.8
    best_rho = 0.5
    if dataset_name.lower() == "paul15":
        best_beta = 1.0
        best_gamma = 1.0
        best_rho = 1.0

    simlr = SIMLR(n_clusters=n_clusters, n_iterations=30, beta=best_beta, gamma=best_gamma, rho=best_rho, logging=False)
    embedding = simlr.fit_transform(X, embedding_dim=2)
    y_pred = simlr.predict()
    
    # Create visualizations
    create_process_visualizations(simlr, dataset_name, output_dir)
    metrics = create_results_visualizations(X, simlr, y_true, y_pred, 
                                          embedding, dataset_name, output_dir)
    
    return metrics


def create_comparison_figure(results, output_dir):
    """
    Create overall comparison figure
    """
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    datasets = list(results.keys())
    
    # ARI
    ari_scores = [results[d]['ARI'] for d in datasets]
    axes[0].bar(datasets, ari_scores, color='#3498db', alpha=0.7, edgecolor='black')
    axes[0].set_ylabel('Score', fontsize=12)
    axes[0].set_title('Adjusted Rand Index (ARI)', fontsize=14, fontweight='bold')
    axes[0].set_ylim([0, 1])
    axes[0].axhline(y=0.5, color='gray', linestyle='--', alpha=0.5)
    axes[0].grid(True, alpha=0.3, axis='y')
    for i, v in enumerate(ari_scores):
        axes[0].text(i, v + 0.02, f'{v:.4f}', ha='center', fontweight='bold')
    
    # NMI
    nmi_scores = [results[d]['NMI'] for d in datasets]
    axes[1].bar(datasets, nmi_scores, color='#e74c3c', alpha=0.7, edgecolor='black')
    axes[1].set_ylabel('Score', fontsize=12)
    axes[1].set_title('Normalized Mutual Information (NMI)', fontsize=14, fontweight='bold')
    axes[1].set_ylim([0, 1])
    axes[1].axhline(y=0.5, color='gray', linestyle='--', alpha=0.5)
    axes[1].grid(True, alpha=0.3, axis='y')
    for i, v in enumerate(nmi_scores):
        axes[1].text(i, v + 0.02, f'{v:.4f}', ha='center', fontweight='bold')
    
    # Silhouette
    sil_scores = [results[d]['Silhouette'] for d in datasets]
    axes[2].bar(datasets, sil_scores, color='#2ecc71', alpha=0.7, edgecolor='black')
    axes[2].set_ylabel('Score', fontsize=12)
    axes[2].set_title('Silhouette Score', fontsize=14, fontweight='bold')
    axes[2].set_ylim([0, 1])
    axes[2].axhline(y=0.5, color='gray', linestyle='--', alpha=0.5)
    axes[2].grid(True, alpha=0.3, axis='y')
    for i, v in enumerate(sil_scores):
        axes[2].text(i, v + 0.02, f'{v:.4f}', ha='center', fontweight='bold')
    
    plt.tight_layout()
    output_file = os.path.join(output_dir, 'overall_comparison.png')
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()


if __name__ == "__main__":
    output_dir = 'simlr/figures'
    os.makedirs(output_dir, exist_ok=True)
    
    results = {}
    
    # Process PBMC3k
    metrics = process_dataset("data/pbmc3k.h5ad", "pbmc3k", output_dir)
    results['PBMC3k'] = {'ARI': metrics[0], 'NMI': metrics[1], 'Silhouette': metrics[2]}
    
    # Process Paul15
    metrics = process_dataset("data/paul15.h5ad", "paul15", output_dir)
    results['Paul15'] = {'ARI': metrics[0], 'NMI': metrics[1], 'Silhouette': metrics[2]}
    
    # Create comparison
    create_comparison_figure(results, output_dir)
    
    print(f"\nFigures saved in: {output_dir}/")