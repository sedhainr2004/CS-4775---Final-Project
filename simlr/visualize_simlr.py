import scanpy as sc
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, confusion_matrix
from scipy import sparse

from SIMLR_implementation import SIMLR

def visualize_simlr_results(dataset_path, dataset_name, output_prefix):
    """
    Run SIMLR and create comprehensive visualizations.
    """
    print(f"\n{'='*50}")
    print(f"Processing {dataset_name}")
    print('='*50)
    
    # Load data
    adata = sc.read_h5ad(dataset_path)
    print(f"Loaded: {adata.n_obs} cells x {adata.n_vars} genes")
    
    # Prepare input
    if sparse.issparse(adata.X):
        X = adata.X.toarray()
    else:
        X = adata.X
    
    y_true = adata.obs['cell_type'].values.astype(int)  # Convert to int
    n_clusters = len(np.unique(y_true))
    print(f"Number of clusters: {n_clusters}")
    
    # Run SIMLR
    print("Fitting SIMLR...")
    simlr = SIMLR(n_clusters=n_clusters, n_iterations=30, logging=False)
    embedding = simlr.fit_transform(X, embedding_dim=2)
    y_pred = simlr.predict()
    
    # Calculate metrics
    ari = adjusted_rand_score(y_true, y_pred)
    nmi = normalized_mutual_info_score(y_true, y_pred)
    print(f"ARI: {ari:.4f}, NMI: {nmi:.4f}")
    
    # Create figure with subplots
    fig = plt.figure(figsize=(20, 10))
    
    # 1. Embedding colored by true labels
    ax1 = plt.subplot(2, 3, 1)
    scatter1 = ax1.scatter(embedding[:, 0], embedding[:, 1], 
                          c=y_true.astype(int), cmap='tab10', 
                          s=10, alpha=0.6)
    ax1.set_title(f'{dataset_name}: True Cell Types', fontsize=14, fontweight='bold')
    ax1.set_xlabel('t-SNE 1')
    ax1.set_ylabel('t-SNE 2')
    plt.colorbar(scatter1, ax=ax1, label='True Cluster')
    
    # 2. Embedding colored by predicted labels
    ax2 = plt.subplot(2, 3, 2)
    scatter2 = ax2.scatter(embedding[:, 0], embedding[:, 1], 
                          c=y_pred, cmap='tab10', 
                          s=10, alpha=0.6)
    ax2.set_title(f'{dataset_name}: SIMLR Predictions', fontsize=14, fontweight='bold')
    ax2.set_xlabel('t-SNE 1')
    ax2.set_ylabel('t-SNE 2')
    plt.colorbar(scatter2, ax=ax2, label='Predicted Cluster')
    
    # 3. Confusion Matrix
    ax3 = plt.subplot(2, 3, 3)
    cm = confusion_matrix(y_true, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax3, cbar_kws={'label': 'Count'})
    ax3.set_title('Confusion Matrix', fontsize=14, fontweight='bold')
    ax3.set_xlabel('Predicted Cluster')
    ax3.set_ylabel('True Cluster')
    
    # 4. Metrics Bar Chart
    ax4 = plt.subplot(2, 3, 4)
    metrics = ['ARI', 'NMI']
    values = [ari, nmi]
    colors = ['#2ecc71' if v > 0.5 else '#e74c3c' for v in values]
    bars = ax4.bar(metrics, values, color=colors, alpha=0.7, edgecolor='black')
    ax4.set_ylim([0, 1])
    ax4.set_ylabel('Score', fontsize=12)
    ax4.set_title('Clustering Performance Metrics', fontsize=14, fontweight='bold')
    ax4.axhline(y=0.5, color='gray', linestyle='--', linewidth=1, label='Threshold')
    ax4.legend()
    
    # Add value labels on bars
    for bar, value in zip(bars, values):
        height = bar.get_height()
        ax4.text(bar.get_x() + bar.get_width()/2., height,
                f'{value:.4f}', ha='center', va='bottom', fontweight='bold')
    
    # 5. Cluster size distribution (True vs Predicted)
    ax5 = plt.subplot(2, 3, 5)
    true_counts = pd.Series(y_true).value_counts().sort_index()
    pred_counts = pd.Series(y_pred).value_counts().sort_index()
    
    x = np.arange(len(true_counts))
    width = 0.35
    ax5.bar(x - width/2, true_counts.values, width, label='True', alpha=0.7, color='#3498db')
    ax5.bar(x + width/2, pred_counts.values, width, label='Predicted', alpha=0.7, color='#e67e22')
    ax5.set_xlabel('Cluster ID')
    ax5.set_ylabel('Number of Cells')
    ax5.set_title('Cluster Size Distribution', fontsize=14, fontweight='bold')
    ax5.legend()
    ax5.set_xticks(x)
    
    # 6. Kernel Weights Distribution
    ax6 = plt.subplot(2, 3, 6)
    if simlr.w_ is not None:
        ax6.plot(simlr.w_, marker='o', linestyle='-', color='#9b59b6', linewidth=2, markersize=4)
        ax6.set_xlabel('Kernel Index')
        ax6.set_ylabel('Weight')
        ax6.set_title('Learned Kernel Weights', fontsize=14, fontweight='bold')
        ax6.grid(True, alpha=0.3)
        ax6.set_ylim([0, max(simlr.w_) * 1.1])
    
    plt.tight_layout()
    
    # Save figure
    output_file = f'{output_prefix}_visualization.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Saved visualization: {output_file}")
    plt.close()
    
    return ari, nmi


def create_comparison_plot(results_dict, output_file='simlr/comparison_plot.png'):
    """
    Create a comparison plot across datasets.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    datasets = list(results_dict.keys())
    ari_scores = [results_dict[d]['ARI'] for d in datasets]
    nmi_scores = [results_dict[d]['NMI'] for d in datasets]
    
    x = np.arange(len(datasets))
    width = 0.35
    
    # ARI comparison
    bars1 = ax1.bar(x, ari_scores, width, color='#3498db', alpha=0.7, edgecolor='black')
    ax1.set_ylabel('ARI Score', fontsize=12)
    ax1.set_title('Adjusted Rand Index Comparison', fontsize=14, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(datasets)
    ax1.set_ylim([0, 1])
    ax1.axhline(y=0.5, color='gray', linestyle='--', linewidth=1)
    ax1.grid(True, alpha=0.3, axis='y')
    
    for bar, score in zip(bars1, ari_scores):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{score:.4f}', ha='center', va='bottom', fontweight='bold')
    
    # NMI comparison
    bars2 = ax2.bar(x, nmi_scores, width, color='#e74c3c', alpha=0.7, edgecolor='black')
    ax2.set_ylabel('NMI Score', fontsize=12)
    ax2.set_title('Normalized Mutual Information Comparison', fontsize=14, fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels(datasets)
    ax2.set_ylim([0, 1])
    ax2.axhline(y=0.5, color='gray', linestyle='--', linewidth=1)
    ax2.grid(True, alpha=0.3, axis='y')
    
    for bar, score in zip(bars2, nmi_scores):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{score:.4f}', ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"\nSaved comparison plot: {output_file}")
    plt.close()


if __name__ == "__main__":
    import os
    os.makedirs('simlr/results', exist_ok=True)
    
    results = {}
    
    # Process PBMC 3k
    ari, nmi = visualize_simlr_results(
        "data/pbmc3k.h5ad", 
        "PBMC 3k",
        "simlr/results/pbmc3k"
    )
    results['PBMC 3k'] = {'ARI': ari, 'NMI': nmi}
    
    # Process Paul15
    ari, nmi = visualize_simlr_results(
        "data/paul15.h5ad", 
        "Paul15",
        "simlr/results/paul15"
    )
    results['Paul15'] = {'ARI': ari, 'NMI': nmi}
    
    # Create comparison plot
    create_comparison_plot(results)
    
    print("\n" + "="*50)
    print("All visualizations completed!")
    print("="*50)
    print("\nResults Summary:")
    for dataset, metrics in results.items():
        print(f"{dataset:15s} - ARI: {metrics['ARI']:.4f}, NMI: {metrics['NMI']:.4f}")
