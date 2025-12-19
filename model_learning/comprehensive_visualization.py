"""
Comprehensive SIMLR Visualization Suite
Generates all figures needed for presentation:
1. SIMLR Process Visualizations (Methods section)
2. Results Visualizations (Results section)
"""

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
    Shows HOW SIMLR works (for Methods section)
    """
    print(f"Creating process visualizations for {dataset_name}...")
    
    # Figure 1A: Kernel Weight Evolution
    fig = plt.figure(figsize=(20, 12))
    
    # 1. Kernel weights over iterations
    ax1 = plt.subplot(3, 3, 1)
    w_history = np.array(simlr.history_['w_history'])
    n_iters = len(w_history)
    
    # Plot evolution of top 10 kernels
    for i in range(min(10, w_history.shape[1])):
        ax1.plot(range(n_iters), w_history[:, i], alpha=0.7, linewidth=1.5)
    ax1.set_xlabel('Iteration', fontsize=11)
    ax1.set_ylabel('Kernel Weight', fontsize=11)
    ax1.set_title('Kernel Weight Evolution (Top 10)', fontsize=12, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    
    # 2. Initial vs Final Kernel Weights
    ax2 = plt.subplot(3, 3, 2)
    x = np.arange(len(simlr.w_))
    width = 0.35
    ax2.bar(x - width/2, w_history[0], width, label='Initial (Uniform)', alpha=0.7, color='lightgray')
    ax2.bar(x + width/2, simlr.w_, width, label='Final (Learned)', alpha=0.7, color='#e74c3c')
    ax2.set_xlabel('Kernel Index', fontsize=11)
    ax2.set_ylabel('Weight', fontsize=11)
    ax2.set_title('Initial vs Final Kernel Weights', fontsize=12, fontweight='bold')
    ax2.legend()
    ax2.set_xlim([-1, min(50, len(simlr.w_))])  # Show first 50
    
    # 3. Top 10 Final Kernel Weights
    ax3 = plt.subplot(3, 3, 3)
    top_indices = np.argsort(simlr.w_)[-10:][::-1]
    top_weights = simlr.w_[top_indices]
    colors_grad = plt.cm.Reds(np.linspace(0.4, 0.9, len(top_weights)))
    ax3.barh(range(len(top_weights)), top_weights, color=colors_grad, edgecolor='black')
    ax3.set_yticks(range(len(top_weights)))
    ax3.set_yticklabels([f'K{i}' for i in top_indices])
    ax3.set_xlabel('Weight', fontsize=11)
    ax3.set_title('Top 10 Learned Kernel Weights', fontsize=12, fontweight='bold')
    ax3.invert_yaxis()
    
    # 4. Objective Function Convergence
    ax4 = plt.subplot(3, 3, 4)
    obj_vals = simlr.history_['objective']
    ax4.plot(range(1, len(obj_vals)+1), obj_vals, marker='o', linewidth=2, 
            markersize=4, color='#3498db')
    ax4.set_xlabel('Iteration', fontsize=11)
    ax4.set_ylabel('Objective Function Value', fontsize=11)
    ax4.set_title('Objective Function Convergence', fontsize=12, fontweight='bold')
    ax4.grid(True, alpha=0.3)
    
    # 5. S Matrix Change (Frobenius Norm)
    ax5 = plt.subplot(3, 3, 5)
    S_changes = simlr.history_['S_changes']
    ax5.semilogy(range(1, len(S_changes)+1), S_changes, marker='s', linewidth=2, 
                markersize=4, color='#2ecc71')
    ax5.set_xlabel('Iteration', fontsize=11)
    ax5.set_ylabel('||S(t) - S(t-1)||_F', fontsize=11)
    ax5.set_title('Similarity Matrix Convergence', fontsize=12, fontweight='bold')
    ax5.grid(True, alpha=0.3)
    
    # 6. Eigenvalue Spectrum
    ax6 = plt.subplot(3, 3, 6)
    eigenvalues, _ = np.linalg.eigh(simlr.S_)
    eigenvalues = np.sort(eigenvalues)[::-1][:50]  # Top 50
    ax6.plot(range(1, len(eigenvalues)+1), eigenvalues, marker='d', linewidth=2,
            markersize=4, color='#9b59b6')
    ax6.axvline(x=simlr.C, color='red', linestyle='--', linewidth=2, label=f'k={simlr.C}')
    ax6.set_xlabel('Component', fontsize=11)
    ax6.set_ylabel('Eigenvalue', fontsize=11)
    ax6.set_title('Eigenvalue Spectrum of Final S', fontsize=12, fontweight='bold')
    ax6.legend()
    ax6.grid(True, alpha=0.3)
    
    # 7-9: Similarity Matrix Heatmaps at different iterations
    snapshots = sorted(simlr.history_['S_snapshots'].keys())
    selected_iters = [snapshots[0], snapshots[len(snapshots)//2], snapshots[-1]]
    
    for idx, iter_num in enumerate(selected_iters):
        ax = plt.subplot(3, 3, 7 + idx)
        S_snap = simlr.history_['S_snapshots'][iter_num]
        
        # Subsample for visualization if too large
        if S_snap.shape[0] > 500:
            sample_idx = np.random.choice(S_snap.shape[0], 500, replace=False)
            sample_idx = np.sort(sample_idx)
            S_snap = S_snap[np.ix_(sample_idx, sample_idx)]
        
        im = ax.imshow(S_snap, cmap='viridis', aspect='auto', vmin=0, vmax=np.percentile(S_snap, 95))
        ax.set_title(f'Similarity Matrix (Iter {iter_num})', fontsize=11, fontweight='bold')
        ax.set_xlabel('Cell Index', fontsize=10)
        ax.set_ylabel('Cell Index', fontsize=10)
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    
    plt.tight_layout()
    output_file = os.path.join(output_dir, f'{dataset_name}_process.png')
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"  Saved: {output_file}")
    plt.close()


def create_results_visualizations(X, simlr, y_true, y_pred, embedding, dataset_name, output_dir):
    """
    Figure Set 2: Results and Performance
    Shows WHAT SIMLR achieved (for Results section)
    """
    print(f"Creating results visualizations for {dataset_name}...")
    
    # Calculate metrics
    ari = adjusted_rand_score(y_true, y_pred)
    nmi = normalized_mutual_info_score(y_true, y_pred)
    silhouette = silhouette_score(embedding, y_pred)
    
    fig = plt.figure(figsize=(20, 12))
    
    # 1. t-SNE: True Labels
    ax1 = plt.subplot(3, 3, 1)
    scatter1 = ax1.scatter(embedding[:, 0], embedding[:, 1], c=y_true, 
                          cmap='tab20', s=15, alpha=0.6, edgecolors='none')
    ax1.set_title(f'True Cell Types (n={len(np.unique(y_true))})', 
                 fontsize=12, fontweight='bold')
    ax1.set_xlabel('t-SNE 1', fontsize=11)
    ax1.set_ylabel('t-SNE 2', fontsize=11)
    plt.colorbar(scatter1, ax=ax1, label='True Cluster')
    
    # 2. t-SNE: SIMLR Predictions
    ax2 = plt.subplot(3, 3, 2)
    scatter2 = ax2.scatter(embedding[:, 0], embedding[:, 1], c=y_pred, 
                          cmap='tab20', s=15, alpha=0.6, edgecolors='none')
    ax2.set_title(f'SIMLR Predictions (ARI={ari:.3f})', 
                 fontsize=12, fontweight='bold')
    ax2.set_xlabel('t-SNE 1', fontsize=11)
    ax2.set_ylabel('t-SNE 2', fontsize=11)
    plt.colorbar(scatter2, ax=ax2, label='Predicted Cluster')
    
    # 3. PCA of Latent Space
    ax3 = plt.subplot(3, 3, 3)
    if simlr.L_.shape[1] >= 2:
        ax3.scatter(simlr.L_[:, 0], simlr.L_[:, 1], c=y_pred, 
                   cmap='tab20', s=15, alpha=0.6, edgecolors='none')
        ax3.set_title('Latent Space (First 2 Components)', fontsize=12, fontweight='bold')
        ax3.set_xlabel('L1', fontsize=11)
        ax3.set_ylabel('L2', fontsize=11)
    
    # 4. Confusion Matrix
    ax4 = plt.subplot(3, 3, 4)
    cm = confusion_matrix(y_true, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax4, 
               cbar_kws={'label': 'Count'}, square=True)
    ax4.set_title('Confusion Matrix', fontsize=12, fontweight='bold')
    ax4.set_xlabel('Predicted Cluster', fontsize=11)
    ax4.set_ylabel('True Cluster', fontsize=11)
    
    # 5. Performance Metrics
    ax5 = plt.subplot(3, 3, 5)
    metrics_names = ['ARI', 'NMI', 'Silhouette']
    metrics_vals = [ari, nmi, silhouette]
    colors = ['#2ecc71' if v > 0.5 else '#e74c3c' if v > 0.3 else '#95a5a6' 
             for v in metrics_vals]
    bars = ax5.bar(metrics_names, metrics_vals, color=colors, alpha=0.7, edgecolor='black')
    ax5.set_ylim([0, 1])
    ax5.set_ylabel('Score', fontsize=11)
    ax5.set_title('Clustering Performance Metrics', fontsize=12, fontweight='bold')
    ax5.axhline(y=0.5, color='gray', linestyle='--', linewidth=1, alpha=0.5)
    ax5.grid(True, alpha=0.3, axis='y')
    
    for bar, val in zip(bars, metrics_vals):
        height = bar.get_height()
        ax5.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                f'{val:.3f}', ha='center', va='bottom', fontweight='bold', fontsize=10)
    
    # 6. Cluster Size Distribution
    ax6 = plt.subplot(3, 3, 6)
    true_counts = pd.Series(y_true).value_counts().sort_index()
    pred_counts = pd.Series(y_pred).value_counts().sort_index()
    
    x = np.arange(max(len(true_counts), len(pred_counts)))
    width = 0.35
    
    # Pad with zeros if lengths don't match
    true_vals = np.zeros(len(x))
    pred_vals = np.zeros(len(x))
    true_vals[:len(true_counts)] = true_counts.values
    pred_vals[:len(pred_counts)] = pred_counts.values
    
    ax6.bar(x - width/2, true_vals, width, label='True', alpha=0.7, color='#3498db')
    ax6.bar(x + width/2, pred_vals, width, label='Predicted', alpha=0.7, color='#e67e22')
    ax6.set_xlabel('Cluster ID', fontsize=11)
    ax6.set_ylabel('Number of Cells', fontsize=11)
    ax6.set_title('Cluster Size Distribution', fontsize=12, fontweight='bold')
    ax6.legend()
    ax6.set_xticks(x)
    ax6.grid(True, alpha=0.3, axis='y')
    
    # 7. Silhouette Score per Cluster
    ax7 = plt.subplot(3, 3, 7)
    from sklearn.metrics import silhouette_samples
    silhouette_vals = silhouette_samples(embedding, y_pred)
    
    cluster_silhouettes = []
    cluster_ids = []
    for cluster in sorted(np.unique(y_pred)):
        cluster_sil = silhouette_vals[y_pred == cluster]
        cluster_silhouettes.append(np.mean(cluster_sil))
        cluster_ids.append(cluster)
    
    colors_sil = ['#2ecc71' if s > 0 else '#e74c3c' for s in cluster_silhouettes]
    ax7.barh(cluster_ids, cluster_silhouettes, color=colors_sil, alpha=0.7, edgecolor='black')
    ax7.axvline(x=0, color='black', linewidth=1)
    ax7.axvline(x=silhouette, color='red', linestyle='--', linewidth=2, 
               label=f'Mean={silhouette:.3f}')
    ax7.set_xlabel('Silhouette Score', fontsize=11)
    ax7.set_ylabel('Cluster ID', fontsize=11)
    ax7.set_title('Cluster Quality (Silhouette)', fontsize=12, fontweight='bold')
    ax7.legend()
    ax7.invert_yaxis()
    
    # 8. Per-cluster Accuracy (Purity)
    ax8 = plt.subplot(3, 3, 8)
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
    
    colors_pur = plt.cm.RdYlGn(cluster_purity)
    ax8.bar(sorted(np.unique(y_pred)), cluster_purity, color=colors_pur, 
           alpha=0.7, edgecolor='black')
    ax8.set_xlabel('Cluster ID', fontsize=11)
    ax8.set_ylabel('Purity', fontsize=11)
    ax8.set_ylim([0, 1])
    ax8.set_title('Cluster Purity (Homogeneity)', fontsize=12, fontweight='bold')
    ax8.axhline(y=np.mean(cluster_purity), color='red', linestyle='--', 
               linewidth=2, label=f'Mean={np.mean(cluster_purity):.3f}')
    ax8.legend()
    ax8.grid(True, alpha=0.3, axis='y')
    
    # 9. Summary Statistics
    ax9 = plt.subplot(3, 3, 9)
    ax9.axis('off')
    
    summary_text = f"""
    SIMLR Results Summary
    {'='*35}
    
    Dataset: {dataset_name}
    Cells: {X.shape[0]:,}
    Genes: {X.shape[1]:,}
    True Clusters: {len(np.unique(y_true))}
    Predicted Clusters: {len(np.unique(y_pred))}
    
    Performance Metrics:
    {'─'*35}
    ARI:                {ari:.4f}
    NMI:                {nmi:.4f}
    Silhouette:         {silhouette:.4f}
    Mean Purity:        {np.mean(cluster_purity):.4f}
    
    Algorithm Parameters:
    {'─'*35}
    Iterations:         {simlr.n_iter}
    Beta (β):           {simlr.beta}
    Gamma (γ):          {simlr.gamma}
    Rho (ρ):            {simlr.rho}
    """
    
    ax9.text(0.1, 0.9, summary_text, transform=ax9.transAxes,
            fontsize=10, verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
    
    plt.tight_layout()
    output_file = os.path.join(output_dir, f'{dataset_name}_results.png')
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"  Saved: {output_file}")
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
    
    # Run SIMLR
    print("Running SIMLR...")
    simlr = SIMLR(n_clusters=n_clusters, n_iterations=30, verbose=True)
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
    print("\nCreating comparison figure...")
    
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
    print(f"Saved: {output_file}")
    plt.close()


if __name__ == "__main__":
    # Setup
    output_dir = 'model_learning/figures'
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
    
    print("\n" + "="*60)
    print("ALL FIGURES GENERATED SUCCESSFULLY!")
    print("="*60)
    print(f"\nFigures saved in: {output_dir}/")
    print("\nGenerated files:")
    for dataset in ['pbmc3k', 'paul15']:
        print(f"  - {dataset}_process.png (SIMLR algorithm process)")
        print(f"  - {dataset}_results.png (clustering results)")
    print(f"  - overall_comparison.png (dataset comparison)")
