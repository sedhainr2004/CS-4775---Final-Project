import scanpy as sc
import numpy as np
import pandas as pd
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from scipy import sparse

# --- IMPORT YOUR SIMLR CLASS HERE ---
from SIMLR_implementation import SIMLR # (Or paste the class definition provided in your prompt here)

def run_simlr_experiment(dataset_path, dataset_name):
    print(f"\n{'='*10} Processing {dataset_name} {'='*10}")
    
    # 1. Load Data
    adata = sc.read_h5ad(dataset_path)
    print(f"Loaded {dataset_name}: {adata.n_obs} cells x {adata.n_vars} genes")
    
    # 2. Prepare Input (X)
    # The SIMLR implementation uses scipy.pdist which requires a dense array.
    # Scanpy stores X as sparse often, so we must densify it.
    if sparse.issparse(adata.X):
        X = adata.X.toarray()
    else:
        X = adata.X
        
    # 3. Prepare Labels (y) & Determine k
    # We use the 'cell_type' column you created earlier as ground truth
    y_true = adata.obs['cell_type'].values
    n_clusters = len(np.unique(y_true))
    print(f"Detected {n_clusters} clusters/cell types.")
    
    # 4. Initialize and Run SIMLR
    # Note: beta/gamma/rho are defaults, but you can tune them if results are poor.
    # n_iterations=30 is usually sufficient for convergence.
    simlr = SIMLR(n_clusters=n_clusters, n_iterations=30, verbose=True)
    
    print("Fitting SIMLR model...")
    simlr.fit(X)
    
    # 5. Predict Labels
    # Your predict method uses K-Means on the learned Latent Matrix (L_)
    y_pred = simlr.predict()
    
    # 6. Evaluate
    ari = adjusted_rand_score(y_true, y_pred)
    nmi = normalized_mutual_info_score(y_true, y_pred)
    
    print(f"\n--- {dataset_name} Results ---")
    print(f"ARI: {ari:.4f}")
    print(f"NMI: {nmi:.4f}")
    
    return ari, nmi

# ==========================================
# Main Execution
# ==========================================
if __name__ == "__main__":
    # Run on PBMC 3k
    run_simlr_experiment("data/pbmc3k.h5ad", "PBMC 3k")
    
    # Run on Paul15
    run_simlr_experiment("data/paul15.h5ad", "Paul15")