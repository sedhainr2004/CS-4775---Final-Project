import numpy as np
from scipy.spatial.distance import pdist, squareform
from scipy.sparse.linalg import eigsh
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE

class SIMLR:
    """
    Reimplementation based on Wang et al. (2017)
    """
    def __init__(self, n_clusters, beta=0.8, gamma=1.0, rho=1.0, n_iterations=30, logging=True):
        """
        Args:
            n_clusters (C): Number of desired clusters (rank constraint).
            beta: Regularization parameter for ||S||^2 (frob norm).
            gamma: Regularization parameter for trace(L' (I-S) L).
            rho: Regularization parameter for w log w (kernel weights entropy).
            n_iterations: Number of optimization iterations.
        """
        self.C = n_clusters
        self.beta = beta
        self.gamma = gamma
        self.rho = rho
        self.n_iter = n_iterations
        self.logging = logging
        
        self.S_ = None 
        self.L_ = None 
        self.w_ = None   
        self.kernels = []  
        self.labels_ = None # Cluster labels

        self.history_ = {
            'w_history': [],      # Kernel weights per iteration
            'S_changes': [],      # Frobenius norm of S change per iteration
            'objective': [],      # Objective function value per iteration
            'S_snapshots': {}     # Similarity matrix at key iterations
        }

    def _cal_kernels(self, X):
        """
        Constructs multiple Gaussian kernels based on Euclidean distances
        """
        # For very large N, approximate NN should be used
        dist_matrix = squareform(pdist(X, metric='euclidean'))
        dist_sq = dist_matrix ** 2
        
        kernels = []
        k_range = range(10, 32, 2) 
        sigma_range = [1.0, 1.25, 1.5, 1.75, 2.0]

        # Sort distances to find knn distance
        sorted_dist = np.sort(dist_matrix, axis=1)

        for k in k_range:
            mu = np.mean(sorted_dist[:, 1:k+1], axis=1, keepdims=True)
            mu_ij = (mu + mu.T) / 2
            
            for sigma in sigma_range:
                epsilon_ij = sigma * mu_ij
                epsilon_ij = np.maximum(epsilon_ij, 1e-10)  
                
                coeff = 1.0 / (epsilon_ij * np.sqrt(2 * np.pi))
                K = coeff * np.exp(-dist_sq / (2 * epsilon_ij ** 2))

                np.fill_diagonal(K, 0)
                kernels.append(K)
                
        return np.array(kernels)

    def _project_simplex(self, v_i):
        """
        Simplex projection using centering transformation + Newton's method.
        """
        N = len(v_i)

        u_i = v_i - np.mean(v_i) + (1.0 / N)
        
        sigma = 0.0
        for _ in range(50):
            diff = sigma - u_i
            positive_parts = np.maximum(diff, 0)
            
            f_val = np.sum(positive_parts) / (N - 1) - sigma
            f_prime = np.sum(diff > 0) / (N - 1) - 1.0
            
            if abs(f_prime) < 1e-12:
                break
            
            sigma_new = sigma - f_val / f_prime
            if abs(sigma_new - sigma) < 1e-10:
                break
            sigma = sigma_new
        
        return np.maximum(u_i - sigma, 0)


    def _update_S(self, K_sum, L_dist, N):
        """
        Full S update step.
        """
        V = (K_sum + self.gamma * L_dist) / (2 * self.beta)
        
        S_new = np.zeros((N, N))
        for i in range(N):
            S_new[i, :] = self._project_simplex(V[i, :])
        
        return (S_new + S_new.T) / 2
    

    def fit(self, X):
        """
        Main optimization loop
        """
        N = X.shape[0]
        
        self.kernels = self._cal_kernels(X)
        num_kernels = self.kernels.shape[0]
        
        # Step 0: Initialization
        # w: uniform weights
        self.w_ = np.ones(num_kernels) / num_kernels
        # S: average of kernels
        self.S_ = np.mean(self.kernels, axis=0)
        # L: Top C eigenvectors of (I_N - S)
        I_N = np.eye(N)
        L_matrix = I_N - self.S_

        vals, vecs = eigsh(L_matrix, k=self.C, which='SM')
        self.L_ = vecs
        
        # initial state
        S_prev = self.S_.copy()
        self.history_['w_history'].append(self.w_.copy())
        self.history_['S_snapshots'][0] = self.S_.copy()
            
        for itr in range(self.n_iter):
            # Step 1: Update S
            K_sum = np.tensordot(self.w_, self.kernels, axes=(0, 0)) # N x N
            L_dist = np.dot(self.L_, self.L_.T) 
            self.S_ = self._update_S(K_sum, L_dist, N)
            
            # Step 2: Update L
            I_N = np.eye(N)
            L_matrix = I_N - self.S_
            eig_vals, eig_vecs = eigsh(L_matrix, k=self.C, which='SM')
            self.L_ = eig_vecs
            
            # Step 3: Update w
            kernel_trace = np.zeros(num_kernels)
            for l in range(num_kernels):
                kernel_trace[l] = np.sum(self.kernels[l] * self.S_)
                
            # Closed form update
            kernel_trace_normalized = kernel_trace - np.max(kernel_trace)
            w_num = np.exp(kernel_trace_normalized / self.rho)
            self.w_ = w_num / np.sum(w_num)
            
            # Track metrics
            S_change = np.linalg.norm(self.S_ - S_prev, 'fro')
            self.history_['S_changes'].append(S_change)
            self.history_['w_history'].append(self.w_.copy())
            
            # Calculate objective function value
            obj_val = (self.beta * np.sum(self.S_ ** 2) - 
                      np.sum(self.w_ * kernel_trace) + 
                      self.rho * np.sum(self.w_ * np.log(self.w_ + 1e-10)))
            self.history_['objective'].append(obj_val)
            
            # Store S
            if itr in [5, 10, 15, 20, 25, 29]:
                self.history_['S_snapshots'][itr+1] = self.S_.copy()
            
            S_prev = self.S_.copy()
            
            if self.logging and itr % 5 == 0:
                print(f"Iteration {itr}/{self.n_iter} complete.")
                
        # Step 4: Diffusion Enhancement 
        self.S_ = self._diffusion_enhancement(self.S_)
        
        return self

    def _diffusion_enhancement(self, S, alpha=0.8, K=20):
        """
        Enhances similarity matrix using diffusion process.
        H_{t+1} = (1 - alpha) * P^T * H_t * P + alpha * S
        """
        N = S.shape[0]

        P = np.zeros_like(S)
        for i in range(N):
            top_k_idx = np.argsort(S[i, :])[-K:]
            P[i, top_k_idx] = S[i, top_k_idx]
        
        row_sums = P.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1
        P = P / row_sums
        
        H = S.copy()
        for _ in range(5):
            H = (1 - alpha) * np.dot(P.T, np.dot(H, P)) + alpha * S
            
        return H

    def fit_transform(self, X, embedding_dim=2):
        """
        Runs fit and then performs dimensionality reduction with t-SNE with similarity S directly as P_ij.
        Returns:
            embedding: N x 2 (or 3) matrix for visualization.
        """
        self.fit(X)
        
        S_sym = (self.S_ + self.S_.T) / 2
        distances = np.sqrt(np.maximum(0, 2 * (1 - S_sym)))
        
        tsne = TSNE(n_components=embedding_dim, metric='precomputed', init='random', 
                   perplexity=min(30, (len(S_sym) - 1) // 3))
        embedding = tsne.fit_transform(distances)
        
        return embedding

    def predict(self, X=None):
        """
        Performs clustering on the learned latent space L.
        """
        if self.L_ is None:
            raise ValueError("Model not fitted. Run fit() first.")
            
        kmeans = KMeans(n_clusters=self.C, n_init=10, random_state=42)
        self.labels_ = kmeans.fit_predict(self.L_)
        return self.labels_


# Example for synthetic dataset
if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from sklearn.datasets import make_blobs
    from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

    X_syn, y_syn = make_blobs(n_samples=200, n_features=50, centers=3,
                              cluster_std=3.0, center_box=(-2, 2), random_state=42)

    simlr = SIMLR(n_clusters=3, logging=True)

    embedding = simlr.fit_transform(X_syn)
    clusters = simlr.predict()

    ari = adjusted_rand_score(y_syn, clusters)
    nmi = normalized_mutual_info_score(y_syn, clusters)
    print(f"\nARI: {ari:.4f}, NMI: {nmi:.4f}")

    # Visualize embedding with predicted clusters vs actual labels
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Plot 1: Predicted clusters
    scatter1 = axes[0].scatter(embedding[:, 0], embedding[:, 1],
                              c=clusters, cmap='viridis', s=50, alpha=0.7, edgecolors='k', linewidth=0.5)
    axes[0].set_title(f'SIMLR Predicted Clusters\nARI: {ari:.4f}', fontsize=12, fontweight='bold')
    axes[0].set_xlabel('t-SNE Component 1', fontsize=10)
    axes[0].set_ylabel('t-SNE Component 2', fontsize=10)
    plt.colorbar(scatter1, ax=axes[0], label='Predicted Cluster')
    axes[0].grid(True, alpha=0.3)

    # Plot 2: Actual labels
    scatter2 = axes[1].scatter(embedding[:, 0], embedding[:, 1],
                              c=y_syn, cmap='viridis', s=50, alpha=0.7, edgecolors='k', linewidth=0.5)
    axes[1].set_title(f'Ground Truth Labels\nNMI: {nmi:.4f}', fontsize=12, fontweight='bold')
    axes[1].set_xlabel('t-SNE Component 1', fontsize=10)
    axes[1].set_ylabel('t-SNE Component 2', fontsize=10)
    plt.colorbar(scatter2, ax=axes[1], label='True Label')
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('simlr_embedding_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()