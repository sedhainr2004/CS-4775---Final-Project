import numpy as np
from scipy.spatial.distance import pdist, squareform, cdist
from scipy.sparse.linalg import eigsh
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
import warnings

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
        
        self.S_ = None      # Learned Similarity Matrix
        self.L_ = None      # Latent Matrix
        self.w_ = None      # Kernel Weights
        self.kernels = []  # Computed Kernels
        self.labels_ = None # Cluster labels

        self.history_ = {
            'w_history': [],      # Kernel weights per iteration
            'S_changes': [],      # Frobenius norm of S change per iteration
            'objective': [],      # Objective function value per iteration
            'S_snapshots': {}     # Similarity matrix at key iterations
        }

    def _cal_kernels(self, X):
        """
        Constructs multiple Gaussian kernels based on Euclidean distance.
        Parameters: Sigma (1.0 to 2.0) and K neighbors (10 to 30).
        """
        N = X.shape[0]
        # Calculate full pairwise euclidean distances
        # Note: For very large N, approximate NN should be used (e.g., Annoy) as per paper.
        # Here we implement the exact version for standard usage.
        dist_matrix = squareform(pdist(X, metric='euclidean'))
        dist_sq = dist_matrix ** 2
        
        kernels = []
        
        k_range = range(10, 32, 2) 
        sigma_range = [1.0, 1.25, 1.5, 1.75, 2.0]
        
        for k in k_range:
            # Determine epsilon_ij scale based on k-th neighbor
            # Sort distances to find k-th neighbor distance
            sorted_dist = np.sort(dist_matrix, axis=1)
            # Avoid self-distance (index 0)
            k_dist = sorted_dist[:, k]
            
            # Average scale approximation for variance
            # sigma_ij = (scale_i + scale_j) / 2
            # Here we simplify to a global scale per k for vectorization or 
            # use the broadcasting approach. The paper implies specific mu_i.
            mu = k_dist.reshape(-1, 1) # N x 1
            mu_ij = (mu + mu.T) / 2
            
            for sigma in sigma_range:
                # Kernel entry: exp( - ||x_i - x_j||^2 / (2 * (sigma * mu_ij)^2) )
                # Note: Paper Eq(3) uses simple variance term epsilon_ij
                denom = (sigma * mu_ij) ** 2
                # Avoid division by zero
                denom[denom == 0] = 1e-10
                
                K = np.exp(-dist_sq / (2 * denom))
                
                # Zero out diagonal
                np.fill_diagonal(K, 0)
                kernels.append(K)
                
        return np.array(kernels)

    def _project_simplex(self, v):
        """
        Solves the projection problem: min ||x - v||^2 s.t. sum(x)=1, x>=0.
        Used to update rows of S.
        """
        n_features = v.shape[0]
        u = np.sort(v)[::-1]
        cssv = np.cumsum(u)
        ind = np.arange(n_features) + 1
        cond = u - (cssv - 1) / ind > 0
        
        # Handle edge case where no elements satisfy condition
        if not np.any(cond):
            # Return uniform distribution
            return np.ones(n_features) / n_features
            
        rho = ind[cond][-1]
        theta = (cssv[rho - 1] - 1) / rho
        w = np.maximum(v - theta, 0)
        return w

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
            target = (K_sum + self.gamma * L_dist) / (2 * self.beta)
            
            for i in range(N):
                self.S_[i, :] = self._project_simplex(target[i, :])
            
            self.S_ = (self.S_ + self.S_.T) / 2
            
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
        Paper formula (Step 4): H_{t+1} = (1 - alpha) * P^T * H_t * P + alpha * S
        """
        N = S.shape[0]
        # Construct transition matrix P
        # Only keep top K neighbors for sparsity/noise reduction
        P = np.zeros_like(S)
        for i in range(N):
            # Indices of top K neighbors
            top_k_idx = np.argsort(S[i, :])[-K:]
            P[i, top_k_idx] = S[i, top_k_idx]
        
        # Normalize rows to sum to 1
        row_sums = P.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1 # avoid div zero
        P = P / row_sums
        
        # Apply paper's diffusion formula: H_{t+1} = (1 - alpha) * P^T * H_t * P + alpha * S
        H = S.copy()
        # Small number of diffusion steps
        for _ in range(5):
            H = (1 - alpha) * np.dot(P.T, np.dot(H, P)) + alpha * S
            
        return H

    def fit_transform(self, X, embedding_dim=2):
        """
        Runs fit and then performs dimensionality reduction.
        Paper uses modified t-SNE with similarity S directly as P_ij.
        We approximate by converting S to distance and using precomputed metric.
        Returns:
            embedding: N x 2 (or 3) matrix for visualization.
        """
        self.fit(X)
        
        # Use learned similarity S directly in t-SNE
        # Standard t-SNE computes P_ij from Gaussian kernel on distances
        # SIMLR's modification: Use S as P_ij directly
        
        S_sym = (self.S_ + self.S_.T) / 2
        # Convert similarity to distance
        distances = np.sqrt(np.maximum(0, 2 * (1 - S_sym)))
        
        # Apply t-SNE with precomputed distances
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
            
        # "For clustering with k-means, we use... N x B latent matrix Z (L in code)"
        kmeans = KMeans(n_clusters=self.C, n_init=10, random_state=42)
        self.labels_ = kmeans.fit_predict(self.L_)
        return self.labels_

# --- Example Usage ---
if __name__ == "__main__":
    # Create synthetic single-cell data (e.g., 3 blobs)
    from sklearn.datasets import make_blobs
    X_syn, y_syn = make_blobs(n_samples=200, n_features=50, centers=3, random_state=42)
    
    # Initialize SIMLR
    # C=3 clusters
    simlr = SIMLR(n_clusters=3, verbose=True)
    
    # Fit and get visualization embedding
    embedding = simlr.fit_transform(X_syn)
    
    # Get clusters
    clusters = simlr.predict()
    
    print("Computed Clusters:", clusters[:10], "...")
    print("Embedding Shape:", embedding.shape)