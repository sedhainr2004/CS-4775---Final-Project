import numpy as np
from scipy.spatial.distance import pdist, squareform, cdist
from scipy.sparse.linalg import eigsh
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
import warnings

class SIMLR:
    """
    Single-cell Interpretation via Multi-kernel Learning (SIMLR).
    
    Reimplementation based on Wang et al. (2017) Nature Methods.
    """
    def __init__(self, n_clusters, beta=0.8, gamma=1.0, rho=1.0, n_iterations=30, verbose=True):
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
        self.verbose = verbose
        
        # Model artifacts
        self.S_ = None      # Learned Similarity Matrix
        self.L_ = None      # Latent Matrix
        self.w_ = None      # Kernel Weights
        self.kernels_ = []  # Computed Kernels
        self.labels_ = None # Cluster labels

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
        
        # Heuristic parameters from the Methods section
        # k in {10, 12, ..., 30}
        k_range = range(10, 32, 2) 
        # sigma in {1.0, 1.25, ..., 2.0}
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
        rho = ind[cond][-1]
        theta = (cssv[rho - 1] - 1) / rho
        w = np.maximum(v - theta, 0)
        return w

    def fit(self, X):
        """
        Main optimization loop.
        """
        N = X.shape[0]
        
        if self.verbose:
            print(f"Computing kernels for {N} cells...")
        self.kernels_ = self._cal_kernels(X)
        num_kernels = self.kernels_.shape[0]
        
        # 0. Initialization
        # w: uniform weights
        self.w_ = np.ones(num_kernels) / num_kernels
        # S: average of kernels
        self.S_ = np.mean(self.kernels_, axis=0)
        # L: Top C eigenvectors of S
        # We use S directly because maximizing tr(L' S L) is equivalent to minimizing tr(L'(I-S)L)
        vals, vecs = eigsh(self.S_, k=self.C, which='LA')
        self.L_ = vecs
        
        if self.verbose:
            print("Starting optimization loop...")
            
        for itr in range(self.n_iter):
            # --- Step 1: Update S (Similarity Matrix) ---
            # Problem: min_S \sum -w_l K_l . S + beta ||S||^2 + gamma tr(L' (I-S) L)
            # Rearranging w.r.t S_ij:
            # min \sum (beta * S_ij^2 - ( \sum w_l K_l(i,j) + gamma * (L_i . L_j) ) * S_ij )
            # This is a projection of a vector onto the simplex.
            # The 'vector' to project is: ( \sum w_l K_l + gamma * L L^T ) / (2 * beta)
            
            # Weighted kernel sum
            K_sum = np.tensordot(self.w_, self.kernels_, axes=(0, 0)) # N x N
            
            # L term
            L_dist = np.dot(self.L_, self.L_.T) # N x N
            
            # Target for projection
            # Factor 2*beta comes from derivative of beta*S^2 -> 2*beta*S
            target = (K_sum + self.gamma * L_dist) / (2 * self.beta)
            
            # Apply constraints: S_ij >= 0, sum_j S_ij = 1
            # We treat diagonal as 0 or ignore it in the optimization usually, 
            # but formally SIMLR optimizes S_ij.
            # Row-wise simplex projection
            for i in range(N):
                self.S_[i, :] = self._project_simplex(target[i, :])
            
            # Symmetrize S (heuristic to maintain stability, though optimization should favor symmetry)
            self.S_ = (self.S_ + self.S_.T) / 2
            
            # --- Step 2: Update L (Latent Matrix) ---
            # Maximize tr(L^T (S) L) -> Top C eigenvectors of S
            # We solve standard eigenvalue problem
            eig_vals, eig_vecs = eigsh(self.S_, k=self.C, which='LA')
            self.L_ = eig_vecs
            
            # --- Step 3: Update w (Kernel Weights) ---
            # Problem: min_w - \sum w_l <K_l, S> + rho \sum w_l log w_l
            # Solution: w_l \propto exp( <K_l, S> / rho )
            
            # Compute inner products <K_l, S>
            # Frobenius inner product = sum(K_l * S) elementwise
            kernel_trace = np.zeros(num_kernels)
            for l in range(num_kernels):
                kernel_trace[l] = np.sum(self.kernels_[l] * self.S_)
                
            # Closed form update
            w_num = np.exp(kernel_trace / self.rho)
            self.w_ = w_num / np.sum(w_num)
            
            if self.verbose and itr % 5 == 0:
                print(f"Iteration {itr}/{self.n_iter} complete.")
                
        # --- Step 4: Diffusion Enhancement (Optional but recommended in paper) ---
        # "We apply a diffusion-based step to enhance the similarity matrix S"
        # Eq (4) and (5) in Methods
        self.S_ = self._diffusion_enhancement(self.S_)
        
        return self

    def _diffusion_enhancement(self, S, alpha=0.8, K=20):
        """
        Enhances similarity matrix using diffusion process.
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
        
        # Diffusion iteration: H(t+1) = S * P^T + ... (Paper formula varies slightly by implementation)
        # Using specific formula from SIMLR code/paper Eq 5:
        # H_{t+1} = alpha * H_t * P + (1 - alpha) * I  (Typical PageRank style)
        # But paper says: H(t+1) = S * P * S^T ... 
        # Here we use the simplified accumulation often found in the actual SIMLR code:
        # H = S; iterate H = alpha * H * P + (1-alpha) * S
        
        H = S.copy()
        # Small number of diffusion steps
        for _ in range(5):
            H = alpha * np.dot(H, P) + (1 - alpha) * S
            
        return H

    def fit_transform(self, X, embedding_dim=2):
        """
        Runs fit and then performs dimensionality reduction.
        Returns:
            embedding: N x 2 (or 3) matrix for visualization.
        """
        self.fit(X)
        
        # SIMLR uses a modified t-SNE that takes the learned Similarity S directly.
        # We can approximate this by using sklearn TSNE with metric='precomputed'.
        # However, TSNE expects distance, not similarity.
        # We convert Similarity to Distance: D = 1 - S (or similar transform)
        # Note: Standard SIMLR code modifies the t-SNE objective to use P=S directly.
        # Here we use the 'precomputed' distance trick.
        
        # Normalize S to be stochastic (like P matrix in t-SNE)
        S_norm = self.S_ / np.sum(self.S_)
        
        # Convert to a distance-like matrix for sklearn input
        # Standard t-SNE computes P from distances. We already have P (S_norm).
        # To strictly use sklearn, we need to bypass the P computation, which is hard.
        # ALTERNATIVE: Use the Latent Matrix L for projection?
        # The paper says: "For visualization... project the data... 
        # to which we apply k-means... resulting in an N x B latent matrix Z"
        
        # Visualization typically uses the S matrix into t-SNE. 
        # For this implementation, we will project the Latent Matrix L using t-SNE
        # which captures the block structure well.
        
        tsne = TSNE(n_components=embedding_dim, metric='cosine', init='random')
        embedding = tsne.fit_transform(self.L_)
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