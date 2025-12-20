import os, argparse, yaml, numpy as np, scanpy as sc, torch
from torch.utils.data import DataLoader, TensorDataset
from sklearn.cluster import KMeans
from desc_model import AutoEncoder, student_t_soft_assign, target_distribution, kl_divergence, batch_center_loss
from utils import compute_metrics, save_json
from plots import plot_losses, compute_umap, scatter2d

def set_seeds(seed=42):
    import random, numpy as np, torch
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)

def prepare_data(cfg):
    adata = sc.read_h5ad(cfg['dataset_path'])

    # ---- Force dense float32 matrix ----
    if hasattr(adata.X, "toarray"):
        X = adata.X.toarray()
    else:
        X = np.array(adata.X)
    X = X.astype(np.float32)
    X[np.isnan(X)] = 0
    adata.X = X

    # ---- Annotate mitochondrial genes ----
    if 'mt' not in adata.var.columns:
        adata.var['mt'] = adata.var_names.str.startswith('MT-') | adata.var_names.str.startswith('mt-')

    # ---- QC metrics ----
    sc.pp.calculate_qc_metrics(adata, qc_vars=['mt'], inplace=True)

    # ---- QC filtering ----
    adata = adata[adata.obs['n_genes_by_counts'] >= cfg['min_genes']].copy()
    adata = adata[adata.obs['pct_counts_mt'] <= (cfg['max_mito_pct']*100)].copy()

    # ---- Re-extract X and force numeric again ----
    if hasattr(adata.X, "toarray"):
        X = adata.X.toarray()
    else:
        X = np.array(adata.X)
    X = X.astype(np.float32)
    X[np.isnan(X)] = 0

    # ---- MANUAL NORMALIZATION (bypass Scanpy) ----
    # Total-count normalize to target_sum
    counts = X.sum(axis=1, keepdims=True)
    counts[counts == 0] = 1.0  # avoid divide-by-zero
    X = X / counts * float(cfg['target_sum'])

    # ---- Log1p ----
    X = np.log1p(X)

    # ---- Save back ----
    adata.X = X

    # ---- HVGs (we can keep Scanpy here; it only uses numpy ops) ----
    sc.pp.highly_variable_genes(adata, n_top_genes=cfg['n_hvgs'], subset=True, flavor="seurat_v3")

    # ---- Extract final numeric matrix ----
    if hasattr(adata.X, "toarray"):
        X = adata.X.toarray()
    else:
        X = np.array(adata.X)
    X = X.astype(np.float32)
    X[np.isnan(X)] = 0

    # ---- Z-score per gene ----
    X = (X - X.mean(axis=0, keepdims=True)) / (X.std(axis=0, keepdims=True) + 1e-6)

    # ---- Metadata ----
    batch = adata.obs["batch"].astype("category").cat.codes.values.astype(np.int64) if "batch" in adata.obs.columns else None
    cell_type = adata.obs["cell_type"].astype("category").cat.codes.values.astype(np.int64) if "cell_type" in adata.obs.columns else None

    return adata, X, batch, cell_type

def infer_k(emb, k_min=5, k_max=15):
    scores = []
    for k in range(k_min, k_max+1):
        km = KMeans(n_clusters=k, n_init=10, random_state=42).fit(emb)
        scores.append(km.inertia_)
    diffs = np.diff(scores)
    return int(np.argmin(diffs) + k_min)

def train(cfg):
    set_seeds(cfg['random_seed'])
    # io
    ds_name = os.path.splitext(os.path.basename(cfg['dataset_path']))[0]
    if not cfg['output_dir'].endswith(ds_name):
        cfg['output_dir'] = os.path.join(os.path.dirname(cfg['output_dir']), f"{ds_name}_desc")
    outdir = cfg['output_dir']; os.makedirs(outdir, exist_ok=True)

    adata, X, batch, cell_type = prepare_data(cfg)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # model + optim
    model = AutoEncoder(X.shape[1], tuple(cfg['hidden_dims']), cfg['latent_dim'], cfg['dropout']).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=cfg['lr'])
    dl = DataLoader(TensorDataset(torch.from_numpy(X)), batch_size=cfg['batch_size'], shuffle=True)

    # warm-up (recon only)
    loss_recon_hist = []
    for epoch in range(cfg['epochs_warmup']):
        model.train(); epoch_loss=0.0
        for (xb,) in dl:
            xb = xb.to(device)
            z, xhat = model(xb)
            loss = ((xhat - xb)**2).mean()
            opt.zero_grad(); loss.backward(); opt.step()
            epoch_loss += loss.item() * xb.size(0)
        loss_recon_hist.append(epoch_loss / X.shape[0])

    # get embeddings
    model.eval(); Z=[]
    with torch.no_grad():
        for i in range(0, X.shape[0], 4096):
            xb = torch.from_numpy(X[i:i+4096]).to(device)
            z = model.encode(xb); Z.append(z.cpu().numpy())
    Z = np.vstack(Z)

    # init centers
    K = cfg['n_clusters'] if cfg['n_clusters']>0 else infer_k(Z, 5, 15)
    kmeans = KMeans(n_clusters=K, n_init=20, random_state=cfg['random_seed']).fit(Z)
    centers = torch.from_numpy(kmeans.cluster_centers_.astype(np.float32)).to(device)

    # refinement (recon + KL + batch)
    total_hist, kl_hist, batch_hist = [], [], []
    X_tensor = torch.from_numpy(X).to(device)
    batch_tensor = torch.from_numpy(batch).to(device) if batch is not None else None
    P = None
    for epoch in range(cfg['epochs_refine']):
        model.train()
        z, xhat = model(X_tensor)
        Q = student_t_soft_assign(z, centers, alpha=1.0)
        if epoch % cfg['refresh_target_every'] == 0 or P is None:
            P = target_distribution(Q).detach()
        loss_recon = ((xhat - X_tensor)**2).mean()
        loss_kl = kl_divergence(P, Q)
        loss_batch = batch_center_loss(z, batch_tensor)
        loss = loss_recon + cfg['lambda_kl']*loss_kl + cfg['beta_batch']*loss_batch
        opt.zero_grad(); loss.backward(); opt.step()
        # EMA centers
        with torch.no_grad():
            assign = Q / (Q.sum(dim=0, keepdim=True) + 1e-8)
            new_centers = assign.t() @ z
            centers = 0.9*centers + 0.1*new_centers
        total_hist.append(float(loss.item())); kl_hist.append(float(loss_kl.item()))
        batch_hist.append(float(loss_batch.item() if batch is not None else 0.0))

    # final outputs
    model.eval()
    with torch.no_grad():
        z, _ = model(X_tensor)
        Q = student_t_soft_assign(z, centers, alpha=1.0)
        labels = Q.argmax(dim=1).cpu().numpy()
        Z = z.cpu().numpy()

    coords = compute_umap(Z, cfg['umap_n_neighbors'], cfg['umap_min_dist'], cfg['random_seed'])
    labels_true = adata.obs['cell_type'].values if 'cell_type' in adata.obs.columns else None
    metrics = compute_metrics(coords, labels, labels_true)

    # save
    os.makedirs(os.path.join(outdir, "figs"), exist_ok=True)
    np.save(os.path.join(outdir, "latent.npy"), Z)
    np.save(os.path.join(outdir, "clusters.npy"), labels)
    np.save(os.path.join(outdir, "umap.npy"), coords)
    save_json(metrics, os.path.join(outdir, "metrics.json"))
    plot_losses({"recon_warmup":loss_recon_hist, "total_refine":total_hist, "kl_refine":kl_hist, "batch_refine":batch_hist},
                os.path.join(outdir, "figs", "loss_curves.png"))
    scatter2d(coords, labels, "UMAP by DESC cluster", os.path.join(outdir, "figs", "umap_by_cluster.png"))
    if 'batch' in adata.obs.columns:
        bcodes = adata.obs['batch'].astype('category').cat.codes.values
        scatter2d(coords, bcodes, "UMAP by batch", os.path.join(outdir, "figs", "umap_by_batch.png"))
    if 'cell_type' in adata.obs.columns:
        ct = adata.obs['cell_type'].astype('category').cat.codes.values
        scatter2d(coords, ct, "UMAP by cell type", os.path.join(outdir, "figs", "umap_by_celltype.png"))

    meta = dict(cfg); meta['n_clusters_used'] = int(K); save_json(meta, os.path.join(outdir, "run_config.json"))
    print("Done. Outputs:", outdir)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, required=True)
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.config))
    train(cfg)
