import os, numpy as np, matplotlib.pyplot as plt
import umap

def plot_losses(train_log, out_png):
    plt.figure(figsize=(6,4))
    for k,v in train_log.items(): plt.plot(v, label=k)
    plt.xlabel('Epoch'); plt.ylabel('Loss'); plt.legend(); plt.tight_layout()
    os.makedirs(os.path.dirname(out_png), exist_ok=True)
    plt.savefig(out_png, dpi=150); plt.close()

def compute_umap(emb, n_neighbors=15, min_dist=0.3, random_state=42):
    reducer = umap.UMAP(n_neighbors=n_neighbors, min_dist=min_dist, random_state=random_state)
    return reducer.fit_transform(emb)

def scatter2d(coords, color, title, out_png):
    plt.figure(figsize=(5,5))
    sc = plt.scatter(coords[:,0], coords[:,1], c=color, s=5, cmap='tab20')
    plt.title(title); plt.colorbar(sc, fraction=0.046, pad=0.04)
    os.makedirs(os.path.dirname(out_png), exist_ok=True)
    plt.tight_layout(); plt.savefig(out_png, dpi=150); plt.close()
