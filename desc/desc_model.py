import torch
import torch.nn as nn

class AutoEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dims=(1024,256), latent_dim=32, dropout=0.1):
        super().__init__()
        h1, h2 = hidden_dims
        self.enc = nn.Sequential(
            nn.Linear(input_dim, h1), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(h1, h2), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(h2, latent_dim)
        )
        self.dec = nn.Sequential(
            nn.Linear(latent_dim, h2), nn.ReLU(),
            nn.Linear(h2, h1), nn.ReLU(),
            nn.Linear(h1, input_dim)
        )

    def encode(self, x): return self.enc(x)
    def decode(self, z): return self.dec(z)
    def forward(self, x):
        z = self.encode(x); x_hat = self.decode(z)
        return z, x_hat

def student_t_soft_assign(z, centers, alpha=1.0, eps=1e-8):
    z_exp = z.unsqueeze(1); c_exp = centers.unsqueeze(0)
    dist2 = ((z_exp - c_exp)**2).sum(dim=2)
    num = (1.0 + dist2/alpha) ** (-(alpha+1.0)/2.0)
    return num / (num.sum(dim=1, keepdim=True) + eps)

def target_distribution(Q, eps=1e-8):
    weight = Q**2 / (Q.sum(dim=0, keepdim=True) + eps)
    return weight / (weight.sum(dim=1, keepdim=True) + eps)

def kl_divergence(P, Q, eps=1e-8):
    return (P * (P.add(eps).log() - Q.add(eps).log())).sum()

def batch_center_loss(z, batch_ids):
    if batch_ids is None: return torch.tensor(0.0, device=z.device)
    uniq = torch.unique(batch_ids)
    if uniq.numel() <= 1: return torch.tensor(0.0, device=z.device)
    g = z.mean(dim=0, keepdim=True)
    loss = 0.0
    for b in uniq:
        zb = z[batch_ids==b]
        if zb.shape[0]==0: continue
        mb = zb.mean(dim=0, keepdim=True)
        loss = loss + ((mb - g)**2).mean()
    return loss / uniq.numel()
