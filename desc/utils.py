import numpy as np, json, os
from sklearn.metrics import silhouette_score, adjusted_rand_score, normalized_mutual_info_score

def save_json(obj, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w') as f: json.dump(obj, f, indent=2)

def compute_metrics(emb, labels_pred, labels_true=None):
    out = {}
    if emb is not None and len(np.unique(labels_pred)) > 1:
        try: out['silhouette'] = float(silhouette_score(emb, labels_pred))
        except Exception: out['silhouette'] = None
    else:
        out['silhouette'] = None
    if labels_true is not None:
        try:
            out['ari'] = float(adjusted_rand_score(labels_true, labels_pred))
            out['nmi'] = float(normalized_mutual_info_score(labels_true, labels_pred))
        except Exception:
            out['ari'] = None; out['nmi'] = None
    else:
        out['ari'] = None; out['nmi'] = None
    return out
