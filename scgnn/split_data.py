import argparse
import logging
from typing import Dict, Tuple

import numpy as np
import scanpy as sc
from sklearn.model_selection import train_test_split

from pipeline_utils import load_config, resolve_path, save_json


logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create deterministic data splits.")
    parser.add_argument("--config", required=True, help="Path to YAML config.")
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Regenerate splits even if cached artifacts exist.",
    )
    return parser.parse_args()


def perform_splits(
    indices: np.ndarray, labels: np.ndarray, split_cfg: Dict
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    train_frac = split_cfg["train_fraction"]
    val_frac = split_cfg["val_fraction"]
    test_frac = split_cfg.get("test_fraction", 1.0 - train_frac - val_frac)
    assert abs(train_frac + val_frac + test_frac - 1.0) < 1e-6, "Fractions must sum to 1."
    seed = split_cfg.get("random_state", 0)

    train_idx, holdout_idx = train_test_split(
        indices,
        train_size=train_frac,
        stratify=labels,
        random_state=seed,
        shuffle=True,
    )
    holdout_labels = labels[holdout_idx]
    val_ratio = val_frac / (val_frac + test_frac)
    val_idx, test_idx = train_test_split(
        holdout_idx,
        train_size=val_ratio,
        stratify=holdout_labels,
        random_state=seed,
        shuffle=True,
    )
    return train_idx, val_idx, test_idx


def main():
    args = parse_args()
    cfg = load_config(args.config)
    splits_cfg = cfg["splits"]
    split_json = resolve_path(cfg["artifacts"]["splits_index_path"])
    split_h5ad = resolve_path(cfg["artifacts"]["splits_h5ad_path"])
    if not args.overwrite and split_json.exists() and split_h5ad.exists():
        logging.info("Split artifacts already exist; skipping.")
        return

    processed_path = resolve_path(cfg["artifacts"]["processed_path"])
    logging.info("Reading %s for split generation", processed_path)
    adata = sc.read(processed_path)

    stratify_key = splits_cfg["stratify_key"]
    if stratify_key not in adata.obs:
        raise KeyError(f"Stratify key '{stratify_key}' not found in AnnData obs.")
    labels = adata.obs[stratify_key].astype(str).values
    indices = np.arange(adata.n_obs)
    train_idx, val_idx, test_idx = perform_splits(indices, labels, splits_cfg)

    split_values = np.array(["train"] * adata.n_obs, dtype=object)
    split_values[val_idx] = "val"
    split_values[test_idx] = "test"
    adata.obs["split"] = split_values

    logging.info("Saving split annotations to %s", split_h5ad)
    split_h5ad.parent.mkdir(parents=True, exist_ok=True)
    adata.write(split_h5ad)

    split_summary = {
        "train_indices": train_idx.tolist(),
        "val_indices": val_idx.tolist(),
        "test_indices": test_idx.tolist(),
        "config": {
            "train_fraction": splits_cfg["train_fraction"],
            "val_fraction": splits_cfg["val_fraction"],
            "test_fraction": splits_cfg.get("test_fraction", 1.0 - splits_cfg["train_fraction"] - splits_cfg["val_fraction"]),
            "stratify_key": stratify_key,
            "random_state": splits_cfg.get("random_state", 0),
        },
    }
    save_json(split_summary, split_json)
    logging.info("Stored split indices at %s", split_json)


if __name__ == "__main__":
    main()
