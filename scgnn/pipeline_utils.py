import json
from pathlib import Path
from typing import Any, Dict

import numpy as np
import scipy.sparse as sp
import yaml


PROJECT_ROOT = Path(__file__).resolve().parent


def resolve_path(path_like: str) -> Path:
    """Return an absolute path inside the project."""
    path = Path(path_like)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path


def ensure_parent_dir(path: Path) -> None:
    """Create the parent directory for path if needed."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)


def load_config(config_path: str) -> Dict[str, Any]:
    """Load a YAML config with paths resolved relative to the repo root."""
    resolved = resolve_path(config_path)
    with resolved.open("r") as handle:
        return yaml.safe_load(handle)


def save_csr_matrix(matrix: sp.spmatrix, output_path: str) -> None:
    """Persist a CSR sparse matrix using the same structure as pbmc artifacts."""
    matrix = sp.csr_matrix(matrix)
    path = resolve_path(output_path)
    ensure_parent_dir(path)
    np.savez_compressed(
        path,
        data=matrix.data,
        indices=matrix.indices,
        indptr=matrix.indptr,
        shape=matrix.shape,
        format=np.bytes_(b"csr"),
    )


def save_json(obj: Dict[str, Any], output_path: str) -> None:
    """Write a JSON file with pretty formatting."""
    path = resolve_path(output_path)
    ensure_parent_dir(path)
    with path.open("w") as handle:
        json.dump(obj, handle, indent=2)
