import argparse
import json
from pathlib import Path

import pandas as pd

from pipeline_utils import resolve_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize scGNN evaluation metrics.")
    parser.add_argument("--results_root", default="results", help="Root directory of results.")
    parser.add_argument("--output_csv", default="reports/metrics_summary.csv", help="CSV output path.")
    return parser.parse_args()


def read_metrics(result_dir: Path) -> dict:
    metrics_path = result_dir / "metrics.json"
    eval_path = result_dir / "eval_metrics.json"
    payload = {"run": result_dir.name}
    if metrics_path.exists():
        with metrics_path.open("r") as handle:
            metrics = json.load(handle)
        payload["best_epoch"] = metrics.get("best_epoch")
        best = metrics.get("best", {})
        for split in ["train", "val", "test"]:
            if split in best:
                payload[f"{split}_ari"] = best[split].get("ari")
                payload[f"{split}_nmi"] = best[split].get("nmi")
                payload[f"{split}_silhouette"] = best[split].get("silhouette")
    if eval_path.exists():
        with eval_path.open("r") as handle:
            eval_metrics = json.load(handle)
        for split in ["train", "val", "test"]:
            if split in eval_metrics:
                payload[f"{split}_ari_eval"] = eval_metrics[split].get("ari")
                payload[f"{split}_nmi_eval"] = eval_metrics[split].get("nmi")
                payload[f"{split}_silhouette_eval"] = eval_metrics[split].get("silhouette")
    return payload


def main():
    args = parse_args()
    root = resolve_path(args.results_root)
    rows = []
    for path in sorted(root.iterdir()):
        if path.is_dir():
            rows.append(read_metrics(path))
    df = pd.DataFrame(rows)
    output = resolve_path(args.output_csv)
    output.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output, index=False)
    print(f"Saved summary to {output}")


if __name__ == "__main__":
    main()
