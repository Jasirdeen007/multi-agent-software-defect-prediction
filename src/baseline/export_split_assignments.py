from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def export_split_assignments(
    split_dir: str | Path = "data/processed/fixed_splits",
    output_path: str | Path = "data/processed/kaggle_split_assignments/split_assignments.parquet",
) -> Path:
    """Export frozen split identities plus within-split order for row-order-independent reuse."""
    split_root = Path(split_dir)
    keys = ["source", "project", "issue_id"]
    frames: list[pd.DataFrame] = []
    for name in ("train", "validation", "test"):
        path = split_root / f"{name}.parquet"
        if not path.is_file():
            raise FileNotFoundError(f"Frozen split missing: {path}")
        part = pd.read_parquet(path, columns=keys)
        part[keys] = part[keys].astype(str)
        if part.duplicated(keys).any():
            raise ValueError(f"Composite issue identity is duplicated in {path}")
        part["split"] = name
        part["split_order"] = range(len(part))
        frames.append(part)

    assignments = pd.concat(frames, ignore_index=True)
    if assignments.duplicated(keys).any():
        raise ValueError("Composite issue identities overlap across frozen splits.")
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    assignments.to_parquet(output, index=False)
    print({"path": str(output), "rows": len(assignments), "splits": assignments["split"].value_counts().to_dict()})
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--split-dir", default="data/processed/fixed_splits")
    parser.add_argument("--output", default="data/processed/kaggle_split_assignments/split_assignments.parquet")
    args = parser.parse_args()
    export_split_assignments(args.split_dir, args.output)
