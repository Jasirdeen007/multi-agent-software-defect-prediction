from __future__ import annotations

import json
import argparse
import hashlib
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


def load_dataset(
    dataset_path: str | Path,
    text_column: str = "model_text",
    label_column: str = "original_label",
    sample_size: int | None = None,
    random_seed: int = 42,
) -> pd.DataFrame:
    path = Path(dataset_path)
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found at {path.resolve()}")

    columns_to_load = [
        "issue_id", "source", "project", "title", "description", "component",
        "severity", "priority", "status", "resolution", label_column,
        "label_source", "created_at", "updated_at", text_column,
    ]
    columns_to_load = list(dict.fromkeys(columns_to_load))
    print(f"Loading columns {columns_to_load} from {path}...")
    df = pd.read_parquet(path, columns=columns_to_load)

    initial_len = len(df)
    df = df.dropna(subset=[text_column, label_column]).copy()
    df[text_column] = df[text_column].astype(str).str.strip()
    df = df[df[text_column].str.len() >= 20].copy()

    if len(df) < initial_len:
        print(f"Filtered out {initial_len - len(df)} records with short/empty text. Remaining: {len(df):,}")

    if sample_size is not None and sample_size < len(df):
        print(f"Extracting stratified sample of {sample_size:,} records from {len(df):,} total records...")
        df, _ = train_test_split(
            df,
            train_size=sample_size,
            stratify=df[label_column],
            random_state=random_seed,
        )
        df = df.reset_index(drop=True)

    print(f"Loaded dataset: {len(df):,} records.")
    return df


def load_or_create_fixed_splits(
    dataset_path: str | Path,
    split_dir: str | Path = "data/processed/fixed_splits",
    label_column: str = "original_label",
    sample_size: int | None = None,
    test_size: float = 0.15,
    val_size: float = 0.15,
    random_seed: int = 42,
    split_assignments_path: str | Path | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Create once and then reuse immutable train/validation/test Parquet splits."""
    split_path = Path(split_dir)
    files = [split_path / f"{part}.parquet" for part in ("train", "validation", "test")]
    if all(path.exists() for path in files):
        splits = [pd.read_parquet(path) for path in files]
        validate_split_disjointness(*splits)
        manifest_path = split_path / "split_manifest.json"
        if manifest_path.exists():
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            if manifest["random_seed"] != random_seed:
                raise ValueError("Requested seed differs from the frozen split manifest; use a new split directory to change it.")
            if manifest["test_size"] != test_size or manifest["val_size"] != val_size:
                raise ValueError("Requested split ratios differ from the frozen split manifest.")
            for name, frame in zip(("train", "validation", "test"), splits):
                if manifest["identity_sha256"][name] != _identity_digest(frame):
                    raise ValueError(f"Frozen {name} split identities changed after manifest creation.")
        else:
            _write_split_manifest(split_path, splits, random_seed, test_size, val_size)
        if split_assignments_path is not None:
            source_df = load_dataset(dataset_path, sample_size=sample_size, random_seed=random_seed)
            assigned_splits = apply_split_assignments(source_df, split_assignments_path)
            for name, current, expected in zip(("train", "validation", "test"), splits, assigned_splits):
                if _identity_digest(current) != _identity_digest(expected):
                    raise ValueError(f"Cached {name} identities do not match the supplied frozen split assignments.")
        return tuple(splits)  # type: ignore[return-value]

    df = load_dataset(dataset_path, sample_size=sample_size, random_seed=random_seed)
    if split_assignments_path is not None:
        train, val, test = apply_split_assignments(df, split_assignments_path)
    else:
        train, val, test = split_dataset(
            df, label_column=label_column, test_size=test_size, val_size=val_size,
            random_seed=random_seed,
        )
    # Remove derived numeric target: each experiment derives its target from its own label column.
    for part in (train, val, test):
        part.drop(columns=["target"], errors="ignore", inplace=True)
    split_path.mkdir(parents=True, exist_ok=True)
    for path, part in zip(files, (train, val, test)):
        part.to_parquet(path, index=False)
    _write_split_manifest(split_path, (train, val, test), random_seed, test_size, val_size)
    validate_split_disjointness(train, val, test)
    return train, val, test


def apply_split_assignments(
    df: pd.DataFrame,
    assignments_path: str | Path,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Apply a frozen identity-to-split mapping independent of source row order."""
    keys = ["source", "project", "issue_id"]
    assignments = pd.read_parquet(assignments_path, columns=keys + ["split", "split_order"])
    source = df.copy()
    source[keys] = source[keys].astype(str)
    assignments[keys] = assignments[keys].astype(str)
    if assignments.duplicated(keys).any() or source.duplicated(keys).any():
        raise ValueError("Canonical data and split assignments must have unique composite identities.")
    if len(source) != len(assignments):
        raise ValueError(f"Split assignment row count {len(assignments)} differs from eligible data count {len(source)}.")
    merged = assignments.merge(source, on=keys, how="left", validate="one_to_one", indicator=True)
    if not merged["_merge"].eq("both").all():
        absent = int((merged["_merge"] != "both").sum())
        raise ValueError(f"{absent} split identities are absent from the eligible canonical dataset.")
    unexpected = set(merged["split"].astype(str)) - {"train", "validation", "test"}
    if unexpected:
        raise ValueError(f"Invalid split names in assignment file: {sorted(unexpected)}")

    parts = []
    for name in ("train", "validation", "test"):
        part = merged.loc[merged["split"].eq(name)].sort_values("split_order")
        part = part.drop(columns=["split", "split_order", "_merge"]).reset_index(drop=True)
        parts.append(part)
    return tuple(parts)  # type: ignore[return-value]


def _write_split_manifest(split_path: Path, splits: tuple[pd.DataFrame, ...] | list[pd.DataFrame], seed: int, test_size: float, val_size: float) -> None:
    key = ["source", "project", "issue_id"]
    manifest: dict[str, Any] = {
        "random_seed": seed,
        "test_size": test_size,
        "val_size": val_size,
        "train_size": len(splits[0]),
        "validation_size_records": len(splits[1]),
        "test_records": len(splits[2]),
        "identity_columns": key,
        "identity_sha256": {},
    }
    for name, frame in zip(("train", "validation", "test"), splits):
        manifest["identity_sha256"][name] = _identity_digest(frame)
    (split_path / "split_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def _identity_digest(frame: pd.DataFrame) -> str:
    key = ["source", "project", "issue_id"]
    ids = frame[key].astype(str).to_csv(index=False, header=False).encode("utf-8")
    return hashlib.sha256(ids).hexdigest()


def validate_split_disjointness(train: pd.DataFrame, validation: pd.DataFrame, test: pd.DataFrame) -> None:
    """Raise if issue identities overlap between frozen splits."""
    keys = ["source", "project", "issue_id"]
    sets = [set(map(tuple, part[keys].astype(str).itertuples(index=False, name=None))) for part in (train, validation, test)]
    if sets[0] & sets[1] or sets[0] & sets[2] or sets[1] & sets[2]:
        raise ValueError("Composite issue identities overlap across fixed data splits.")


def split_dataset(
    df: pd.DataFrame,
    label_column: str = "original_label",
    label_map: dict[str, int] | None = None,
    test_size: float = 0.15,
    val_size: float = 0.15,
    random_seed: int = 42,
    output_dir: str | Path | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if label_map is None:
        label_map = {"bug": 1, "non-bug": 0}

    df["target"] = df[label_column].map(label_map)
    if df["target"].isna().any():
        unmapped = df[df["target"].isna()][label_column].unique()
        raise ValueError(f"Unmapped labels found: {unmapped}")
    df["target"] = df["target"].astype(int)

    train_val_df, test_df = train_test_split(
        df,
        test_size=test_size,
        stratify=df["target"],
        random_state=random_seed,
    )

    val_relative_size = val_size / (1.0 - test_size)
    train_df, val_df = train_test_split(
        train_val_df,
        test_size=val_relative_size,
        stratify=train_val_df["target"],
        random_state=random_seed,
    )

    train_df = train_df.reset_index(drop=True)
    val_df = val_df.reset_index(drop=True)
    test_df = test_df.reset_index(drop=True)

    splits_info: dict[str, Any] = {
        "random_seed": random_seed,
        "total_records": len(df),
        "train": {
            "records": len(train_df),
            "percentage": round(len(train_df) / len(df) * 100, 2),
            "bug_count": int((train_df["target"] == 1).sum()),
            "non_bug_count": int((train_df["target"] == 0).sum()),
            "bug_ratio": round(float((train_df["target"] == 1).mean()), 4),
        },
        "validation": {
            "records": len(val_df),
            "percentage": round(len(val_df) / len(df) * 100, 2),
            "bug_count": int((val_df["target"] == 1).sum()),
            "non_bug_count": int((val_df["target"] == 0).sum()),
            "bug_ratio": round(float((val_df["target"] == 1).mean()), 4),
        },
        "test": {
            "records": len(test_df),
            "percentage": round(len(test_df) / len(df) * 100, 2),
            "bug_count": int((test_df["target"] == 1).sum()),
            "non_bug_count": int((test_df["target"] == 0).sum()),
            "bug_ratio": round(float((test_df["target"] == 1).mean()), 4),
        },
    }

    print("--- Dataset Splits Summary ---")
    print(f"Train: {splits_info['train']['records']:,} ({splits_info['train']['bug_ratio']*100:.1f}% bug)")
    print(f"Val:   {splits_info['validation']['records']:,} ({splits_info['validation']['bug_ratio']*100:.1f}% bug)")
    print(f"Test:  {splits_info['test']['records']:,} ({splits_info['test']['bug_ratio']*100:.1f}% bug)")

    if output_dir is not None:
        out_path = Path(output_dir)
        out_path.mkdir(parents=True, exist_ok=True)
        summary_file = out_path / "splits_summary.json"
        with open(summary_file, "w", encoding="utf-8") as f:
            json.dump(splits_info, f, indent=2)
        print(f"Splits summary saved to: {summary_file}")

    return train_df, val_df, test_df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create/reuse the fixed baseline-treatment split.")
    parser.add_argument("--dataset", default="data/processed/bughubs_canonical.parquet")
    parser.add_argument("--split-dir", default="data/processed/fixed_splits")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    parts = load_or_create_fixed_splits(args.dataset, args.split_dir, random_seed=args.seed)
    print("Fixed split rows:", {name: len(part) for name, part in zip(("train", "validation", "test"), parts)})
