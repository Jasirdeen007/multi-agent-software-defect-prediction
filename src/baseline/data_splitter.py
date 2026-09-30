from __future__ import annotations

import json
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

    columns_to_load = [text_column, label_column, "source", "project", "issue_id"]
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
