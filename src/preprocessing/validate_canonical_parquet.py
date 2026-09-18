"""Stage 2 preprocessing validation for the canonical Parquet dataset."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd
import pyarrow.parquet as pq


DEFAULT_DATASET = Path("data/processed/bughubs_canonical.parquet")
DEFAULT_REPORT = Path("outputs/tables/stage2_preprocessing_validation.json")

REQUIRED_COLUMNS = [
    "issue_id",
    "source",
    "project",
    "title",
    "description",
    "component",
    "severity",
    "priority",
    "status",
    "resolution",
    "original_label",
    "label_source",
    "created_at",
    "updated_at",
    "model_text",
]
VALID_LABELS = {"bug", "non-bug"}
IDENTITY_COLUMNS = ["source", "project", "issue_id"]


def blank_count(series: pd.Series) -> int:
    values = series.astype("string")
    return int((values.isna() | values.str.strip().eq("")).sum())


def validate_dataset(dataset_path: Path) -> dict[str, Any]:
    if not dataset_path.exists():
        raise FileNotFoundError(f"Parquet dataset not found: {dataset_path}")

    parquet_file = pq.ParquetFile(dataset_path)
    schema_columns = parquet_file.schema.names
    missing_columns = [column for column in REQUIRED_COLUMNS if column not in schema_columns]
    unexpected_columns = [column for column in schema_columns if column not in REQUIRED_COLUMNS]

    if missing_columns:
        return {
            "dataset_path": dataset_path.as_posix(),
            "row_count": int(parquet_file.metadata.num_rows),
            "column_count": len(schema_columns),
            "required_columns_present": False,
            "missing_required_columns": missing_columns,
            "unexpected_columns": unexpected_columns,
            "stage2_passed": False,
        }

    df = pd.read_parquet(dataset_path, columns=REQUIRED_COLUMNS)
    labels = set(df["original_label"].dropna().unique())
    invalid_labels = sorted(labels - VALID_LABELS)
    duplicate_count = int(df.duplicated(IDENTITY_COLUMNS).sum())

    created = pd.to_datetime(df["created_at"], errors="coerce", utc=True)
    updated = pd.to_datetime(df["updated_at"], errors="coerce", utc=True)
    model_text = df["model_text"].fillna("").astype("string").str.strip()
    title = df["title"].fillna("").astype("string").str.strip()
    description = df["description"].fillna("").astype("string").str.strip()

    report = {
        "dataset_path": dataset_path.as_posix(),
        "row_count": int(parquet_file.metadata.num_rows),
        "column_count": len(schema_columns),
        "required_columns_present": True,
        "missing_required_columns": missing_columns,
        "unexpected_columns": unexpected_columns,
        "valid_labels": sorted(VALID_LABELS),
        "observed_labels": sorted(labels),
        "invalid_labels": invalid_labels,
        "duplicate_composite_identities": duplicate_count,
        "missing_identity_values": {
            column: blank_count(df[column]) for column in IDENTITY_COLUMNS
        },
        "unparseable_created_at": int(df["created_at"].notna().sum() - created.notna().sum()),
        "unparseable_updated_at": int(df["updated_at"].notna().sum() - updated.notna().sum()),
        "blank_title": int(title.eq("").sum()),
        "blank_description": int(description.eq("").sum()),
        "blank_model_text": int(model_text.eq("").sum()),
        "model_text_lt_20_chars": int(model_text.str.len().lt(20).sum()),
        "warnings": [],
        "stage2_passed": False,
    }

    if report["unparseable_created_at"] > 0:
        report["warnings"].append(
            "Some created_at values could not be parsed by pandas; inspect temporal analyses before using time-based splits."
        )
    if report["model_text_lt_20_chars"] > 0:
        report["warnings"].append(
            "Some model_text values are shorter than 20 characters; no records were filtered by this validation script."
        )

    report["stage2_passed"] = (
        not report["invalid_labels"]
        and report["duplicate_composite_identities"] == 0
        and all(value == 0 for value in report["missing_identity_values"].values())
        and report["blank_model_text"] == 0
    )
    return report


def write_report(report: dict[str, Any], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate Stage 2 preprocessing assumptions for canonical Parquet data."
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=DEFAULT_DATASET,
        help=f"Canonical Parquet dataset. Default: {DEFAULT_DATASET}",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_REPORT,
        help=f"Validation report path. Default: {DEFAULT_REPORT}",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = validate_dataset(args.dataset)
    write_report(report, args.output)
    print(json.dumps(report, indent=2))
    print(f"Validation report written to {args.output}")


if __name__ == "__main__":
    main()
