"""Create a reproducible representative sample for multi-agent label auditing.

This script does not create new labels and does not modify the canonical dataset.
It selects records for later Policy/Data/Pattern/Judge auditing while preserving
all canonical fields and original BugHub label provenance.
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


DEFAULT_DATASET = Path("data/processed/fixed_splits/train.parquet")
DEFAULT_SAMPLE = Path("data/processed/audit_train_sample.parquet")
DEFAULT_SUMMARY = Path("outputs/results/audit_sampling/audit_train_sample_summary.json")
DEFAULT_IDS = Path("outputs/results/audit_sampling/audit_train_sample_ids.csv")
DEFAULT_SAMPLE_SIZE = 5_000
DEFAULT_RANDOM_SEED = 42

CANONICAL_COLUMNS = [
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
IDENTITY_COLUMNS = ["source", "project", "issue_id"]
STRATA_COLUMNS = ["project", "original_label"]
METADATA_COLUMNS = ["component", "severity", "priority", "status", "resolution", "created_at", "updated_at"]


def load_dataset(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Canonical dataset not found: {path}")
    import pyarrow.parquet as pq
    available = set(pq.read_schema(path).names)
    optional_signals = [
        "agent_disagreement", "judge_confidence",
        "policy_label", "data_label", "pattern_label",
    ]
    columns = CANONICAL_COLUMNS + [column for column in optional_signals if column in available]
    df = pd.read_parquet(path, columns=columns)
    missing = [column for column in CANONICAL_COLUMNS if column not in df.columns]
    if missing:
        raise ValueError(f"Missing required canonical columns: {missing}")
    return df


def assign_sample_counts(strata_counts: pd.Series, sample_size: int) -> dict[tuple[str, str], int]:
    total = int(strata_counts.sum())
    if sample_size > total:
        raise ValueError(f"sample_size={sample_size} exceeds dataset rows={total}")

    raw = strata_counts / total * sample_size
    floor_counts = np.floor(raw).astype(int)
    remainder = sample_size - int(floor_counts.sum())
    fractions = (raw - floor_counts).sort_values(ascending=False)
    for key in fractions.index[:remainder]:
        floor_counts.loc[key] += 1
    return {key: int(value) for key, value in floor_counts.items()}


def create_sample(df: pd.DataFrame, sample_size: int, random_seed: int) -> pd.DataFrame:
    if df.duplicated(IDENTITY_COLUMNS).any():
        raise ValueError("Canonical dataset contains duplicate composite identities.")
    if df[IDENTITY_COLUMNS].isna().any().any():
        raise ValueError("Canonical dataset contains missing identity values.")

    # Allocate by sqrt(project population): every project is represented while
    # large projects receive less than their population-proportional share.
    project_counts = df.groupby("project", dropna=False).size()
    if sample_size < len(project_counts):
        raise ValueError("sample_size must be at least the number of projects for project coverage.")
    weights = np.sqrt(project_counts.astype(float))
    raw = weights / weights.sum() * sample_size
    project_quota = np.floor(raw).astype(int).clip(lower=1)
    while int(project_quota.sum()) > sample_size:
        candidates = project_quota[project_quota > 1]
        project_quota.loc[candidates.index[np.argmin((raw.loc[candidates.index] - candidates).to_numpy())]] -= 1
    while int(project_quota.sum()) < sample_size:
        candidates = project_quota[project_quota < project_counts]
        if candidates.empty:
            raise ValueError("Requested sample exceeds available training records.")
        project_quota.loc[candidates.index[np.argmax((raw.loc[candidates.index] - candidates).to_numpy())]] += 1

    pieces: list[pd.DataFrame] = []
    for project, project_n in project_quota.items():
        project_df = df[df["project"] == project]
        label_counts = project_df.groupby("original_label", dropna=False).size()
        label_quota = assign_sample_counts(label_counts, int(project_n))
        for label, count in label_quota.items():
            stratum_df = project_df[project_df["original_label"] == label].copy()
            if count > len(stratum_df):
                raise ValueError(f"Requested {count} rows from {project}|{label}, only {len(stratum_df)} exist.")
            # If candidate triage signals exist, prioritize disagreement and
            # uncertainty within each project/class quota; otherwise sample randomly.
            score = pd.Series(0.0, index=stratum_df.index)
            if "agent_disagreement" in stratum_df:
                score += pd.to_numeric(stratum_df["agent_disagreement"], errors="coerce").fillna(0)
            elif {"policy_label", "data_label", "pattern_label"}.issubset(stratum_df.columns):
                agent_labels = stratum_df[["policy_label", "data_label", "pattern_label"]].astype("string")
                score += agent_labels.nunique(axis=1).sub(1).div(2).fillna(0)
            if "judge_confidence" in stratum_df:
                confidence = pd.to_numeric(stratum_df["judge_confidence"], errors="coerce")
                score += 1.0 - confidence.fillna(1.0)
            stratum_df["_priority"] = score
            if score.any():
                selected = stratum_df.sort_values("_priority", ascending=False, kind="stable").head(count)
                selected = selected.sample(frac=1.0, random_state=random_seed)
            else:
                selected = stratum_df.sample(n=count, random_state=random_seed)
            pieces.append(selected.drop(columns=["_priority"], errors="ignore"))

    sample = pd.concat(pieces, ignore_index=True)
    sample = sample.sample(frac=1.0, random_state=random_seed).reset_index(drop=True)
    sample["sampling_method"] = "sqrt_project_weighted_stratified_original_label"
    sample["sampling_seed"] = random_seed
    sample["sampling_stratum"] = sample["project"].astype(str) + "|" + sample["original_label"].astype(str)
    return sample


def include_required_identities(
    sample: pd.DataFrame,
    population: pd.DataFrame,
    required_keys: set[tuple[str, str, str]],
    random_seed: int,
) -> pd.DataFrame:
    """Include existing audit identities while preserving every project/label quota."""
    required = population[
        population[IDENTITY_COLUMNS].astype(str).apply(tuple, axis=1).isin(required_keys)
    ].copy()
    found = set(map(tuple, required[IDENTITY_COLUMNS].astype(str).itertuples(index=False, name=None)))
    if found != required_keys:
        raise ValueError(f"{len(required_keys - found)} requested preserved audit identity(ies) are not in the sample population.")

    sample = sample.copy()
    selected = set(map(tuple, sample[IDENTITY_COLUMNS].astype(str).itertuples(index=False, name=None)))
    missing = required[~required[IDENTITY_COLUMNS].astype(str).apply(tuple, axis=1).isin(selected)]
    for offset, (_, row) in enumerate(missing.iterrows()):
        same_stratum = (sample["project"].astype(str) == str(row["project"])) & (
            sample["original_label"].astype(str) == str(row["original_label"])
        )
        removable = sample.loc[same_stratum]
        removable_keys = set(map(tuple, removable[IDENTITY_COLUMNS].astype(str).itertuples(index=False, name=None)))
        removable = removable[
            removable[IDENTITY_COLUMNS].astype(str).apply(tuple, axis=1).map(lambda key: key not in required_keys)
        ]
        if removable.empty:
            raise ValueError(f"Cannot preserve required audit identity in stratum {row['project']}|{row['original_label']}.")
        drop_index = removable.sample(n=1, random_state=random_seed + offset).index[0]
        sample = sample.drop(index=drop_index)
        sample = pd.concat([sample, row.to_frame().T], ignore_index=True)

    if len(sample) != len(selected) or sample.duplicated(IDENTITY_COLUMNS).any():
        raise ValueError("Preserving existing audits changed sample size or introduced duplicate identities.")
    return sample.reset_index(drop=True)


def distribution(df: pd.DataFrame, columns: list[str]) -> list[dict[str, Any]]:
    rows = (
        df.groupby(columns, dropna=False, observed=False)
        .size()
        .reset_index(name="records")
        .sort_values(columns)
    )
    rows["percentage"] = (rows["records"] / len(df) * 100).round(4)
    return rows.to_dict(orient="records")


def add_text_buckets(df: pd.DataFrame) -> pd.Series:
    lengths = df["model_text"].fillna("").astype(str).str.len()
    return pd.cut(
        lengths,
        bins=[0, 100, 250, 500, 1000, 2500, 5000, 10000, np.inf],
        labels=["0-99", "100-249", "250-499", "500-999", "1000-2499", "2500-4999", "5000-9999", "10000+"],
        right=False,
    )


def metadata_missing_bucket(df: pd.DataFrame) -> pd.Series:
    missing_counts = df[METADATA_COLUMNS].isna().sum(axis=1)
    return pd.cut(
        missing_counts,
        bins=[-1, 0, 2, len(METADATA_COLUMNS)],
        labels=["complete", "partially_missing", "sparse"],
    )


def build_summary(df: pd.DataFrame, sample: pd.DataFrame, dataset_path: Path, sample_path: Path, sample_size: int, random_seed: int) -> dict[str, Any]:
    sample_for_summary = sample.copy()
    sample_for_summary["text_length_bucket"] = add_text_buckets(sample_for_summary)
    sample_for_summary["metadata_missing_bucket"] = metadata_missing_bucket(sample_for_summary)

    dataset_for_summary = df.copy()
    dataset_for_summary["text_length_bucket"] = add_text_buckets(dataset_for_summary)
    dataset_for_summary["metadata_missing_bucket"] = metadata_missing_bucket(dataset_for_summary)

    return {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "dataset_path": dataset_path.as_posix(),
        "sample_path": sample_path.as_posix(),
        "sampling_method": "sqrt_project_weighted_stratified_original_label",
        "sample_size_requested": sample_size,
        "sample_size_actual": int(len(sample)),
        "random_seed": random_seed,
        "identity_columns": IDENTITY_COLUMNS,
        "strata_columns": STRATA_COLUMNS,
        "preserved_columns": CANONICAL_COLUMNS,
        "dataset_rows": int(len(df)),
        "dataset_unique_projects": int(df["project"].nunique(dropna=True)),
        "sample_unique_projects": int(sample["project"].nunique(dropna=True)),
        "sample_duplicate_composite_identities": int(sample.duplicated(IDENTITY_COLUMNS).sum()),
        "dataset_distribution": {
            "source": distribution(dataset_for_summary, ["source"]),
            "original_label": distribution(dataset_for_summary, ["original_label"]),
            "source_original_label": distribution(dataset_for_summary, ["source", "original_label"]),
            "text_length_bucket": distribution(dataset_for_summary, ["text_length_bucket"]),
            "metadata_missing_bucket": distribution(dataset_for_summary, ["metadata_missing_bucket"]),
        },
        "sample_distribution": {
            "source": distribution(sample_for_summary, ["source"]),
            "original_label": distribution(sample_for_summary, ["original_label"]),
            "source_original_label": distribution(sample_for_summary, ["source", "original_label"]),
            "text_length_bucket": distribution(sample_for_summary, ["text_length_bucket"]),
            "metadata_missing_bucket": distribution(sample_for_summary, ["metadata_missing_bucket"]),
        },
        "top_projects_in_sample": (
            sample["project"]
            .value_counts(dropna=False)
            .head(25)
            .rename_axis("project")
            .reset_index(name="records")
            .to_dict(orient="records")
        ),
    }


def write_outputs(sample: pd.DataFrame, summary: dict[str, Any], sample_path: Path, summary_path: Path, ids_path: Path) -> None:
    sample_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    ids_path.parent.mkdir(parents=True, exist_ok=True)

    sample.to_parquet(sample_path, index=False)
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    id_columns = IDENTITY_COLUMNS + ["original_label", "sampling_method", "sampling_seed", "sampling_stratum"]
    sample[id_columns].to_csv(ids_path, index=False)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create representative audit sample from fixed training split only.")
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--sample-output", type=Path, default=DEFAULT_SAMPLE)
    parser.add_argument("--summary-output", type=Path, default=DEFAULT_SUMMARY)
    parser.add_argument("--ids-output", type=Path, default=DEFAULT_IDS)
    parser.add_argument("--sample-size", type=int, default=DEFAULT_SAMPLE_SIZE)
    parser.add_argument("--random-seed", type=int, default=DEFAULT_RANDOM_SEED)
    parser.add_argument("--preserve-audited-json", type=Path, action="append", default=[])
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    df = load_dataset(args.dataset)
    sample = create_sample(df, args.sample_size, args.random_seed)
    if args.preserve_audited_json:
        required_keys: set[tuple[str, str, str]] = set()
        for path in args.preserve_audited_json:
            records = json.loads(path.read_text(encoding="utf-8"))
            required_keys.update((str(row["source"]), str(row["project"]), str(row["issue_id"])) for row in records)
        sample = include_required_identities(sample, df, required_keys, args.random_seed)
    summary = build_summary(df, sample, args.dataset, args.sample_output, args.sample_size, args.random_seed)
    write_outputs(sample, summary, args.sample_output, args.summary_output, args.ids_output)

    print(f"Audit sample written to: {args.sample_output}")
    print(f"Sample identities written to: {args.ids_output}")
    print(f"Sampling summary written to: {args.summary_output}")
    print(f"Rows: {len(sample):,} / Seed: {args.random_seed}")


if __name__ == "__main__":
    main()
