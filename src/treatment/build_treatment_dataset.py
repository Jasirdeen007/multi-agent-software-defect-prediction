from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq


IDENTITY = ["source", "project", "issue_id"]
LABELS = {"bug", "non-bug"}


def _key(row: Any) -> tuple[str, str, str]:
    return tuple(str(row[name]) for name in IDENTITY)


def _label_counts(values: Counter[str]) -> dict[str, int]:
    return {label: int(values.get(label, 0)) for label in ("bug", "non-bug")}


def build_treatment_dataset(
    train_path: str | Path = "data/processed/fixed_splits/train.parquet",
    audit_sample_path: str | Path = "data/processed/audit_train_sample_1000_from_5000.parquet",
    audited_json_path: str | Path = "outputs/results/audited_train_1000_onecall/audited_records.json",
    output_path: str | Path = "data/processed/fixed_splits/treatment_train.parquet",
    summary_path: str | Path = "outputs/results/treatment/treatment_dataset_summary_current.json",
    overwrite: bool = False,
) -> dict[str, Any]:
    """Copy the frozen training split and replace labels only for audited training issues."""
    train_path, sample_path = Path(train_path), Path(audit_sample_path)
    audited_json_path, output_path, summary_path = map(Path, (audited_json_path, output_path, summary_path))
    for required in (train_path, sample_path, audited_json_path):
        if not required.is_file():
            raise FileNotFoundError(f"Required treatment input not found: {required}")
    if output_path.exists() and not overwrite:
        raise FileExistsError(f"Refusing to overwrite treatment dataset: {output_path}; pass --overwrite explicitly.")

    raw_records = json.loads(audited_json_path.read_text(encoding="utf-8"))
    if not isinstance(raw_records, list) or not raw_records:
        raise ValueError("Audit JSON must contain a non-empty list of audited issue records.")
    audit_map: dict[tuple[str, str, str], dict[str, Any]] = {}
    for record in raw_records:
        identity = _key(record)
        if identity in audit_map:
            raise ValueError(f"Duplicate audit identity: {identity}")
        for label_field in ("original_label", "judge_label", "policy_label", "data_label", "pattern_label"):
            if str(record.get(label_field, "")).strip().lower() not in LABELS:
                raise ValueError(f"Invalid or missing {label_field} for {identity}")
        audit_map[identity] = record

    sample = pd.read_parquet(sample_path, columns=IDENTITY + ["original_label"])
    sample_map = {
        _key(row): str(row["original_label"]).strip().lower()
        for _, row in sample.iterrows()
    }
    outside_sample = set(audit_map) - set(sample_map)
    if outside_sample:
        raise ValueError(f"{len(outside_sample)} audited records are outside the selected training-only audit sample.")
    for identity, record in audit_map.items():
        if str(record["original_label"]).strip().lower() != sample_map[identity]:
            raise ValueError(f"Audit original label does not match the preserved sample label for {identity}")

    train_keys = pd.read_parquet(train_path, columns=IDENTITY)
    train_key_set = set(map(tuple, train_keys.astype(str).itertuples(index=False, name=None)))
    outside_train = set(audit_map) - train_key_set
    if outside_train:
        raise ValueError(f"{len(outside_train)} audited records are outside the frozen training split.")
    if len(train_key_set) != len(train_keys):
        raise ValueError("Frozen training split contains duplicate composite identities.")
    del train_keys, train_key_set

    train_file = pq.ParquetFile(train_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = output_path.with_suffix(output_path.suffix + ".tmp")
    writer: pq.ParquetWriter | None = None
    original_counts: Counter[str] = Counter()
    treatment_counts: Counter[str] = Counter()
    audited_count = changed_count = written_rows = 0
    try:
        for batch in train_file.iter_batches(batch_size=50_000):
            table = pa.Table.from_batches([batch])
            data = table.to_pandas()
            originals = data["original_label"].astype(str).str.strip().str.lower().tolist()
            if any(label not in LABELS for label in originals):
                raise ValueError("Frozen training split has invalid original_label values.")
            keys = data[IDENTITY].astype(str).itertuples(index=False, name=None)
            audited_labels: list[str | None] = []
            treatment_labels: list[str] = []
            is_audited: list[bool] = []
            for identity, original in zip(keys, originals):
                record = audit_map.get(tuple(identity))
                if record is None:
                    audit_label = None
                    treatment_label = original
                    audited = False
                else:
                    if str(record["original_label"]).strip().lower() != original:
                        raise ValueError(f"Audit original label differs from frozen train label for {identity}")
                    audit_label = str(record["judge_label"]).strip().lower()
                    treatment_label = audit_label
                    audited = True
                audited_labels.append(audit_label)
                treatment_labels.append(treatment_label)
                is_audited.append(audited)
                audited_count += int(audited)
                changed_count += int(treatment_label != original)
                original_counts[original] += 1
                treatment_counts[treatment_label] += 1

            table = table.append_column("audited_label", pa.array(audited_labels, type=pa.string()))
            table = table.append_column("treatment_label", pa.array(treatment_labels, type=pa.string()))
            table = table.append_column("is_audited", pa.array(is_audited, type=pa.bool_()))
            if writer is None:
                writer = pq.ParquetWriter(temp_path, table.schema, compression="zstd")
            writer.write_table(table)
            written_rows += len(data)
        if writer is None:
            raise ValueError("Frozen training split is empty.")
        writer.close()
        writer = None
        if written_rows != train_file.metadata.num_rows or audited_count != len(audit_map):
            raise ValueError(
                f"Post-write coverage mismatch: wrote {written_rows}/{train_file.metadata.num_rows} training rows; "
                f"matched {audited_count}/{len(audit_map)} audit records."
            )
        temp_path.replace(output_path)
    except Exception:
        if writer is not None:
            writer.close()
        if temp_path.exists():
            temp_path.unlink()
        raise

    audit_methods = Counter(str(record.get("audit_method", "unknown")) for record in raw_records)
    summary = {
        "dataset_path": str(output_path),
        "baseline_train_path": str(train_path),
        "audit_sample_path": str(sample_path),
        "audit_records_path": str(audited_json_path),
        "training_records": int(written_rows),
        "audited_records": int(audited_count),
        "unaudited_records": int(written_rows - audited_count),
        "audit_coverage_of_training": float(audited_count / written_rows),
        "label_changes": int(changed_count),
        "label_change_rate_among_audited": float(changed_count / audited_count) if audited_count else 0.0,
        "original_label_distribution": _label_counts(original_counts),
        "treatment_label_distribution": _label_counts(treatment_counts),
        "audit_methods": dict(audit_methods),
        "label_policy": "judge_label for audited training issues; original_label for all others",
        "original_label_column_preserved": True,
        "validation_and_test_modified": False,
        "caution": "Audited labels are model-generated decisions, not independently verified ground truth.",
    }
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build the treatment training set from the frozen train split and agent audit labels.")
    parser.add_argument("--train", default="data/processed/fixed_splits/train.parquet")
    parser.add_argument("--audit-sample", default="data/processed/audit_train_sample_1000_from_5000.parquet")
    parser.add_argument("--audited-json", default="outputs/results/audited_train_1000_onecall/audited_records.json")
    parser.add_argument("--output", default="data/processed/fixed_splits/treatment_train.parquet")
    parser.add_argument("--summary", default="outputs/results/treatment/treatment_dataset_summary_current.json")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    print(json.dumps(build_treatment_dataset(
        train_path=args.train,
        audit_sample_path=args.audit_sample,
        audited_json_path=args.audited_json,
        output_path=args.output,
        summary_path=args.summary,
        overwrite=args.overwrite,
    ), indent=2))
