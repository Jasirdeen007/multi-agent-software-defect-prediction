from __future__ import annotations

import argparse
import json
from pathlib import Path
import pandas as pd


def build_treatment_dataset(
    train_path: str = "data/processed/fixed_splits/train.parquet",
    audited_parquet: str = "data/processed/audited_train_labels.parquet",
    output_parquet: str = "data/processed/fixed_splits/treatment_train.parquet",
    summary_path: str = "outputs/results/treatment/treatment_dataset_summary.json",
) -> pd.DataFrame:
    """Attach audited labels to the full frozen training split only."""
    train = pd.read_parquet(train_path)
    train["issue_id"] = train["issue_id"].astype(str)
    audited = pd.read_parquet(audited_parquet)
    audited["issue_id"] = audited["issue_id"].astype(str)
    key = ["source", "project", "issue_id"]
    if audited.duplicated(key).any():
        raise ValueError("Audited output has duplicate composite issue identities.")
    train_keys = set(map(tuple, train[key].astype(str).itertuples(index=False, name=None)))
    audit_keys = set(map(tuple, audited[key].astype(str).itertuples(index=False, name=None)))
    outside = audit_keys - train_keys
    if outside:
        raise ValueError(f"Found {len(outside)} audited issue(s) outside the frozen training split.")

    if not {"judge_label", "judge_confidence", "agent_agreement", "label_flipped"}.issubset(audited.columns):
        raise ValueError("Audited output is missing required Judge/audit provenance fields.")
    cols = key + ["judge_label", "judge_confidence", "agent_agreement", "label_flipped"]
    merged = train.merge(audited[cols], on=key, how="left", validate="one_to_one")
    merged["is_audited"] = merged["judge_label"].notna()
    merged["treatment_label"] = merged["original_label"]
    merged.loc[merged["is_audited"], "treatment_label"] = merged.loc[merged["is_audited"], "judge_label"]
    flips = merged["is_audited"] & (merged["treatment_label"] != merged["original_label"])
    summary = {
        "training_records": len(merged),
        "audited_records": int(merged["is_audited"].sum()),
        "audit_coverage": float(merged["is_audited"].mean()),
        "flipped_labels": int(flips.sum()),
        "flip_rate_among_audited": float(flips.sum() / max(merged["is_audited"].sum(), 1)),
        "training_only_audit_check": True,
        "label_policy": "judge_label for audited training issues; original_label otherwise",
    }
    out = Path(output_parquet)
    out.parent.mkdir(parents=True, exist_ok=True)
    merged.to_parquet(out, index=False)
    summary_file = Path(summary_path)
    summary_file.parent.mkdir(parents=True, exist_ok=True)
    summary_file.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return merged


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", default="data/processed/fixed_splits/train.parquet")
    parser.add_argument("--audited", default="data/processed/audited_train_labels.parquet")
    parser.add_argument("--output", default="data/processed/fixed_splits/treatment_train.parquet")
    args = parser.parse_args()
    build_treatment_dataset(args.train, args.audited, args.output)
