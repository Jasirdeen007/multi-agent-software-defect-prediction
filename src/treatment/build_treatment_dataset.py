from __future__ import annotations

import argparse
import json
from pathlib import Path
import pandas as pd


def build_treatment_dataset(
    sample_parquet: str = "data/processed/audit_sample.parquet",
    audited_parquet: str = "data/processed/audited_labels.parquet",
    output_parquet: str = "data/processed/treatment_sample.parquet",
    summary_path: str = "outputs/results/treatment/treatment_dataset_summary.json",
) -> pd.DataFrame:
    """Merges selective sample with multi-agent audited labels to produce the treatment dataset.

    Preserves `original_label` and assigns `treatment_label = judge_label` where audited,
    ensuring strict provenance and no silent overwrites.
    """
    print(f"Loading selective sample from {sample_parquet}...")
    sample_df = pd.read_parquet(sample_parquet)
    sample_df["issue_id"] = sample_df["issue_id"].astype(str)
    total_sample = len(sample_df)

    audited_file = Path(audited_parquet)
    if not audited_file.exists():
        raise FileNotFoundError(f"Audited labels parquet not found at: {audited_parquet}")

    print(f"Loading audited labels from {audited_parquet}...")
    audited_df = pd.read_parquet(audited_parquet)
    audited_df["issue_id"] = audited_df["issue_id"].astype(str)
    
    # Audit columns to merge
    audit_cols = [
        "source", "project", "issue_id", "judge_label", "judge_confidence",
        "agent_agreement", "label_flipped", "judge_reasoning"
    ]
    audited_sub = audited_df[[c for c in audit_cols if c in audited_df.columns]].copy()
    
    # Merge on composite key (source, project, issue_id)
    merged = sample_df.merge(
        audited_sub,
        on=["source", "project", "issue_id"],
        how="left",
    )

    # Treatment label assignment
    merged["is_audited"] = merged["judge_label"].notna()
    merged["treatment_label"] = merged["original_label"]
    merged.loc[merged["is_audited"], "treatment_label"] = merged.loc[merged["is_audited"], "judge_label"]

    # Normalize labels to binary strings 'bug' / 'non-bug'
    merged["treatment_label"] = merged["treatment_label"].apply(
        lambda x: "non-bug" if "non" in str(x).lower() else "bug"
    )
    merged["original_label"] = merged["original_label"].apply(
        lambda x: "non-bug" if "non" in str(x).lower() else "bug"
    )

    audited_count = int(merged["is_audited"].sum())
    flipped_count = int((merged["is_audited"] & (merged["treatment_label"] != merged["original_label"])).sum())
    flip_rate = float(flipped_count / audited_count) if audited_count > 0 else 0.0

    orig_dist = merged["original_label"].value_counts(normalize=True).to_dict()
    treat_dist = merged["treatment_label"].value_counts(normalize=True).to_dict()

    summary = {
        "total_records": total_sample,
        "audited_records": audited_count,
        "unaudited_records": total_sample - audited_count,
        "flipped_labels": flipped_count,
        "audit_flip_rate": round(flip_rate, 4),
        "original_distribution": {k: round(float(v), 4) for k, v in orig_dist.items()},
        "treatment_distribution": {k: round(float(v), 4) for k, v in treat_dist.items()},
        "timestamp": pd.Timestamp.now().isoformat(),
    }

    out_file = Path(output_parquet)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    merged.to_parquet(out_file, index=False)
    print(f"Saved treatment dataset ({len(merged)} rows) to: {out_file}")

    sum_file = Path(summary_path)
    sum_file.parent.mkdir(parents=True, exist_ok=True)
    with open(sum_file, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"Saved summary to: {sum_file}")

    print("\n--- Treatment Dataset Summary ---")
    print(f"  Total records:         {total_sample}")
    print(f"  Audited records:       {audited_count}")
    print(f"  Flipped labels:        {flipped_count} ({flip_rate * 100:.2f}%)")
    print(f"  Original Bug/Non-Bug:  {orig_dist.get('bug', 0)*100:.1f}% / {orig_dist.get('non-bug', 0)*100:.1f}%")
    print(f"  Treatment Bug/Non-Bug: {treat_dist.get('bug', 0)*100:.1f}% / {treat_dist.get('non-bug', 0)*100:.1f}%")

    return merged


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build Agent-Audited Treatment Dataset")
    parser.add_argument("--sample-parquet", default="data/processed/audit_sample.parquet")
    parser.add_argument("--audited-parquet", default="data/processed/audited_labels.parquet")
    parser.add_argument("--output-parquet", default="data/processed/treatment_sample.parquet")
    parser.add_argument("--summary-path", default="outputs/results/treatment/treatment_dataset_summary.json")
    args = parser.parse_args()

    build_treatment_dataset(
        sample_parquet=args.sample_parquet,
        audited_parquet=args.audited_parquet,
        output_parquet=args.output_parquet,
        summary_path=args.summary_path,
    )
