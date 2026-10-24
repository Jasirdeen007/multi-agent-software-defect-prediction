from __future__ import annotations

import json
from pathlib import Path
import pandas as pd
from src.agents.schemas import AuditedIssueRecord


def save_audited_records(
    records: list[AuditedIssueRecord],
    output_dir: str | Path = "outputs/results/audited",
    parquet_path: str | Path = "data/processed/audited_labels.parquet",
) -> None:
    """Saves audited issue records to disk as JSON and Parquet while preserving data integrity."""
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    json_path = out_path / "audited_records.json"
    parq_file = Path(parquet_path)
    parq_file.parent.mkdir(parents=True, exist_ok=True)

    records_dict = [r.model_dump() for r in records]
    if not records_dict:
        raise ValueError("Refusing to write an empty audit result over existing artifacts.")

    # Merge with existing if present
    if json_path.exists():
        try:
            with open(json_path, "r", encoding="utf-8") as f:
                existing = json.load(f)
                if isinstance(existing, list):
                    # Index by composite key
                    merged_map = { (r["source"], r["project"], str(r["issue_id"])): r for r in existing }
                    for r in records_dict:
                        merged_map[(r["source"], r["project"], str(r["issue_id"]))] = r
                    records_dict = list(merged_map.values())
        except Exception as e:
            print(f"[Warning] Failed to merge with existing records: {e}")

    df = pd.DataFrame(records_dict)
    temp_json = json_path.with_suffix(".json.tmp")
    temp_parquet = parq_file.with_suffix(parq_file.suffix + ".tmp")
    temp_json.write_text(json.dumps(records_dict, indent=2), encoding="utf-8")
    df.to_parquet(temp_parquet, index=False)
    temp_json.replace(json_path)
    temp_parquet.replace(parq_file)

    print(f"--> Saved {len(records_dict)} total audited records to:")
    print(f"    JSON:    {json_path}")
    print(f"    Parquet: {parq_file}")

