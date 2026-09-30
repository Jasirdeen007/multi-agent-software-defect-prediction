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
    
    # Save full JSON audit records
    json_path = out_path / "audited_records.json"
    records_dict = [r.model_dump() for r in records]
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(records_dict, f, indent=2)

    # Convert to DataFrame and save to Parquet
    df = pd.DataFrame(records_dict)
    parq_file = Path(parquet_path)
    parq_file.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(parq_file, index=False)
    
    print(f"--> Saved {len(records)} audited records to:")
    print(f"    JSON:    {json_path}")
    print(f"    Parquet: {parq_file}")
