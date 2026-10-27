from __future__ import annotations

import pandas as pd
import pytest

from src.audit_sampling.create_audit_sample import create_sample
from src.baseline.data_splitter import load_or_create_fixed_splits, validate_split_disjointness
from src.treatment.build_treatment_dataset import build_treatment_dataset


def tiny_dataset() -> pd.DataFrame:
    rows = []
    for project in ("p1", "p2", "p3"):
        for label in ("bug", "non-bug"):
            for i in range(8):
                rows.append({
                    "issue_id": f"{project}-{label}-{i}", "source": "GH", "project": project,
                    "title": "title", "description": "description", "component": "core",
                    "severity": None, "priority": None, "status": "open", "resolution": None,
                    "original_label": label, "label_source": "test", "created_at": None,
                    "updated_at": None, "model_text": "a sufficiently long issue description",
                })
    return pd.DataFrame(rows)


def test_split_files_are_reused_and_disjoint(tmp_path):
    source = tmp_path / "all.parquet"
    tiny_dataset().to_parquet(source)
    first = load_or_create_fixed_splits(source, tmp_path / "splits", test_size=.25, val_size=.25)
    second = load_or_create_fixed_splits(source, tmp_path / "splits", test_size=.25, val_size=.25)
    validate_split_disjointness(*first)
    assert [x["issue_id"].tolist() for x in first] == [x["issue_id"].tolist() for x in second]
    with pytest.raises(ValueError, match="differs from the frozen split manifest"):
        load_or_create_fixed_splits(source, tmp_path / "splits", random_seed=7, test_size=.25, val_size=.25)


def test_sampling_is_reproducible_and_project_stratified():
    df = tiny_dataset()
    a = create_sample(df, 12, 17)
    b = create_sample(df, 12, 17)
    assert a[["source", "project", "issue_id"]].equals(b[["source", "project", "issue_id"]])
    assert set(a.project) == set(df.project)
    assert set(a.original_label) == {"bug", "non-bug"}


def test_audit_must_be_inside_training_and_keeps_original_label(tmp_path):
    import json
    train = tiny_dataset().iloc[:12].copy()
    train_path = tmp_path / "train.parquet"
    train.to_parquet(train_path)

    sample = train.iloc[:1].copy()
    sample_path = tmp_path / "audit_sample.parquet"
    sample.to_parquet(sample_path)

    record = {
        "source": train.iloc[0].source, "project": train.iloc[0].project,
        "issue_id": train.iloc[0].issue_id, "original_label": train.iloc[0].original_label,
        "policy_label": "non-bug", "data_label": "non-bug", "pattern_label": "non-bug",
        "judge_label": "non-bug", "judge_confidence": .8, "agent_agreement": 1.0, "label_flipped": True,
    }
    audit_json = tmp_path / "audited_records.json"
    audit_json.write_text(json.dumps([record]))

    treatment_path = tmp_path / "treatment.parquet"
    summary_path = tmp_path / "summary.json"
    build_treatment_dataset(
        train_path=train_path,
        audit_sample_path=sample_path,
        audited_json_path=audit_json,
        output_path=treatment_path,
        summary_path=summary_path,
        overwrite=True,
    )
    output = pd.read_parquet(treatment_path)
    assert len(output) == len(train)
    assert output.iloc[0].original_label == train.iloc[0].original_label
    assert output.iloc[0].treatment_label == "non-bug"

    bad_record = dict(record)
    bad_record["issue_id"] = "not-in-training"
    bad_json = tmp_path / "bad.json"
    bad_json.write_text(json.dumps([bad_record]))
    with pytest.raises(ValueError, match="outside"):
        build_treatment_dataset(
            train_path=train_path,
            audit_sample_path=sample_path,
            audited_json_path=bad_json,
            output_path=tmp_path / "bad.parquet",
            summary_path=tmp_path / "bad_summary.json",
            overwrite=True,
        )
