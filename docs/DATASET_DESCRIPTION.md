# Dataset Description

## Project Context

This repository supports a Multi-Agent Software Defect Prediction (SDP) research project. The central comparison is:

```text
Original BugHub source labels
        vs.
Multi-agent audited labels
```

The current implementation starts from the prepared canonical Parquet dataset and implements the architecture through Stage 2: canonical input plus preprocessing validation and EDA.

## Source Dataset

The source data comes from BugHub issue-report data collected from:

- Bugzilla
- GitHub
- JIRA

The project dataset is not the complete BugHub population. It is the processed canonical working dataset prepared for this project.

## Canonical Artifact

Current project data artifact:

```text
data/processed/bughubs_canonical.parquet
```

The Parquet file was converted from:

```text
data/processed/bughubs_canonical.csv
```

The Parquet artifact is the Stage 1 input described by the end-to-end architecture document.

## Dataset Size

Current validated size:

```text
Records: 926,461
Columns: 15
```

Class distribution:

```text
bug      648,841
non-bug  277,620
```

Approximate percentages:

```text
bug      70.03%
non-bug  29.97%
```

## Canonical Identity

Issue IDs are source/project scoped. The canonical issue identity is:

```text
(source, project, issue_id)
```

Do not use `issue_id` alone as a global unique identifier.

## Canonical Fields

| Field | Meaning |
|---|---|
| `issue_id` | Issue identifier within its source/project |
| `source` | BugHub source system: Bugzilla, GitHub, or JIRA |
| `project` | Software project |
| `title` | Normalized issue title |
| `description` | Normalized issue body/description |
| `component` | Component when available |
| `severity` | Severity when available |
| `priority` | Priority when available |
| `status` | Issue status |
| `resolution` | Resolution when available |
| `original_label` | Original BugHub source label |
| `label_source` | Label provenance |
| `created_at` | Issue creation timestamp |
| `updated_at` | Last update timestamp |
| `model_text` | Model input text formed from title and description |

Missing metadata is expected for some sources. For example, GitHub records may not provide fields such as severity, priority, component, or resolution.

## Label Provenance

`original_label` contains the BugHub source label. It is the baseline label used before multi-agent auditing.

These labels must not be described as human gold labels or human-reviewed ground truth. Future agent or human decisions must be stored in separate fields and must not overwrite `original_label`.

## Model Text

The current model input is:

```text
model_text = title + description
```

No metadata fields, labels, comments, or agent outputs are appended to `model_text` in the current dataset.

## Stage 2 Validation

Stage 2 preprocessing validation is implemented in:

```text
src/preprocessing/validate_canonical_parquet.py
```

It checks:

- required schema columns
- label values
- composite identity duplicates
- missing identity values
- timestamp parseability
- blank or very short `model_text`

Run:

```bash
python src/preprocessing/validate_canonical_parquet.py
```

The validation report is written to:

```text
outputs/tables/stage2_preprocessing_validation.json
```

## EDA Process

EDA is generated directly from the Parquet artifact:

```bash
python src/eda/run_eda_pipeline.py
```

Outputs:

```text
outputs/tables/
outputs/figures/
outputs/reports/EDA_Report.html
```

The EDA process does not modify the dataset and does not create audited labels. Agent-audited labels belong to later architecture stages.
