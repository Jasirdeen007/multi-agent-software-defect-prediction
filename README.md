# AgentTriage

Multi-Agent Knowledge-Guided Software Defect Prediction with Noise-Resilient Label Auditing.

## Current phase
Phase I — BugHub data preparation and EDA.

Current canonical supervised dataset: 926,461 labelled issue reports after removing records with model text shorter than 20 characters.

Labels are the original BugHub source labels (`bug`, `non-bug`). They are not treated as human-verified gold labels.

## Repository structure
- `sql/` — PostgreSQL inspection, extraction, validation and canonical-dataset SQL.
- `src/preprocessing/` — reusable Python preprocessing/validation scripts.
- `src/eda/` — Parquet-based EDA pipeline and report generation.
- `data/raw/` — local raw data/database exports; ignored by Git.
- `data/processed/` — processed project data. The canonical Parquet dataset is intentionally kept in the repo.
- `outputs/` — generated EDA tables and figures.
- `notebooks/` — exploratory notebooks only; reusable logic belongs in `src/`.
- `docs/` — methodology and dataset notes.
- `configs/` — local configuration templates.
- `tests/` — data-quality and preprocessing tests.

## Data provenance
Source: BugHub PostgreSQL dataset, using the supplied BugHub labelled views:
- `issues.vw_bugzilla_data`
- `issues.vw_github_data`
- `issues.vw_jira_data`

Issue identity: `(source, project, issue_id)`.

Do not commit the full BugHub SQL dump or sensitive/local database credentials.
