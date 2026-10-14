# AgentTriage / Multi-Agent-SDP

Multi-Agent Knowledge-Guided Software Defect Prediction with label auditing.

The core experiment compares:

```text
Original BugHub source labels
vs.
Multi-agent audited labels
```

Original labels are the BugHub source labels. They are not human gold labels.

## Current Status

Completed:

```text
Stage 1 - Canonical Parquet dataset (926,461 records)
Stage 2 - Data validation and EDA
Stage 3 - Dataset splitting strategy finalized
Stage 4 - Full baseline model trained on Kaggle GPU
Stage 5 - Selective audit sample created (5,000 records)
```

Current continuation point:

```text
Stage 6 - Knowledge Graph construction
Stage 7 - Multi-agent label auditing system (Policy/Data/Pattern agents + Judge)
Stage 10/11 - Agent-audited dataset + treatment model training
Stage 12 - Baseline vs Treatment comparison
```

Agent continuation note: The baseline model has been successfully trained on the full canonical dataset (926,321 records after filtering) using Kaggle GPU. Model outputs are stored in `outputs/models/baseline/full/` and `outputs/results/baseline/full/`. A stratified audit sample of 5,000 records has been created at `data/processed/audit_sample.parquet` using proportional sampling by source and original_label. The next critical tasks are Knowledge Graph construction and implementing the multi-agent label auditing system.

## Key Data Artifacts

Canonical dataset:

```text
data/processed/bughubs_canonical.parquet
```

Current canonical size:

```text
926,461 records
bug      648,841
non-bug  277,620
```

Full baseline training used:

```text
926,321 records
```

because 140 very short/empty `model_text` rows were filtered by the baseline loader.

Selective audit sample:

```text
data/processed/audit_sample.parquet
outputs/results/audit_sampling/audit_sample_summary.json
outputs/results/audit_sampling/audit_sample_ids.csv
```

Audit sample details:

```text
sample size: 5,000
random seed: 42
method: proportional stratified sampling by source + original_label
duplicate composite identities: 0
projects covered: 79 / 79
```

## Baseline Model

Full baseline artifacts:

```text
outputs/models/baseline/full/xgb_baseline.json
outputs/models/baseline/full/threshold_meta.json
outputs/results/baseline/full/metrics.json
outputs/results/baseline/full/classification_report.txt
outputs/results/baseline/full/splits_summary.json
outputs/figures/baseline/full/
```

Baseline model flow:

```text
model_text
-> DeBERTa-v3-small frozen embedding
-> XGBoost classifier
-> bug / non-bug prediction
```

Baseline metrics from full run:

```text
accuracy: 0.7852
balanced_accuracy: 0.7044
macro_f1: 0.7195
roc_auc: 0.8301
mcc: 0.4547
brier_score: 0.1483
optimal_threshold: 0.52
```

Do not confuse this with older sample/tuned artifacts. Old baseline root/smoke/tuned outputs were cleared; the relevant baseline is under `outputs/models/baseline/full/`.

## Agent Audit Configuration

Agent council config:

```text
configs/agents_config.yaml
```

It currently points to:

```yaml
audit:
  dataset_path: "data/processed/audit_sample.parquet"
  sample_size: 5000
  output_dir: "outputs/results/audited"
  output_parquet: "data/processed/audited_labels.parquet"
```

Old pilot audited outputs were cleared. Fresh agent audit outputs should be written to:

```text
outputs/results/audited/
data/processed/audited_labels.parquet
```

## Important Invariants

Preserve:

```text
issue identity = (source, project, issue_id)
original label field = original_label
model input = model_text = title + description
labels = bug / non-bug
```

Do not:

```text
overwrite original_label
modify raw BugHub tables
call original labels human gold labels
train baseline with audited labels
mix baseline and treatment labels
```

Future audited labels must be stored separately, for example:

```text
policy_label
data_label
pattern_label
judge_label
judge_confidence
label_flipped
```

## Run Commands

Validate canonical Parquet:

```powershell
python src/preprocessing/validate_canonical_parquet.py
```

Generate EDA:

```powershell
python src/eda/run_eda_pipeline.py
```

Recreate selective audit sample:

```powershell
python -m src.audit_sampling.create_audit_sample
```

Run a small agent audit pilot:

```powershell
python -m src.agents.orchestrator --config configs/agents_config.yaml --sample-size 10 --batch-size 2
```

Only run the full audit after the pilot output schema and quality are validated:

```powershell
python -m src.agents.orchestrator --config configs/agents_config.yaml --sample-size 5000 --batch-size 5
```

## Dashboard / API Shell

Backend:

```powershell
python -m uvicorn backend.main:app --host 127.0.0.1 --port 8001
```

Frontend:

```powershell
cd frontend
$env:VITE_API_BASE_URL="http://127.0.0.1:8001"
npm run dev -- --host 127.0.0.1
```

Open:

```text
http://127.0.0.1:5173/
http://127.0.0.1:8001/docs
```

The API/dashboard currently provide the finished-product flow and preview inference contract. Real model inference still needs to be wired to the baseline model artifacts.

## Next Steps

1. Confirm LLM/API credentials for the agent council.
2. Run `sample-size 10` audit pilot.
3. Validate `data/processed/audited_labels.parquet` schema.
4. Run full 5,000-record audit.
5. Build treatment dataset from canonical records plus Judge labels.
6. Train audited-label treatment model using the same split/protocol.
7. Compare baseline vs treatment metrics.
8. Wire real model inference into FastAPI.

## Data Provenance

Source: BugHub PostgreSQL dataset using supplied labelled views:

```text
issues.vw_bugzilla_data
issues.vw_github_data
issues.vw_jira_data
```

The canonical SQL copies:

```text
type -> original_label
label_source = "BugHub source view"
```

The project dataset is the processed canonical working dataset, not the complete BugHub population.

Do not commit secrets, `.env`, API keys, raw BugHub dumps, large embeddings, or temporary Kaggle output folders.
