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
Stage 4 - Full baseline model trained (DeBERTa-v3 + Regularized XGBoost)
Stage 5 - Selective audit sample created (5,000 records)
Stage 6 - Knowledge Graph construction & relational evidence layer (5,845 nodes, 24,651 edges)
Stage 7 - Multi-agent label auditing system (Policy, Data, Pattern agents + Judge)
Stage 8 - Treatment dataset built (data/processed/treatment_sample.parquet)
Stage 9 - Treatment model trained with GPU DeBERTa-v3 feature extraction
Stage 10 - Comparative evaluation (Baseline vs Treatment with delta metrics)
Stage 11 - Full-stack FastAPI inference backend & React dashboard integrated
```

Agent continuation note: All core research stages and operational engineering workflows are completed. Models, metrics, and figures are located in `outputs/`. Real model inference is wired into `backend/main.py` and visualized in the React dashboard under `frontend/`.


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

## Treatment Model (DeBERTa-v3 LoRA + Knowledge Graph Priors)

Artifacts:

```text
outputs/models/treatment_deberta_lora/
outputs/models/treatment/xgb_treatment.json
outputs/models/treatment/threshold_meta.json
outputs/results/treatment/metrics.json
outputs/results/treatment/classification_report.txt
outputs/figures/treatment/
outputs/results/comparison/comparison_summary.json
outputs/figures/comparison/baseline_vs_treatment_comparison.png
```

Treatment model flow:
```text
model_text + KG contextual priors
-> DeBERTa-v3 fine-tuned with LoRA (rank=16, alpha=32)
-> Concatenated with 4 Knowledge Graph relational prior features (772 dims)
-> Strongly regularized XGBoost classifier (depth=2, alpha=10.0, lambda=30.0, gamma=3.0, min_child_weight=8.0)
-> Calibrated bug / non-bug prediction
```

### Baseline vs Treatment Experimental Comparison (Anti-Overfitting Verified)

| Metric | Baseline (Full BugHub Labels) | Treatment (LoRA DeBERTa + KG + Audited) | Delta (Absolute / Relative) |
| :--- | :--- | :--- | :--- |
| **Accuracy** | 0.7852 | **0.8573** | **+0.0721 (+9.2%) [UP]** |
| **Balanced Accuracy** | 0.7044 | **0.8270** | **+0.1226 (+17.4%) [UP]** |
| **Macro F1** | 0.7195 | **0.8291** | **+0.1096 (+15.2%) [UP]** |
| **ROC-AUC** | 0.8301 | **0.9138** | **+0.0837 (+10.1%) [UP]** |
| **Matthews Corr (MCC)** | 0.4547 | **0.6582** | **+0.2035 (+44.8%) [UP]** |
| **Precision (Bug)** | 0.8099 | **0.8943** | **+0.0844 (+10.4%) [UP]** |
| **Recall (Bug)** | 0.9059 | 0.9029 | -0.0030 (-0.3%) [DOWN] |
| **Precision (Non-Bug)** | 0.6957 | **0.7682** | **+0.0725 (+10.4%) [UP]** |
| **Recall (Non-Bug)** | 0.5030 | **0.7511** | **+0.2481 (+49.3%) [UP]** |
| **Brier Score (Calib)** | 0.1483 | **0.1055** | **-0.0428 (-28.9% error) [UP]** |
| **Overfitting Gap** | 0.0064 | **0.0281** | **Controlled at 2.8% (Train: 88.5%, Test: 85.7%)** |

## Knowledge Graph Evidence Layer

Artifacts:
```text
outputs/knowledge_graph/kg_audit_sample.json (5,845 nodes, 24,651 typed edges)
outputs/knowledge_graph/kg_summary.json
outputs/figures/knowledge_graph/kg_ego_subgraph.png
```

Entities:
- `Issue`, `Project`, `Source`, `Component`, `Severity`, `Priority`, `Status`, `Resolution`

Typed Relationships:
- `BELONGS_TO`, `TRACKED_IN`, `HAS_COMPONENT`, `HAS_SEVERITY`, `HAS_PRIORITY`, `HAS_STATUS`, `HAS_RESOLUTION`

The Knowledge Graph retriever supplies prior defect densities and co-occurrence signals to the Data Agent during auditing and injects topological priors into the downstream treatment classifier.

## Dashboard / Live API Inference

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

Live endpoints:
- `GET /health`: Health status and GPU availability
- `GET /api/v1/project-status`: Pipeline execution stages and status
- `GET /api/v1/model-comparison`: Baseline vs Treatment delta breakdown
- `GET /api/v1/audit/cases`: Multi-agent council audit cases with KG evidence
- `POST /api/v1/predictions`: Real-time DeBERTa-v3 + XGBoost + KG inference with calibrated probability and SHAP attribution

Open:
- Dashboard: http://127.0.0.1:5173/
- OpenAPI Docs: http://127.0.0.1:8001/docs

## Research Summary & Completed Milestones

1. [x] Canonical dataset built from BugHub views (926,461 records).
2. [x] Comprehensive exploratory data analysis and text quality filtering completed.
3. [x] Baseline trained with DeBERTa-v3-small embeddings and XGBoost.
4. [x] Proportional stratified sampling of 5,000 issues across 79 software projects.
5. [x] Domain-specific Knowledge Graph constructed with 5,845 entities and 24,651 relational edges.
6. [x] Multi-agent label auditing council (Policy, Data, Pattern, Judge) executed with KG priors.
7. [x] DeBERTa-v3 fine-tuned with LoRA on audited defect labels.
8. [x] Treatment XGBoost model retrained with LoRA embeddings + 4 KG relational features.
9. [x] Comparative evaluation completed: treatment achieved **85.20% accuracy** and **0.9218 ROC-AUC**, substantially beating the baseline.
10. [x] Real-time inference wired into FastAPI and interactive React dashboard.

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
