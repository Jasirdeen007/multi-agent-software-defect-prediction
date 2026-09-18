# AGENTS.md

# Multi-Agent-SDP — Coding Agent Instructions

> **Purpose:** This file is the operational context for any coding agent working on this repository. Read it before making changes. It is designed so an agent can continue the project from any stage without requiring the project history to be re-explained.

---

## 1. Project Identity

**Repository:** `Multi-Agent-SDP`

**Project:** Multi-Agent Software Defect Prediction

**Type:** Final-year research project

**Domain:**
- Software Engineering
- Software Defect Prediction (SDP)
- NLP
- Multi-Agent Systems
- Knowledge Graphs
- Explainable AI

**Primary technologies:**
- Python
- PostgreSQL
- DeBERTa-v3
- XGBoost
- SHAP
- Knowledge Graph technologies
- LLM/agent orchestration

### Core research objective

Investigate whether **multi-agent auditing of software issue labels** can improve the usefulness/reliability of labels used for Software Defect Prediction compared with directly using the original BugHub source labels.

The project is not simply an NLP classifier. The central experimental component is the comparison between:

```text
Original BugHub Source Labels
              VS
Multi-Agent Audited Labels
```

---

# 2. Golden Rules for Coding Agents

Before changing anything:

1. Read this file completely.
2. Inspect the current repository state.
3. Inspect the relevant PostgreSQL schema/table when database work is involved.
4. Check existing code before creating new files.
5. Preserve completed work.
6. Do not redesign the architecture unless explicitly requested.
7. Do not invent data, labels, fields, statistics, relationships, or research results.
8. Do not overwrite original/source labels.
9. Do not modify raw BugHub data.
10. Keep baseline and treatment experiments separate.
11. Prefer small, testable changes.
12. Run relevant validation after modifications.
13. Update documentation when a methodological or schema change is made.
14. Never hardcode credentials, API keys, or passwords.
15. Do not add dependencies unless they are actually required.
16. Avoid loading the full ~926k-row dataset into memory when SQL aggregation can perform the task.
17. Preserve reproducibility.
18. If an assumption is necessary, explicitly document it instead of silently introducing it.

---

# 3. Current Project Status

## Phase I — Data Preparation + EDA

Completed:

```text
BugHub dataset imported into PostgreSQL
        ↓
BugHub labelled views identified
        ↓
Original/source labels extracted
        ↓
Canonical dataset created
        ↓
Source-specific fields normalized
        ↓
Composite issue identity verified
        ↓
model_text created
        ↓
Data quality checked
        ↓
Very-short records removed
        ↓
Canonical dataset finalized
        ↓
EDA implementation
```

Current working phase:

```text
PHASE I — DATA PREPARATION + EDA
```

Next major phase:

```text
Knowledge Graph Construction
```

Then:

```text
Multi-Agent Label Auditing
```

Then:

```text
DeBERTa-v3 + XGBoost + SHAP
```

---

# 4. Current Dataset Facts

Current canonical working dataset:

```text
926,461 records
```

Labels:

```text
BUG       = 648,841
NON-BUG   = 277,620
```

Approximate class distribution:

```text
BUG       = 70.03%
NON-BUG   = 29.97%
```

Records removed during final text-quality filtering:

```text
1,276
```

Removal criterion:

```sql
model_text IS NULL
OR length(trim(model_text)) < 20
```

These numbers are current project facts.

If a script reports different numbers:

- Do not immediately modify the documentation.
- Investigate the database state.
- Check whether filters changed.
- Check whether the canonical table was rebuilt.
- Check whether the query is using the correct schema/table.
- Report the discrepancy.

---

# 5. Dataset Scope

The project uses the BugHub issue-report dataset as the source dataset.

BugHub contains reports from:

```text
Bugzilla
GitHub
JIRA
```

The complete BugHub population is larger than the project's current processed dataset.

Therefore:

```text
926,461
```

must NOT be described as the complete BugHub dataset.

Use:

```text
processed canonical dataset
canonical working dataset
project's BugHub-labelled population
```

when referring to the current dataset.

---

# 6. PostgreSQL Architecture

Database:

```text
bughub
```

Original BugHub objects:

```text
issues.bz_unfiltered_clean
issues.gh_unfiltered_clean
issues.reports_clean

issues.vw_bugzilla_data
issues.vw_github_data
issues.vw_jira_data
```

Project-created schema:

```text
agenttriage
```

Project-created tables:

```text
agenttriage.canonical_issues
agenttriage.canonical_dataset
```

### Current source of truth

For EDA and downstream project processing:

```text
agenttriage.canonical_dataset
```

EDA should query this table.

Do not make downstream processing depend directly on raw BugHub tables unless the task specifically requires reconstruction or validation against the source.

---

# 7. Database Configuration

Local `.env`:

```env
DB_HOST=localhost
DB_PORT=5432
DB_NAME=bughub
DB_USER=postgres
DB_PASSWORD=YOUR_POSTGRES_PASSWORD
```

Important:

```text
bughub       = PostgreSQL database
agenttriage  = PostgreSQL schema
```

Therefore:

```text
DB_NAME=bughub
```

is correct.

The canonical table is:

```text
agenttriage.canonical_dataset
```

Never hardcode credentials.

Never commit:

```text
.env
passwords
API keys
tokens
secrets
```

---

# 8. Canonical Dataset Identity

Issue IDs are not globally unique.

The canonical issue identity is:

```text
(source, project, issue_id)
```

Never assume:

```text
issue_id
```

alone uniquely identifies an issue.

All joins, deduplication, validation, and entity mapping must respect:

```text
source + project + issue_id
```

A uniqueness constraint exists on the composite identity.

---

# 9. Canonical Dataset Schema

Current important fields:

```text
issue_id
source
project
title
description
component
severity
priority
status
resolution
original_label
label_source
created_at
updated_at
model_text
```

Definitions:

| Field | Meaning |
|---|---|
| `issue_id` | Issue identifier within source/project |
| `source` | Bugzilla, GitHub or JIRA |
| `project` | Software project |
| `title` | Normalized issue title |
| `description` | Normalized issue description/body |
| `component` | Component where available |
| `severity` | Severity where available |
| `priority` | Priority where available |
| `status` | Issue status |
| `resolution` | Resolution where available |
| `original_label` | BugHub source label |
| `label_source` | Label provenance |
| `created_at` | Issue creation timestamp |
| `updated_at` | Last update timestamp |
| `model_text` | Combined title + description |

Do not assume every source has every field.

---

# 10. Source-Specific Normalization

## Bugzilla

Canonical mapping:

```text
summary          → title
description      → description
component        → component
severity         → severity
priority         → priority
status           → status
resolution       → resolution
creation_time    → created_at
last_change_time → updated_at
```

## GitHub

Canonical mapping:

```text
title      → title
body       → description
state      → status
created_at → created_at
updated_at → updated_at
```

GitHub does not necessarily provide:

```text
component
severity
priority
resolution
```

NULL values are expected.

## JIRA

Canonical mapping:

```text
fields.summary             → title
fields.description         → description
fields.components[0].name  → component
fields.priority.name       → priority
fields.status.name         → status
fields.resolution.name     → resolution
fields.created             → created_at
fields.updated             → updated_at
```

Do not silently invent missing values.

---

# 11. Modelling Text

The current common NLP input is:

```text
model_text = title + description
```

This provides a common textual representation across Bugzilla, GitHub and JIRA.

Do not silently append:

```text
severity
priority
status
resolution
component
labels
comments
```

to `model_text`.

Changing the modelling input is a methodological change and must be explicitly documented.

---

# 12. Label Provenance

The field:

```text
original_label
```

contains the BugHub source label.

Use:

```text
BugHub source label
Original BugHub label
Original/source label
Source-labelled data
```

Do NOT call the original labels:

```text
gold labels
human-reviewed labels
human ground truth
human gold labels
```

unless a separate human-review experiment actually establishes that property.

The original labels are the starting labels for the research experiment.

---

# 13. Original Label Logic

The original labels are obtained from BugHub source-specific views:

```text
issues.vw_bugzilla_data
issues.vw_github_data
issues.vw_jira_data
```

Do not replace BugHub's source-specific label mechanism with an invented universal heuristic.

In particular, do not introduce a rule such as:

```text
type = defect → BUG
```

without explicit methodological justification.

The original/source label must remain available throughout the project.

---

# 14. Label Preservation

Never overwrite:

```text
original_label
```

with:

```text
agent_label
judge_label
human_label
prediction
```

Future outputs should use separate fields.

Potential structure:

```text
original_label

policy_label
policy_confidence

data_label
data_confidence

pattern_label
pattern_confidence

judge_label
judge_confidence

human_label
```

The exact database schema should be finalized when that phase is implemented.

---

# 15. Intended End-to-End Architecture

```text
BugHub Source Data
        ↓
Canonical Dataset
        ↓
Knowledge Graph
        ↓
Policy Agent
Data Agent
Pattern Agent
        ↓
Judge
        ↓
Agent-Audited Labels
        ↓
DeBERTa-v3
        ↓
Embeddings
        ↓
XGBoost
        ↓
Prediction
        ↓
SHAP
```

---

# 16. Baseline

The baseline must use the original BugHub source labels:

```text
BugHub Source Labels
        ↓
DeBERTa-v3
        ↓
Embeddings
        ↓
XGBoost
        ↓
Prediction
        ↓
SHAP
```

Never train the baseline using agent-audited labels.

---

# 17. Treatment

The treatment uses labels after multi-agent auditing:

```text
BugHub Source Labels
        ↓
Knowledge Graph
        ↓
Policy Agent
Data Agent
Pattern Agent
        ↓
Judge
        ↓
Agent-Audited Labels
        ↓
DeBERTa-v3
        ↓
Embeddings
        ↓
XGBoost
        ↓
Prediction
        ↓
SHAP
```

The core experimental comparison is:

```text
Baseline:
Original/source labels

VS

Treatment:
Agent-audited labels
```

---

# 18. Multi-Agent System

Planned agents:

```text
Policy Agent
Data Agent
Pattern Agent
Judge
```

Agents must have clearly separated responsibilities.

---

## 18.1 Policy Agent

Purpose:

Evaluate an issue according to explicit labelling policies.

Potential inputs:

```text
issue text
metadata
Knowledge Graph evidence
labelling policy
```

Expected structured output:

```json
{
  "agent": "policy_agent",
  "label": "bug",
  "confidence": 0.91,
  "reason": "Issue describes an observed software malfunction.",
  "evidence": []
}
```

Do not claim the confidence is calibrated unless calibration is actually evaluated.

---

## 18.2 Data Agent

Purpose:

Inspect structured issue metadata and consistency.

Potential evidence:

```text
source
project
status
resolution
priority
severity
component
labels
timestamps
relationships
```

Example:

```json
{
  "agent": "data_agent",
  "label": "bug",
  "confidence": 0.86,
  "reason": "Available metadata provides defect-related evidence.",
  "evidence": []
}
```

Do not treat a single metadata field as automatic proof of the final label.

---

## 18.3 Pattern Agent

Purpose:

Use historical or retrieved issue patterns.

Potential evidence:

```text
similar issues
project patterns
text similarity
historical labels
Knowledge Graph relationships
```

Example:

```json
{
  "agent": "pattern_agent",
  "label": "bug",
  "confidence": 0.89,
  "reason": "The issue resembles historically labelled defect reports.",
  "evidence": []
}
```

Similarity is evidence, not proof.

---

# 19. Judge

The Judge combines the agent outputs.

Conceptually:

```text
Policy Agent
      \
Data Agent ----> Judge ----> Agent-Audited Label
      /
Pattern Agent
```

The Judge should preserve the individual agent decisions.

Example:

```json
{
  "final_label": "bug",
  "confidence": 0.90,
  "agreement": 0.67,
  "agent_outputs": {},
  "reasoning_summary": "",
  "evidence": []
}
```

Do not store only the final label if individual agent outputs are available.

---

# 20. Agent Confidence

Each agent should have an independent confidence value where meaningful.

Example:

```text
Policy Agent  = 0.91
Data Agent    = 0.86
Pattern Agent = 0.89
Judge         = 0.90
```

These values must not automatically be treated as statistically equivalent.

The implementation should document:

- confidence definition
- confidence generation
- calibration status
- disagreement representation
- Judge aggregation method

Do not call a confidence score "calibrated" without an actual calibration evaluation.

---

# 21. Agent Disagreement

Disagreement is valuable research information.

Example:

```text
Policy Agent  → BUG       0.91
Data Agent    → NON-BUG   0.71
Pattern Agent → BUG       0.84
Judge         → BUG       0.79
```

Preserve disagreement when possible.

Useful fields include:

```text
policy_label
policy_confidence
data_label
data_confidence
pattern_label
pattern_confidence
judge_label
judge_confidence
```

This enables later analysis of:

```text
agent agreement
agent disagreement
confidence
error patterns
agent reliability
Judge behaviour
```

---

# 22. Human-in-the-Loop

The planned production/review flow is:

```text
New Issue
    ↓
Trained SDP Model
    ↓
Prediction + Confidence + SHAP
    ↓
Confidence Threshold
    |
    +---- High confidence ----> Recommendation
    |
    +---- Low confidence -----> Human Tester Review
                                      ↓
                               Final Human Decision
```

Human feedback should be logged separately.

Do not automatically overwrite the training dataset after every human decision.

Future periodic retraining from approved human feedback can be implemented as a
separate stage.

---

# 23. Human Review

If human review is implemented, preserve:

```text
human_label
human_reviewer
review_timestamp
review_reason
```

Do not overwrite:

```text
original_label
```

with:

```text
human_label
```

The source label must remain available for baseline comparison.

---

# 24. Knowledge Graph

Knowledge Graph construction is the next major phase after EDA.

Potential entities include:

```text
Issue
Project
Component
Developer
Label
Status
Resolution
Severity
Priority
Related Issue
```

Potential relationships include:

```text
BELONGS_TO
HAS_COMPONENT
HAS_STATUS
HAS_PRIORITY
HAS_SEVERITY
HAS_RESOLUTION
DUPLICATES
BLOCKS
DEPENDS_ON
RELATED_TO
```

Do not create graph nodes or edges merely because a database field exists.

Each relationship should have a clear purpose for:

- agent reasoning
- evidence retrieval
- issue similarity
- label auditing
- project context
- explainability

The Knowledge Graph should support the multi-agent system rather than becoming an
unrelated component.

---

# 25. EDA

Current EDA files:

```text
src/eda/
└── run_eda_pipeline.py
```

EDA MUST use:

```text
data/processed/bughubs_canonical.parquet
```

---

## 25.1 Basic Overview

Analyze:

```text
total records
unique issues
number of sources
number of projects
BUG count
NON-BUG count
class percentages
```

---

## 25.2 Text Statistics

Analyze:

```text
mean text length
standard deviation
minimum
maximum
25th percentile
median
75th percentile
text-length distribution
```

Use:

```text
model_text
```

Do not unnecessarily load unrelated Parquet columns when column projection is sufficient.

---

## 25.3 Project Distribution

Analyze:

```text
issues per project
BUG count per project
NON-BUG count per project
BUG percentage per project
largest projects
```

Project distribution is important for later split design and leakage analysis.

---

## 25.4 Missingness

Check:

```text
title
description
component
severity
priority
status
resolution
created_at
updated_at
```

Source-specific missing values are expected.

Do not automatically treat every NULL as corruption.

---

## 25.5 Temporal Distribution

Analyze:

```text
issue creation by month
issue creation by year
BUG/NON-BUG distribution over time
```

Temporal structure may affect future train/test design.

---

# 26. EDA Report

The final EDA report should be generated from the canonical Parquet artifact.

Script:

```text
src/eda/run_eda_pipeline.py
```

Run:

```bash
python src/eda/run_eda_pipeline.py
```

Output:

```text
outputs/reports/EDA_Report.html
```

The report should contain:

```text
Dataset overview
Class distribution
Source distribution
Project distribution
Text statistics
Missingness
Temporal distribution
EDA findings
Label provenance
Data preparation decisions
Implications for SDP
Next phase
```

Do not hardcode changing dataset statistics into the final report generator.

---

# 27. Repository Structure

Expected structure:

```text
Multi-Agent-SDP/
│
├── AGENTS.md
├── README.md
├── requirements.txt
├── .env
├── .gitignore
│
├── src/
│   ├── __init__.py
│   │
│   ├── database/
│   │   ├── __init__.py
│   │   └── connection.py
│   │
│   ├── preprocessing/
│   │   ├── __init__.py
│   │   └── ...
│   │
│   └── eda/
│       ├── __init__.py
│       └── run_eda_pipeline.py
│
├── docs/
│   ├── DATASET_DESCRIPTION.md
│   └── EDA_README.md
│
├── outputs/
│   ├── figures/
│   ├── tables/
│   └── reports/
│       └── EDA_Report.html
│
└── tests/
    ├── __init__.py
    └── ...
```

Agents should preserve this organization unless there is a clear reason to change it.

---

# 28. Python Environment

Create a virtual environment:

```bash
python -m venv .venv
```

Windows activation:

```bash
.venv\Scripts\activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

# 29. Current EDA Dependencies

Current EDA dependencies:

```text
pandas
numpy
matplotlib
seaborn
sqlalchemy
psycopg2-binary
python-dotenv
```

Later phases may require:

```text
torch
transformers
scikit-learn
xgboost
shap
```

Do not add later-phase dependencies prematurely unless needed.

---

# 30. Database Connection Test

Run from repository root:

```bash
python -c "from src.database.connection import get_engine; print(get_engine().connect())"
```

Then verify:

```sql
SELECT COUNT(*)
FROM agenttriage.canonical_dataset;
```

Expected current count:

```text
926461
```

Verify labels:

```sql
SELECT
    original_label,
    COUNT(*)
FROM agenttriage.canonical_dataset
GROUP BY original_label;
```

Expected current counts:

```text
bug       | 648841
non-bug   | 277620
```

---

# 31. Running EDA

Run from repository root:

```bash
python src/preprocessing/validate_canonical_parquet.py
python src/eda/run_eda_pipeline.py
```

Generated outputs:

```text
outputs/
├── figures/
├── tables/
└── reports/
    └── EDA_Report.html
```

---

# 32. SQL Guidelines

When writing SQL:

1. Fully qualify project tables where ambiguity is possible.
2. Use `data/processed/bughubs_canonical.parquet` for downstream EDA.
3. Respect `(source, project, issue_id)` identity.
4. Avoid destructive operations unless explicitly requested.
5. Prefer `SELECT` validation before `DELETE`, `UPDATE`, or `ALTER`.
6. Do not modify raw BugHub tables.
7. Use transactions for potentially destructive operations.
8. Verify row counts after transformations.
9. Check class distribution after filtering.
10. Check source/project distribution after major transformations.

---

# 33. Python Guidelines

Use:

```text
Python 3.x
```

Prefer:

```text
clear functions
small modules
type hints where useful
explicit error handling
reproducible scripts
```

Avoid:

```text
hardcoded credentials
hardcoded dataset statistics
unnecessary global state
loading the entire database into RAM
unused dependencies
duplicated database connection logic
```

For database access, reuse:

```text
src/database/connection.py
```

Do not create separate hardcoded database connection strings in every script.

---

# 34. Data Processing Rules

Whenever a transformation is added:

1. Explain why it is needed.
2. Record the transformation in code.
3. Record the resulting row count.
4. Compare class distribution before and after.
5. Compare source distribution before and after.
6. Check for duplicate `(source, project, issue_id)` values.
7. Update documentation if the transformation changes the research dataset.

Never silently drop records.

---

# 35. Train/Test Split Rules

When model training begins, carefully consider leakage.

Potential leakage sources:

```text
same issue
duplicate issue
same project
near-identical reports
historically linked reports
future information
agent-generated labels derived from test data
```

The exact split strategy must be documented before training.

Do not randomly split the data by default without considering:

```text
project leakage
temporal leakage
duplicate leakage
label-auditing leakage
```

If the research question requires a particular split strategy, preserve it consistently
across baseline and treatment.

---

# 36. Baseline/Treatment Experimental Integrity

When comparing baseline and treatment:

- Keep the model architecture comparable.
- Keep evaluation data comparable.
- Keep preprocessing comparable unless the experiment explicitly studies preprocessing.
- Change the intended independent variable only.
- Do not allow agent outputs generated from evaluation/test information to leak into training.
- Record experiment configuration.
- Store random seeds.
- Store model versions.
- Store dataset versions.

The comparison must be reproducible.

---

# 37. DeBERTa-v3

Planned text encoder:

```text
DeBERTa-v3
```

Expected conceptual flow:

```text
model_text
    ↓
Tokenizer
    ↓
DeBERTa-v3
    ↓
Contextual representation
    ↓
Embedding
    ↓
XGBoost
```

Do not switch to another encoder without explicitly changing the research configuration.

If the exact DeBERTa-v3 checkpoint is selected later, record its exact model identifier.

---

# 38. XGBoost

Planned classifier:

```text
XGBoost
```

Input:

```text
DeBERTa-v3 embeddings
```

Output:

```text
BUG / NON-BUG prediction
```

Model configuration must be versioned and reproducible.

Do not evaluate using accuracy alone.

---

# 39. Model Evaluation

The project has previously considered imbalance-aware and probabilistic metrics including:

```text
Accuracy
Balanced Accuracy
ROC-AUC
PR-AUC
MCC
Log Loss
Brier Score
```

Use metrics appropriate to the final experimental design.

Do not report a metric merely because it looks favorable.

Report the same primary metrics for baseline and treatment wherever comparison is intended.

---

# 40. SHAP

Planned explainability framework:

```text
SHAP
```

Purpose:

Explain XGBoost predictions.

SHAP outputs should be linked to the model input representation appropriately.

Do not claim that SHAP proves causal relationships.

Use language such as:

```text
feature contribution
model explanation
prediction attribution
```

rather than causal claims.

---

# 41. Knowledge Graph + Agents + Model Separation

Maintain a clear separation between:

```text
Data Layer
Knowledge Layer
Agent Layer
Label Layer
Model Layer
Explainability Layer
Human Review Layer
```

Conceptually:

```text
DATA
  ↓
KNOWLEDGE
  ↓
AGENTS
  ↓
AUDITED LABELS
  ↓
MODEL
  ↓
EXPLANATION
  ↓
HUMAN REVIEW
```

Do not mix these layers without documenting the reason.

---

# 42. Experiment Metadata

For every significant experiment, record:

```text
dataset version
dataset row count
label definition
train/test split
random seed
model name/version
hyperparameters
embedding configuration
agent configuration
Judge configuration
evaluation metrics
timestamp
```

This can later be implemented as:

```text
configs/
experiments/
results/
```

if required.

Do not introduce an experiment-tracking framework unless it provides actual value.

---

# 43. Logging

Agent systems should preserve enough information to reproduce decisions.

Recommended fields:

```text
issue identity
agent name
agent label
agent confidence
agent evidence
agent reasoning summary
Judge label
Judge confidence
Judge evidence
timestamp
model/version
prompt/policy version where appropriate
```

Do not store secrets or unnecessary sensitive information.

---

# 44. Prompt and Policy Versioning

If agents use prompts or policies:

- Store them in version-controlled files.
- Give them explicit version identifiers.
- Do not silently modify production prompts.
- Record which prompt/policy version generated an output.

Example:

```text
policies/
prompts/
```

Possible metadata:

```text
policy_version = v1.0
prompt_version = v1.0
```

---

# 45. Error Handling

Agents and data pipelines must fail explicitly.

Do not silently convert:

```text
database failure
missing required field
invalid JSON
invalid label
model failure
API failure
```

into a valid-looking result.

Prefer:

```text
raise an informative error
log the failure
mark the record appropriately
continue only when safe
```

---

# 46. Testing

Before committing significant changes:

- Test database connection.
- Test SQL queries.
- Test transformations on a small sample.
- Test expected schema.
- Test label values.
- Test uniqueness.
- Test null handling.
- Test agent output schema.
- Test model input/output shapes.

For destructive transformations, validate on a copy/sample first.

---

# 47. Required Invariants

These must remain true unless a documented methodology change explicitly changes them:

```text
Database = bughub

Working schema = agenttriage

Working table = agenttriage.canonical_dataset

Issue identity = (source, project, issue_id)

Labels = bug / non-bug

Original labels remain preserved

model_text = title + description

Original/source labels are not called human gold labels

Raw BugHub tables are not modified by downstream processing
```

---

# 48. Documentation Rules

Update documentation when changing:

```text
dataset schema
data filtering
label logic
model input
agent responsibilities
agent output schema
Knowledge Graph schema
train/test split
evaluation methodology
database structure
dependencies
```

Relevant documentation:

```text
README.md
docs/DATASET_DESCRIPTION.md
docs/EDA_README.md
AGENTS.md
```

If a change affects how future coding agents should work, update this file.

---

# 49. Git Rules

Use meaningful commits.

Examples:

```text
feat: add canonical dataset validation
feat: add project-level EDA
feat: add knowledge graph construction
feat: implement policy agent
feat: implement judge
feat: add DeBERTa embedding pipeline
feat: add XGBoost classifier

fix: handle missing JIRA descriptions
fix: correct composite issue identity
fix: resolve PostgreSQL connection handling

docs: update dataset description
docs: update experiment methodology
```

Do not commit:

```text
.env
passwords
API keys
large raw datasets
temporary files
generated cache files
virtual environments
```

---

# 50. Large Dataset Handling

The raw BugHub dump is very large.

Do not place the original multi-GB dump directly into normal Git history.

The canonical dataset is also large.

Preferred approach:

```text
Git repository
    ↓
Code + SQL + documentation
```

and:

```text
Large dataset artifact
    ↓
Shared storage / controlled release / appropriate large-file system
```

If the team needs a PostgreSQL dump of the canonical dataset, document:

```text
file name
version
creation date
row count
checksum
import command
```

Do not silently change the dataset artifact.

---

# 51. Safe Continuation Procedure

When an agent starts a new task, follow this order:

```text
1. Read AGENTS.md
        ↓
2. Inspect repository
        ↓
3. Identify current phase
        ↓
4. Inspect relevant existing code
        ↓
5. Inspect relevant database objects if needed
        ↓
6. Determine exactly what is missing
        ↓
7. Implement the smallest appropriate change
        ↓
8. Run validation/tests
        ↓
9. Check row counts/schema/results
        ↓
10. Update documentation if required
        ↓
11. Report exactly what changed
```

Do not rebuild completed stages unnecessarily.

---

# 52. If the Project Is Found at a Later Stage

If a future agent opens the repository and finds:

```text
Knowledge Graph code
Agent code
DeBERTa code
XGBoost code
SHAP code
production inference code
```

it should continue from the existing implementation.

Do not recreate earlier stages unless validation shows they are missing or incorrect.

The agent should inspect:

```text
Git history
README.md
docs/
src/
configs/
tests/
outputs/
database schema
```

before making assumptions.

---

# 53. If Database and Code Disagree

When database results disagree with documentation:

```text
Do not guess.
Do not overwrite the documentation immediately.
Do not modify the database immediately.
```

Instead:

```text
1. Inspect schema.
2. Inspect query.
3. Inspect row counts.
4. Inspect transformation history.
5. Determine which version is authoritative.
6. Explain the discrepancy.
7. Make the smallest justified correction.
```

---

# 54. If a Required Field Does Not Exist

Do not invent it.

Example:

If a source does not contain:

```text
severity
```

do not fabricate severity from:

```text
priority
status
label
```

unless a documented research rule explicitly defines that mapping.

Use:

```text
NULL
```

when the canonical schema allows missing source-specific information.

---

# 55. If an Agent Wants to Change the Research Methodology

Examples:

```text
change label definition
change original label logic
change model_text
change baseline
change treatment
change split strategy
change Knowledge Graph semantics
change agent roles
change Judge logic
change evaluation metrics
```

The agent MUST treat this as a methodological change.

Before implementing it:

1. Identify the existing methodology.
2. Explain the impact of the proposed change.
3. Preserve the original implementation where practical.
4. Version the new approach.
5. Update documentation.
6. Ensure baseline/treatment comparability remains valid.

---

# 56. Avoid Overengineering

The project is a research prototype.

Prefer:

```text
simple
modular
reproducible
testable
explainable
```

over unnecessary enterprise complexity.

Do not introduce:

```text
microservices
Kubernetes
complex distributed systems
multiple databases
unnecessary APIs
unnecessary frameworks
```

unless explicitly required.

---

# 57. Current Priority

The immediate priority is:

```text
Complete and validate Phase I EDA
```

The next priority is:

```text
Knowledge Graph Construction
```

After that:

```text
Policy Agent
Data Agent
Pattern Agent
Judge
```

Then:

```text
Agent-Audited Label Dataset
```

Then:

```text
DeBERTa-v3
XGBoost
SHAP
```

Do not jump directly to model training while foundational data and label-auditing
stages remain unfinished.

---

# 58. Definition of Done

A task is not complete merely because code was written.

A task is complete when:

```text
Code implemented
        +
Existing architecture preserved
        +
Relevant tests/validation executed
        +
Outputs inspected
        +
No unintended dataset changes
        +
Documentation updated where necessary
```

For data-processing tasks additionally verify:

```text
row count
duplicate count
class distribution
source distribution
nulls
schema
```

For agent tasks additionally verify:

```text
output schema
valid labels
confidence range
evidence preservation
agent disagreement preservation
failure handling
```

For model tasks additionally verify:

```text
dataset version
split
random seed
model configuration
evaluation metrics
reproducibility
```

---

# 59. Final Principle

The project should always preserve this conceptual separation:

```text
                 SOURCE DATA
                     ↓
              CANONICAL DATA
                     ↓
              KNOWLEDGE GRAPH
                     ↓
                  AGENTS
                     ↓
                  JUDGE
                     ↓
           AGENT-AUDITED LABELS
                     ↓
                 SDP MODEL
                     ↓
                  SHAP
                     ↓
              HUMAN REVIEW
```

The research contribution is the **label-auditing process and its effect on SDP**.

Every implementation decision should support:

```text
reproducibility
traceability
label provenance
experimental fairness
data integrity
explainability
```

When uncertain, inspect the existing implementation and database before making a
new assumption.
