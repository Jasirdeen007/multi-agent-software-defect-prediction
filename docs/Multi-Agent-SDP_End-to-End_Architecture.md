# Multi-Agent-SDP --- End-to-End Architecture and Implementation Workflow

## 1. Project Scope

**Project:** Multi-Agent-SDP\
**Core task:** Software Defect Prediction (SDP) with multi-agent label
auditing.

The central experimental question is whether **agent-audited issue
labels**, produced before model training, provide a measurable change in
downstream software defect prediction compared with the original BugHub
labels.

The implementation starts from the **already prepared canonical Parquet
dataset**. PostgreSQL/SQL extraction is therefore outside the
implementation workflow described here.

------------------------------------------------------------------------

# 2. Final End-to-End Architecture

``` text
                 ┌──────────────────────────────────────┐
                 │  CANONICAL DATASET (.parquet)        │
                 │  926,461 labelled issue reports      │
                 │                                      │
                 │  source, project, issue_id           │
                 │  title, description, model_text      │
                 │  original_label, metadata            │
                 └──────────────────┬───────────────────┘
                                    │
                                    ▼
                 ┌──────────────────────────────────────┐
                 │       DATA PREPROCESSING              │
                 │                                      │
                 │ • schema validation                  │
                 │ • duplicate validation               │
                 │ • text normalization                 │
                 │ • missing-value handling             │
                 │ • model_text validation              │
                 │ • label validation                   │
                 │ • project/source statistics           │
                 └──────────────────┬───────────────────┘
                                    │
                                    ▼
                 ┌──────────────────────────────────────┐
                 │       DATASET SPLITTING               │
                 │                                      │
                 │ • train                            │
                 │ • validation                       │
                 │ • held-out test                     │
                 │ • project-aware / temporal strategy  │
                 └──────────────────┬───────────────────┘
                                    │
                       ┌────────────┴────────────┐
                       │                         │
                       ▼                         ▼
          ┌──────────────────────┐    ┌─────────────────────────┐
          │ BASELINE EXPERIMENT  │    │ AGENT-AUDIT EXPERIMENT  │
          │                      │    │                         │
          │ original_label       │    │ original_label          │
          │       │              │    │       │                 │
          │       ▼              │    │       ▼                 │
          │ DeBERTa-v3           │    │ Knowledge Graph         │
          │       │              │    │       │                 │
          │       ▼              │    │       ▼                 │
          │ embeddings           │    │ Policy Agent             │
          │       │              │    │ Data Agent               │
          │       ▼              │    │ Pattern Agent             │
          │ XGBoost              │    │       │                  │
          │       │              │    │       ▼                  │
          │       ▼              │    │ Judge Agent              │
          │ prediction           │    │       │                  │
          └──────────┬───────────┘    │       ▼                  │
                     │                │ audited label            │
                     │                │ confidence + rationale   │
                     │                └──────────┬──────────────┘
                     │                           │
                     │                           ▼
                     │                ┌─────────────────────────┐
                     │                │ AGENT-AUDITED DATASET   │
                     │                └──────────┬──────────────┘
                     │                           │
                     │                           ▼
                     │                     DeBERTa-v3
                     │                           │
                     │                           ▼
                     │                     embeddings
                     │                           │
                     │                           ▼
                     │                        XGBoost
                     │                           │
                     │                           ▼
                     │                      prediction
                     │
                     └──────────────┬───────────────────────────
                                    │
                                    ▼
                 ┌──────────────────────────────────────┐
                 │      EVALUATION & COMPARISON          │
                 │                                      │
                 │ • Accuracy                            │
                 │ • Balanced Accuracy                   │
                 │ • Precision / Recall / F1             │
                 │ • ROC-AUC                             │
                 │ • MCC                                 │
                 │ • Brier score / calibration metrics  │
                 │                                      │
                 │ Compare:                              │
                 │ original-label model                 │
                 │          vs.                          │
                 │ agent-audited-label model             │
                 └──────────────────┬───────────────────┘
                                    │
                                    ▼
                 ┌──────────────────────────────────────┐
                 │          SHAP EXPLAINABILITY          │
                 │                                      │
                 │ prediction → feature contribution    │
                 │ → explanation for the XGBoost model   │
                 └──────────────────┬───────────────────┘
                                    │
                                    ▼
                 ┌──────────────────────────────────────┐
                 │       INFERENCE / APPLICATION         │
                 │                                      │
                 │ New issue                             │
                 │      ↓                                │
                 │ title + description preprocessing     │
                 │      ↓                                │
                 │ DeBERTa-v3                            │
                 │      ↓                                │
                 │ embedding                             │
                 │      ↓                                │
                 │ XGBoost                               │
                 │      ↓                                │
                 │ prediction + confidence              │
                 │      ↓                                │
                 │ SHAP explanation                      │
                 │      ↓                                │
                 │ confidence threshold                  │
                 │      │                                │
                 │      ├── high → prediction/output    │
                 │      │                                │
                 │      └── low → human tester review    │
                 │                    ↓                  │
                 │              final tester decision    │
                 └──────────────────────────────────────┘
```

------------------------------------------------------------------------

# 3. Stage 1 --- Canonical Parquet Input

The implementation begins with:

``` text
data/processed/bughubs_canonical.parquet
```

The current canonical dataset contains:

-   **926,461** labelled issue reports
-   **15** canonical fields
-   sources: Bugzilla, GitHub, and JIRA
-   issue identity: `(source, project, issue_id)`

Canonical fields:

``` text
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

`model_text` is the model-ready text formed from the issue title and
description.

The `original_label` is retained as the starting BugHub label.
Canonicalization must not silently replace or reinterpret this label.

------------------------------------------------------------------------

# 4. Stage 2 --- Data Preprocessing

The preprocessing module operates on the Parquet file.

## 4.1 Load

``` text
Parquet
   ↓
Pandas DataFrame
```

## 4.2 Schema validation

Check:

-   required columns exist
-   expected data types
-   label values are valid
-   timestamps can be parsed
-   text fields are accessible

## 4.3 Issue identity validation

Use:

``` text
(source, project, issue_id)
```

as the unique issue identity.

Check for:

``` text
duplicate composite identities
```

Do not use `issue_id` alone as the global unique key.

## 4.4 Text validation

Validate:

``` text
title
description
model_text
```

The model input remains:

``` text
model_text = title + description
```

Do not introduce information from the target label into `model_text`.

## 4.5 Missing-value handling

Missing metadata must be handled according to its meaning and source
availability.

Do not create artificial values merely to eliminate missingness.

For the SDP model, the principal input is the issue text. Metadata
should only be included if it is explicitly part of a later experiment.

## 4.6 Label validation

Preserve:

``` text
original_label
```

as the baseline label source.

The preprocessing stage does not create the agent-audited label.

------------------------------------------------------------------------

# 5. Stage 3 --- Dataset Splitting

The dataset must be divided before model training.

The critical rule is:

> The test set must not influence preprocessing decisions, agent
> auditing decisions, model fitting, hyperparameter selection, or
> threshold selection.

A project-aware or temporal split should be used where appropriate to
evaluate generalization and reduce leakage.

The project materials identify within-project temporal evaluation and
cross-project evaluation as experimental settings. These should be
implemented as controlled evaluation configurations rather than mixing
records indiscriminately across train and test.

A reproducible split configuration should record:

``` text
random seed
split strategy
train/validation/test definition
projects included in each split
```

------------------------------------------------------------------------

# 6. Stage 4 --- Baseline Experiment

The baseline establishes the performance obtained when training directly
from the original BugHub labels.

``` text
Canonical dataset
      │
      ▼
original_label
      │
      ▼
DeBERTa-v3
      │
      ▼
text embeddings
      │
      ▼
XGBoost
      │
      ▼
BUG / NON-BUG prediction
      │
      ▼
confidence
      │
      ▼
SHAP
```

The baseline and treatment must use the same:

-   input text
-   data split
-   embedding model
-   classifier
-   hyperparameter procedure
-   evaluation metrics

The principal experimental variable is the **label source**.

------------------------------------------------------------------------

# 7. Stage 5 --- Knowledge Graph

The Knowledge Graph provides structured project context to the
label-auditing agents.

The project design specifies a project-taxonomy Knowledge Graph using
**Neo4j**.

Conceptually:

``` text
Project
   │
   ├── issue/report context
   ├── project taxonomy
   ├── relevant categories
   └── structured evidence
```

The KG is an evidence source for the agents.

It is not itself the final classifier.

The exact KG schema should be implemented from fields/evidence that
actually exist in the canonical data and project sources. Do not invent
taxonomy nodes that are not supported by available data.

------------------------------------------------------------------------

# 8. Stage 6 --- Multi-Agent Label Audit

This is the central research component.

The original label is supplied to the auditing process together with the
issue information and available structured evidence.

``` text
Issue
  │
  ├── title
  ├── description
  ├── project
  ├── original label
  └── KG evidence
          │
          ▼
   ┌───────────────────┐
   │   Policy Agent    │
   ├───────────────────┤
   │ policy/rule-based │
   │ evidence          │
   └─────────┬─────────┘
             │
   ┌─────────▼─────────┐
   │    Data Agent     │
   ├───────────────────┤
   │ data/context       │
   │ evidence           │
   └─────────┬─────────┘
             │
   ┌─────────▼─────────┐
   │  Pattern Agent    │
   ├───────────────────┤
   │ semantic/pattern   │
   │ evidence           │
   └─────────┬─────────┘
             │
             ▼
       ┌──────────┐
       │  Judge   │
       └────┬─────┘
            │
            ▼
   audited label + confidence
   + rationale + agent evidence
```

## 8.1 Policy Agent

Examines the issue against explicit project/data-label policies and
available structured rules.

Output should include:

``` text
agent_label
confidence
evidence
reasoning/rationale
```

## 8.2 Data Agent

Examines available issue metadata and dataset evidence.

Output should include:

``` text
agent_label
confidence
evidence
reasoning/rationale
```

## 8.3 Pattern Agent

Examines the issue text and relevant issue patterns.

Output should include:

``` text
agent_label
confidence
evidence
reasoning/rationale
```

## 8.4 Judge Agent

The Judge receives the individual agent outputs and available KG
evidence.

It consolidates the evidence into:

``` text
audited_label
judge_confidence
judge_rationale
disagreement_information
```

The Judge should not simply select the majority label without
considering evidence.

------------------------------------------------------------------------

# 9. Stage 7 --- Agent Audit Record

For every audited issue, preserve an audit record.

A practical structure is:

``` text
issue_id
source
project

original_label

policy_label
policy_confidence
policy_evidence

data_label
data_confidence
data_evidence

pattern_label
pattern_confidence
pattern_evidence

judge_label
judge_confidence
judge_rationale

agent_agreement
human_review_required
human_label
final_audited_label
```

The exact implementation can be adjusted, but the important principle is
that the agent decisions are **traceable**.

Do not overwrite `original_label`.

------------------------------------------------------------------------

# 10. Stage 8 --- Human-in-the-Loop Validation

Human review is used when the audit decision requires human confirmation
according to the project's review policy.

Typical triggers can include:

``` text
low Judge confidence
high agent disagreement
policy conflict
insufficient evidence
```

The system records:

``` text
original_label
agent_audited_label
agent_confidence
human_label
review_reason
```

If the tester disagrees with the model/agent decision:

``` text
model/agent output ≠ human decision
```

the original machine outputs remain in the audit record.

The human decision becomes the final decision for that reviewed case.

Human feedback is not automatically injected into model training. A
later controlled retraining process can use approved human-labelled
data.

------------------------------------------------------------------------

# 11. Stage 9 --- Agent-Audited Dataset

After the audit process:

``` text
Canonical dataset
       │
       ├── original_label
       │
       └── agent audit
              │
              ▼
       agent_audited_label
```

The resulting dataset contains both label sources.

Conceptually:

``` text
issue_id
source
project
model_text
original_label
agent_audited_label
audit_confidence
audit_rationale
...
```

This allows direct experimental comparison without losing the original
label provenance.

------------------------------------------------------------------------

# 12. Stage 10 --- Treatment Experiment

The treatment uses the same model pipeline as the baseline, but replaces
the training target with the audited label.

``` text
Agent-audited dataset
        │
        ▼
agent_audited_label
        │
        ▼
DeBERTa-v3
        │
        ▼
embeddings
        │
        ▼
XGBoost
        │
        ▼
BUG / NON-BUG
```

The model architecture remains unchanged.

This isolates the research variable:

``` text
original BugHub label
        VS
agent-audited label
```

------------------------------------------------------------------------

# 13. Stage 11 --- DeBERTa-v3 Embedding Pipeline

For both baseline and treatment:

``` text
model_text
    ↓
tokenization
    ↓
DeBERTa-v3
    ↓
fixed-length representation / embedding
    ↓
XGBoost
```

The same embedding procedure must be used for both experiments.

The embedding model should be fit/configured without using the held-out
test labels.

------------------------------------------------------------------------

# 14. Stage 12 --- XGBoost Classifier

The DeBERTa representation becomes the feature vector for XGBoost.

``` text
DeBERTa embedding
       ↓
XGBoost classifier
       ↓
predicted class
       ↓
probability/confidence
```

The project design uses XGBoost as the downstream classifier and SHAP
for explainability.

The baseline and treatment should use the same classifier configuration
or the same controlled hyperparameter-selection procedure.

------------------------------------------------------------------------

# 15. Stage 13 --- SHAP Explainability

SHAP is applied to the trained XGBoost model.

The goal is to provide an explanation of the model's prediction in terms
of the input features used by XGBoost.

``` text
Issue
  ↓
DeBERTa embedding
  ↓
XGBoost
  ↓
prediction
  ↓
SHAP
  ↓
feature contribution explanation
```

SHAP is an explanation layer. It does not change the prediction.

------------------------------------------------------------------------

# 16. Stage 14 --- Evaluation

The two experiments are evaluated using the same held-out data and
evaluation protocol.

Recommended project metrics already identified in the project materials
include:

``` text
Accuracy
Balanced Accuracy
Precision
Recall
F1
ROC-AUC
MCC
Brier Score / calibration-related metrics
```

The evaluation should report:

``` text
Baseline:
original-label → model → metrics

Treatment:
agent-audited-label → same model → metrics
```

Do not claim that the treatment improves performance until the
experiment produces the corresponding evidence.

------------------------------------------------------------------------

# 17. Stage 15 --- Ablation Studies

The project materials identify ablations as part of the evaluation
design.

Potential controlled comparisons include:

``` text
Full multi-agent council
        VS
Single-agent configuration

KG-guided council
        VS
Council without KG guidance

Full council + Judge
        VS
Council without Judge
```

The exact ablation set should be finalized before running the
experiments.

Each ablation should change one major component at a time where
practical.

------------------------------------------------------------------------

# 18. Stage 16 --- Final Inference Workflow

After the treatment model has been trained:

``` text
                NEW ISSUE
                   │
                   ▼
          Title + Description
                   │
                   ▼
            Preprocessing
                   │
                   ▼
             model_text
                   │
                   ▼
             DeBERTa-v3
                   │
                   ▼
              embedding
                   │
                   ▼
               XGBoost
                   │
          ┌────────┴────────┐
          ▼                 ▼
      prediction        confidence
          │                 │
          └────────┬────────┘
                   ▼
                 SHAP
                   │
                   ▼
          confidence threshold
             │           │
          high          low
             │           │
             ▼           ▼
        prediction     Human
          output       tester
                         │
                         ▼
                   final decision
```

For a high-confidence case, the system returns the model prediction.

For a low-confidence case, the case is routed to human review.

The human decision is stored separately from the model prediction.

Example:

``` text
model_prediction = BUG
model_confidence  = 0.58

human_decision    = NON-BUG

final_decision    = NON-BUG
```

The model prediction remains unchanged in the audit log.

------------------------------------------------------------------------

# 19. Application Architecture

The project also includes an application layer for presenting
predictions and explanations.

The project materials specify:

``` text
Backend:
FastAPI

Frontend:
React.js

Database:
PostgreSQL

Containerization:
Docker Compose
```

The current implementation should expose the core SDP workflow first.

A practical application flow is:

``` text
React UI
   │
   ▼
FastAPI
   │
   ├── preprocessing
   ├── DeBERTa embedding
   ├── XGBoost prediction
   └── SHAP explanation
   │
   ▼
Response
   ├── prediction
   ├── confidence
   ├── SHAP explanation
   └── human-review status
```

The current project scope does not require the application layer to
perform the entire training pipeline.

Training and auditing are offline research workflows; inference is the
serving workflow.

------------------------------------------------------------------------

# 20. Experiment Data Flow

The complete research data flow is:

``` text
Canonical Parquet
       │
       ▼
Preprocessing
       │
       ▼
Validated Dataset
       │
       ├─────────────────────────────┐
       │                             │
       ▼                             ▼
Original Labels                 Agent Audit
       │                             │
       ▼                       KG + 3 Agents
Baseline Training                    │
       │                           Judge
       ▼                             │
DeBERTa-v3                           ▼
       │                       Audited Labels
       ▼                             │
XGBoost                               ▼
       │                         Treatment Training
       ▼                             │
Baseline Metrics                      ▼
                              DeBERTa-v3
                                     │
                                     ▼
                                  XGBoost
                                     │
                                     ▼
                              Treatment Metrics
                                     │
                    ┌────────────────┘
                    ▼
             Comparative Evaluation
                    │
                    ▼
               SHAP Analysis
                    │
                    ▼
             Inference System
                    │
                    ▼
             Human Review when needed
```

------------------------------------------------------------------------

# 21. Recommended Repository Mapping

``` text
Multi_agent_sdp/
│
├── configs/
│   ├── model.yaml
│   ├── agents.yaml
│   └── experiments.yaml
│
├── data/
│   ├── raw/
│   ├── processed/
│   │   └── bughubs_canonical.parquet
│   └── README.md
│
├── docs/
│   ├── DATASET_CARD.md
│   ├── DATASET_DICTIONARY.md
│   ├── DATASET_SETUP.md
│   └── ARCHITECTURE.md
│
├── notebooks/
│   ├── exploratory/
│   └── experiments/
│
├── outputs/
│   ├── figures/
│   ├── tables/
│   ├── predictions/
│   └── reports/
│
├── src/
│   ├── preprocessing/
│   ├── agents/
│   │   ├── policy_agent.py
│   │   ├── data_agent.py
│   │   ├── pattern_agent.py
│   │   └── judge_agent.py
│   │
│   ├── knowledge_graph/
│   ├── embeddings/
│   ├── models/
│   │   ├── baseline/
│   │   └── treatment/
│   │
│   ├── explainability/
│   ├── evaluation/
│   ├── inference/
│   └── api/
│
├── tests/
│
├── AGENTS.md
├── README.md
└── requirements.txt
```

------------------------------------------------------------------------

# 22. Implementation Order

Do not implement the whole system simultaneously.

Use this order:

### Phase A --- Data

``` text
1. Load canonical Parquet
2. Validate schema
3. Validate composite uniqueness
4. Validate labels
5. Validate model_text
6. Generate EDA
7. Create reproducible train/validation/test splits
```

### Phase B --- Baseline

``` text
8. Build original-label baseline
9. Generate DeBERTa embeddings
10. Train XGBoost
11. Evaluate
12. Save baseline model
13. Generate SHAP explanations
```

### Phase C --- Knowledge Graph

``` text
14. Define KG schema
15. Populate supported project/taxonomy evidence
16. Validate KG retrieval/evidence
```

### Phase D --- Agent Council

``` text
17. Implement Policy Agent
18. Implement Data Agent
19. Implement Pattern Agent
20. Implement Judge
21. Define confidence/disagreement rules
22. Run audit
23. Save complete audit trail
```

### Phase E --- Human Validation

``` text
24. Select cases requiring human review
25. Record human decisions
26. Produce final audited labels
27. Measure agent/human agreement
```

### Phase F --- Treatment

``` text
28. Build treatment training dataset
29. Generate DeBERTa embeddings
30. Train XGBoost
31. Evaluate using the same protocol
32. Generate SHAP explanations
```

### Phase G --- Research Evaluation

``` text
33. Baseline vs treatment comparison
34. Ablation studies
35. Error analysis
36. Statistical analysis where appropriate
37. Produce final experiment report
```

### Phase H --- Application

``` text
38. FastAPI inference service
39. React dashboard integration
40. Human-review interface
41. Prediction/audit logging
42. Docker Compose deployment
```

------------------------------------------------------------------------

# 23. What Is Actually the Research Contribution?

The project contribution is not simply:

``` text
DeBERTa + XGBoost
```

Those are the downstream modelling components.

The research contribution being evaluated is the **multi-agent
label-auditing layer**:

``` text
heterogeneous issue labels
        ↓
structured KG evidence
        ↓
Policy + Data + Pattern agents
        ↓
Judge
        ↓
audited labels
        ↓
SDP model
        ↓
comparison against original-label baseline
```

The experiment tests whether changing the **label quality before
training** changes downstream SDP results.

------------------------------------------------------------------------

# 24. Important Scope Corrections

The earlier AgentTriage presentation contained a broader four-layer
design that included:

-   developer recommendation
-   resolution-time estimation
-   severity/priority multi-task prediction
-   RAG/Qdrant
-   isotonic calibration

Those elements appeared in the earlier presentation, but they are **not
part of the current finalized core SDP workflow** represented in this
document.

The current implementation should remain focused on:

``` text
Canonical dataset
→ preprocessing
→ multi-agent label auditing
→ Judge
→ human validation when required
→ DeBERTa-v3
→ XGBoost
→ SHAP
→ baseline/treatment comparison
→ inference
```

Do not reintroduce the removed components unless the project scope is
explicitly changed.

------------------------------------------------------------------------

# 25. Reproducibility Requirements

Every experiment should record:

``` text
dataset version
dataset checksum/version identifier
split strategy
random seed
embedding model/version
XGBoost configuration
agent prompt/policy version
KG version
Judge configuration
human-review policy
evaluation configuration
```

The original label must always remain available for comparison.

Agent-audited labels must be versioned rather than silently overwriting
source labels.

------------------------------------------------------------------------

# 26. Final System Summary

The final system can be represented in one line:

``` text
Canonical Parquet
→ preprocessing
→ controlled split
→ original-label baseline
→ KG-guided multi-agent label audit
→ Judge + audit trail
→ human validation when required
→ agent-audited dataset
→ DeBERTa-v3 embeddings
→ XGBoost SDP
→ SHAP
→ baseline/treatment evaluation
→ inference API
→ human review for low-confidence predictions
```

The key experimental comparison is:

``` text
ORIGINAL LABEL
     ↓
DeBERTa-v3 + XGBoost
     ↓
Baseline SDP performance

                VS

AGENT-AUDITED LABEL
     ↓
DeBERTa-v3 + XGBoost
     ↓
Treatment SDP performance
```

The architecture therefore keeps **data preparation, label auditing,
model training, explainability, evaluation, and deployment as separate
modules**, allowing each component to be tested independently and the
effect of agentic label auditing to be measured without changing the
downstream model architecture.
