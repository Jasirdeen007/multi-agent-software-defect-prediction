# Multi-Agent-SDP — Phase I Review
## 7-Page PPT-Ready Research Content

> **Project research statement:** Multi-Agent-SDP proposes a multi-agent label-auditing framework for heterogeneous software issue reports. The audited labels are subsequently used to train a DeBERTa-v3 + XGBoost software defect prediction model, with SHAP for explanation. Human-in-the-loop review belongs to the **inference/operational phase** and is therefore stated separately rather than included in the current training-time architecture.

---

# PAGE 1 — Detailed Literature Review

## 1. BugHub: A Large Scale Issue Report Dataset

**Andrade, Laranjeiro & Vieira (2023), “BugHub: A Large Scale Issue Report Dataset.”**

BugHub provides a large heterogeneous collection of software issue reports from **93 projects** and three issue-tracking ecosystems: Bugzilla, JIRA and GitHub. The reported dataset contains **2,462,666 unique issue reports** and was collected between April 2021 and May 2023.

**Relevance to Multi-Agent-SDP:**  
BugHub provides the large, heterogeneous issue-report population required for evaluating defect classification across different trackers and projects. Its scale also motivates a selective LLM-auditing strategy instead of sending the complete population through an expensive agent pipeline.

**Source:** Zenodo DOI **10.5281/zenodo.10028953**.

---

## 2. Automatic Techniques for Issue Report Classification: A Systematic Mapping Study

**Automated Software Engineering (2026).**

The systematic mapping study reviews research on automatic issue-report classification, covering traditional machine learning, deep learning, Transformer/BERT-family models and emerging LLM-based approaches. It identifies challenges concerning classification quality, explainability, scalability, generalization and practical use.

**Relevance:**  
The study establishes that automated issue classification is an active research area, while motivating research beyond classifier construction alone. Multi-Agent-SDP therefore focuses on the upstream **quality of labels used for defect prediction**.

---

## 3. Deep Learning-Based Software Bug Classification

**Information and Software Technology (2024), 166, 107350.**

The study investigates large-scale software bug classification using heuristically annotated bug-resolution reports and compares Transformer-family approaches including BERT, CodeBERT and DistilBERT. The reported results demonstrate the usefulness of Transformer representations for software-bug classification.

**Relevance:**  
This supports the use of a Transformer encoder for semantic representation of issue reports. Multi-Agent-SDP differs by introducing a label-auditing stage before downstream classification.

---

## 4. Multi-LLM Disagreement as a Scalable Detector of Human Annotation Errors in Structured Data from Clinical Free-Text

**Wittlinger et al. (2026).**

The study investigates multiple independently operating LLMs as an annotation-quality-control mechanism. LLM agreement/disagreement is used to identify cases requiring additional scrutiny, with human adjudication used for difficult cases.

**Relevance:**  
This provides methodological support for the proposed **Agent Council → disagreement detection → adjudication** principle. The application domain is clinical annotation rather than software engineering; therefore, its findings support the quality-control methodology but do not directly establish performance on BugHub.

**Evidence boundary:** The literature supports the *methodological principle*, while Multi-Agent-SDP must experimentally establish whether it improves software defect-label quality.

---

# PAGE 2 — Research Gap Identification

## Problem Definition

Software repositories contain heterogeneous issue reports representing defects, enhancements, documentation problems, configuration problems, questions and other development activities. A defect prediction model is dependent not only on its architecture but also on the reliability of its training labels.

### Research Problem

> **How can heterogeneous software issue labels be selectively audited at scale so that potentially unreliable or ambiguous labels are identified before they are used for downstream software defect prediction?**

## Identified Research Gaps

### Gap 1 — Label quality versus classifier performance

A large portion of issue-classification research concentrates on improving the predictive model. Less attention is placed on systematically auditing the labels supplied to that model.

### Gap 2 — Scalability of LLM-based auditing

Applying several LLM agents to hundreds of thousands of issue reports is computationally and financially expensive. A scalable system therefore requires **selective auditing** rather than exhaustive LLM processing.

### Gap 3 — Single-judgment limitation

A single model or heuristic produces one classification decision. Independent agent perspectives provide an additional mechanism for exposing disagreement and uncertainty.

### Gap 4 — Human review integration

Ambiguous cases should not automatically become training truth. A controlled workflow can retain human review for unresolved cases.

### Gap 5 — Effect of label auditing on SDP

The key experimental question is not only whether agents can classify issues, but whether **audited labels affect downstream software defect prediction performance**.

## Research Gap Statement

> **Existing work provides evidence for automated issue classification and, separately, for multi-LLM disagreement-based annotation quality control. Multi-Agent-SDP investigates their integration as a selective software-issue label-auditing layer before controlled software defect prediction.**

---

# PAGE 3 — System Architecture and Workflow

## Training-Time Research Architecture

```text
                    BUG / ISSUE REPORTS
                           │
                           ▼
              ┌────────────────────────┐
              │   CANONICAL DATASET    │
              │   926,461 LABELED      │
              │        ISSUES          │
              └────────────┬───────────┘
                           │
                           ▼
                SELECTIVE AUDIT SAMPLING
                           │
                           ▼
              ┌─────────────────────────┐
              │     KNOWLEDGE GRAPH     │
              │                         │
              │ Project / Source /      │
              │ Metadata / Label /     │
              │ Relationships / Rules   │
              └────────────┬────────────┘
                           │
                           ▼
             ┌───────────────────────────┐
             │      AGENT COUNCIL        │
             │                           │
             │ Policy Agent              │
             │ Data Agent                │
             │ Pattern Agent             │
             └────────────┬──────────────┘
                          │
                          ▼
                 EVIDENCE + AGREEMENT
                          │
                          ▼
                    JUDGE AGENT
                          │
                          ▼
                 AUDITED SILVER LABEL
                          │
                          ▼
             ┌─────────────────────────┐
             │   DEBERTA-V3 ENCODER    │
             │                         │
             │ Issue text → embedding  │
             └────────────┬────────────┘
                          │
                          ▼
                 ┌─────────────────┐
                 │    XGBOOST      │
                 │ BUG / NON-BUG   │
                 └────────┬────────┘
                          │
                          ▼
                    SHAP EXPLANATION
                          │
                          ▼
                  EVALUATION RESULTS
```

## Knowledge Graph Role

The Knowledge Graph provides structured context to the auditing agents, including relationships among:

- issue
- project
- source/tracker
- metadata
- label
- classification evidence
- project-specific patterns/rules

It is an **evidence/context layer**, not a replacement for the agents or the predictive model.

## Agent Council

**Policy Agent:** checks label consistency against source/project classification policy and explicit rules.

**Data Agent:** examines structured issue metadata and source-level evidence.

**Pattern Agent:** examines issue text for defect/non-defect patterns and ambiguity.

**Judge Agent:** consolidates the structured agent outputs and evidence into the audited label.

### Important architecture boundary

The **Human-in-the-Loop is not part of this training-time architecture diagram.**

It belongs to the **inference/operational phase**:

```text
New Issue
   ↓
DeBERTa-v3 → XGBoost
   ↓
Prediction + Confidence + SHAP
   ↓
Low-confidence / uncertain
   ↓
Human Tester Review
   ↓
Final Operational Decision
```

The human decision is retained separately from the model prediction.

---

# PAGE 4 — Dataset Details

## BugHub Source

**BugHub: A Large Scale Issue Report Dataset**

| Property | Verified / Reported Value |
|---|---:|
| Overall BugHub population | **2,462,666 unique reports** |
| Projects | **93** |
| Sources | **Bugzilla, JIRA, GitHub** |
| Collection period | **April 2021 – May 2023** |
| Database requirement | PostgreSQL 13+ |
| Supplementary dataset | `bughub.sql.gz` |

**Primary source:** Andrade, Laranjeiro & Vieira, Zenodo DOI **10.5281/zenodo.10028953**.

---

## Current Multi-Agent-SDP Canonical Population

The project uses the official BugHub labelled views and constructs a normalized canonical table.

| Measure | Current verified value |
|---|---:|
| Initial labelled canonical records | **927,737** |
| Removed very-short records | **1,276** |
| Final canonical records | **926,461** |
| BUG | **648,841** |
| NON-BUG | **277,620** |
| BUG proportion | **70.03%** |
| NON-BUG proportion | **29.97%** |

## Canonical Schema

`issue_id, source, project, title, description, component, severity, priority, status, resolution, original_label, label_source, created_at, updated_at, model_text`

### Modelling text

```text
model_text = title + description
```

The current preprocessing removed only records with unusable combined modelling text where trimmed length was below 20 characters.

### Label provenance

`original_label` is preserved from the official BugHub source views.

The project retains BugHub's source-specific label-generation logic rather than introducing a new universal `type = defect` rule.

### Issue identity

The canonical uniqueness key is:

```text
(source, project, issue_id)
```

This is necessary because an issue ID can recur across different projects or tracking systems.

### Dataset interpretation

The **926,461 records are the current labelled modelling population**. They are not all intended to pass through the expensive LLM agent council.

---

# PAGE 5 — Proposed Methodology and Solution

## Phase 1 — Canonical Data Preparation

BugHub source-specific records are transformed into a common schema while preserving source, project and original-label provenance.

**Output:** 926,461 validated labelled issues.

---

## Phase 2 — Selective Agent Auditing

A representative audit subset is selected from the full population.

Sampling should preserve relevant variation such as:

- issue-tracking source
- project
- original label
- text length
- metadata completeness
- potentially ambiguous cases

### Why selective auditing?

The objective is not to replace database processing with LLM reasoning. The objective is to use LLM reasoning where it provides additional value: **label verification, evidence extraction and ambiguity detection**.

The multi-LLM disagreement literature provides methodological support for concentrating additional review on low-agreement cases.

---

## Phase 3 — Knowledge-Guided Agent Council

Each selected issue is evaluated independently by:

```text
Policy Agent
Data Agent
Pattern Agent
```

The agents receive the issue text and the relevant structured/Knowledge-Graph evidence.

Each agent returns structured output:

```text
agent_label
confidence
evidence
reason
agent_version
```

The outputs are retained separately for traceability.

---

## Phase 4 — Judge

The Judge receives the independent agent outputs and supporting evidence.

Conceptually:

```text
Policy ─┐
Data ───┼──→ Judge → audited silver label
Pattern ┘
```

Agreement/disagreement is retained as an audit signal.

**The exact confidence thresholds and routing rules are experimental parameters and must be validated rather than presented as established facts.**

---

## Phase 5 — Two Controlled SDP Experiments

### Baseline

```text
Original BugHub labels
        ↓
DeBERTa-v3
        ↓
XGBoost
        ↓
SHAP
        ↓
Baseline evaluation
```

### Proposed

```text
Agent-audited silver labels
        ↓
DeBERTa-v3
        ↓
XGBoost
        ↓
SHAP
        ↓
Proposed evaluation
```

The **same model architecture, preprocessing and evaluation protocol** should be used for both experiments. The principal experimental variable is the label source.

---

## Data Splitting

The split must be performed without information leakage and applied consistently to both baseline and proposed experiments.

Possible research designs include:

**Project-aware split**

```text
Training projects → Validation projects → Unseen test projects
```

**Temporal split**

```text
Earlier issues → Training
Later issues   → Validation/Test
```

The final split strategy should be fixed before final evaluation. Random row-level splitting should not be assumed to measure cross-project generalization because issues from the same project can share terminology, templates and project-specific characteristics.

---

# PAGE 6 — Feasibility Analysis

## Computational Feasibility

The architecture separates full-dataset processing from expensive LLM reasoning.

### Full population — low-cost processing

The complete 926,461-record population can be processed using:

- PostgreSQL
- SQL filtering
- canonicalization
- metadata validation
- deterministic preprocessing
- sampling

### Selective population — expensive reasoning

Only the selected audit population is processed by:

- Policy Agent
- Data Agent
- Pattern Agent
- Judge Agent when required

This avoids multiplying LLM processing across the complete dataset.

---

## Audit Sample Size

A fixed number such as **20,000 must not be presented as a proven requirement**.

A defensible experimental approach is:

```text
Pilot sample
     ↓
Measure agreement / disagreement / label changes
     ↓
Increase sample
     ↓
Check stability of audit estimates
     ↓
Finalize sample size
```

Possible pilot stages:

`5K → 10K → 20K`

These are engineering study points, not literature-established sample sizes.

The final sample should be justified using:

- stratified/representative sampling
- confidence intervals
- disagreement rate
- observed label-change rate
- human-review capacity
- stability of measured audit statistics

---

## Cost-Control Measures

1. Prefer local inference through **Ollama** where hardware permits.
2. Use one compact instruction model for the council rather than requiring three different foundation models.
3. Give each agent a different role, prompt, evidence scope and output schema.
4. Send only necessary issue text and evidence.
5. Cache agent outputs.
6. Version prompts, policies and agent configurations.
7. Run the Judge only when the routing policy requires consolidation.
8. Reserve human review for unresolved operational cases.

### Recommended implementation stack

| Function | Technology |
|---|---|
| Agent orchestration | LangGraph |
| Local LLM runtime | Ollama |
| Agent LLM | Qwen2.5 3B Instruct |
| Knowledge Graph | Neo4j Community Edition |
| Dataset / audit storage | PostgreSQL |
| Semantic encoder | DeBERTa-v3 |
| Classifier | XGBoost |
| Explainability | SHAP |
| Optional experiment tracking | MLflow |

---

# PAGE 7 — Novelty of the Proposed Work

## Novelty 1 — Multi-perspective label auditing

The system does not rely exclusively on one heuristic or one LLM decision. Independent Policy, Data and Pattern agents generate complementary evidence before the Judge consolidates the result.

---

## Novelty 2 — Knowledge-guided auditing

The Knowledge Graph connects issue, project, source, metadata, labels and relevant evidence so that agents can reason with structured context rather than treating each issue as an isolated text sample.

---

## Novelty 3 — Selective LLM processing for large-scale issue data

The complete 926K labelled population is retained, while expensive agent reasoning is concentrated on a selected audit population.

This creates a separation between:

```text
Dataset-scale data engineering
              +
Targeted LLM quality auditing
```

---

## Novelty 4 — Disagreement-aware adjudication

Agent disagreement is treated as an uncertainty/quality-control signal.

```text
Independent agent outputs
          ↓
Agreement analysis
          ↓
Judge consolidation
          ↓
Audited silver label
```

This principle is supported by recent multi-LLM annotation-quality research, but its effectiveness for software issue labels remains an empirical question.

---

## Novelty 5 — Label auditing as an intervention in SDP

The research explicitly compares:

```text
Original labels
      ↓
DeBERTa-v3 + XGBoost
```

against:

```text
Agent-audited labels
      ↓
DeBERTa-v3 + XGBoost
```

under the same evaluation protocol.

This allows the study to investigate whether **label auditing changes downstream software defect prediction performance**.

---

## Novelty 6 — Human-centered inference

Human-in-the-loop review is positioned in the operational inference stage rather than mixed into the training-time architecture.

```text
New Issue
   ↓
DeBERTa-v3
   ↓
XGBoost
   ↓
Prediction + Confidence + SHAP
   ↓
Low-confidence case
   ↓
Human Tester
   ↓
Final operational decision
```

The system therefore functions as **AI-assisted software defect triage**, not as an autonomous replacement for testers.

---

## Final Research Contribution

> **Multi-Agent-SDP proposes a knowledge-guided, selective multi-agent label-auditing framework for heterogeneous software issue reports. Independent policy, data and pattern agents audit selected records, their evidence is consolidated by a Judge, and the resulting audited silver labels are used in a controlled DeBERTa-v3–XGBoost software defect prediction experiment. SHAP provides model-level explanations, while human-in-the-loop review is reserved for uncertain cases during operational inference.**

## Evidence Boundary

### Supported by existing evidence

- BugHub provides a large heterogeneous issue-report population.
- Transformer models are established for issue/bug classification.
- Large-scale heuristic annotation has been used in software-bug classification research.
- Multi-LLM disagreement can serve as an annotation-quality-control signal.
- Human adjudication can be used for difficult/disputed cases.

### To be established by Multi-Agent-SDP

- Whether the Policy/Data/Pattern decomposition improves audit quality.
- Whether the Knowledge Graph improves agent decisions.
- What audit-sample size is sufficient for this dataset.
- Whether audited labels improve downstream SDP performance.
- Whether the approach generalizes across Bugzilla, GitHub and JIRA projects.

---

# SELECTED LITERATURE REFERENCES

1. **Andrade, R., Laranjeiro, N., & Vieira, M. (2023).** *BugHub: A Large Scale Issue Report Dataset.* Zenodo. DOI: **10.5281/zenodo.10028953**.

2. **Automatic Techniques for Issue Report Classification: A Systematic Mapping Study. (2026).** *Automated Software Engineering*, 33, Article 72.

3. **Deep Learning-Based Software Bug Classification. (2024).** *Information and Software Technology*, 166, 107350. DOI: **10.1016/j.infsof.2023.107350**.

4. **Wittlinger, S., Meerjansen, J., Wolf, F., Wiest, I. C., Ebert, M. P., Siegel, F., & Belle, S. (2026).** *Multi-LLM Disagreement as a Scalable Detector of Human Annotation Errors in Structured Data from Clinical Free-Text.* medRxiv. DOI: **10.64898/2026.05.04.26352392**.

---

# KEY METHODOLOGICAL POSITION FOR THE PPT

> **The project does not send all 926,461 records through LLM agents. The full population is processed deterministically, while a representative audit subset is passed through a Knowledge-Guided Agent Council. Agent agreement/disagreement and evidence are used for auditing and Judge-based consolidation. The resulting silver labels are then used for the proposed DeBERTa-v3 + XGBoost SDP experiment. Human-in-the-loop review is reserved for uncertain cases during inference.**

