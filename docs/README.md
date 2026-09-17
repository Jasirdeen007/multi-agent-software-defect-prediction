# Multi-Agent-SDP

## Multi-Agent Software Defect Prediction

A research project for software defect prediction using **multi-agent label auditing**, **Knowledge Graphs**, **DeBERTa-v3**, **XGBoost**, and **SHAP explainability**.

---

# 1. Project Overview

The objective of this project is to investigate whether a multi-agent system can improve the reliability of software defect labels before training a Software Defect Prediction (SDP) model.

The project uses issue reports collected from:

- Bugzilla
- GitHub
- JIRA

The source data is transformed into a common canonical representation and analyzed before being passed to the multi-agent auditing pipeline.

---

# 2. Current Project Status

## Phase I — Data Preparation and Exploratory Data Analysis

The current phase consists of:

1. PostgreSQL dataset setup
2. BugHub source-label extraction
3. Canonical dataset construction
4. Data normalization
5. Data quality filtering
6. Exploratory Data Analysis (EDA)

The next phase will be:

```text
Knowledge Graph Construction
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
XGBoost
        ↓
SHAP Explainability