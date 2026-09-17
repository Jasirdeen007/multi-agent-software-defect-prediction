# Dataset Description

## Project

Multi-Agent Software Defect Prediction (SDP)

## Source Dataset

The project uses the BugHub issue-report dataset as the source data.

BugHub contains issue reports collected from:

- Bugzilla
- GitHub
- JIRA

The complete BugHub dataset is larger than the population used in this project.

## Project Working Dataset

The project's current working dataset is:

```text
PostgreSQL
    Database: bughub
    Schema: agenttriage
    Table: canonical_dataset