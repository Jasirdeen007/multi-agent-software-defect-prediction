# Phase I — Data Status

## Completed
- BugHub PostgreSQL database imported.
- Official BugHub labelled views identified and used.
- Original labels retained as `original_label` (`bug`, `non-bug`).
- Canonical issue identity verified as `(source, project, issue_id)`.
- Source-specific JSON fields normalized.
- `model_text` created from title + description.
- Records with `model_text` shorter than 20 characters removed.

## Current verified dataset
- Total: 926,461
- BUG: 648,841
- NON-BUG: 277,620
- Empty model_text before filtering: 0
- Removed by <20-character rule: 1,276

## Next
Complete EDA, document findings, then freeze the Phase-I dataset before Knowledge Graph construction.
