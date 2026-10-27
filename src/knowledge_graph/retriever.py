from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd


class KnowledgeGraphRetriever:
    """Training-only relational statistics, with each target issue excluded from its own priors."""

    def __init__(self, dataset_path: str = "data/processed/fixed_splits/train.parquet"):
        self.dataset_path = Path(dataset_path)
        if not self.dataset_path.exists():
            raise FileNotFoundError(f"Training-only KG source not found: {self.dataset_path}")
        cols = ["source", "project", "issue_id", "component", "severity", "original_label"]
        self.df = pd.read_parquet(self.dataset_path, columns=cols)
        self.df["issue_id"] = self.df["issue_id"].astype(str)
        self.df["_project_key"] = self.df["project"].fillna("").astype(str).str.casefold()
        self.df["_component_key"] = self.df["component"].fillna("").astype(str).str.casefold()
        self.df["_source_key"] = self.df["source"].fillna("").astype(str).str.casefold()
        self.df["_is_bug"] = self.df["original_label"].fillna("").astype(str).str.casefold().eq("bug")
        self.project_stats = self.df.groupby("_project_key")["_is_bug"].agg(["sum", "count"])
        self.component_stats = self.df.groupby(["_project_key", "_component_key"])["_is_bug"].agg(["sum", "count"])
        self.severity_stats = self.df.groupby(["_source_key", self.df["severity"].fillna("").astype(str).str.casefold()])["_is_bug"].agg(["sum", "count"])

    @staticmethod
    def _loo(stats: pd.DataFrame, key: Any, current_is_bug: bool) -> tuple[int, int]:
        if key not in stats.index:
            return 0, 0
        row = stats.loc[key]
        return int(row["sum"]) - int(current_is_bug), int(row["count"]) - 1

    @staticmethod
    def _format_prior(label: str, bug_count: int, total: int) -> str | None:
        if total <= 0:
            return None
        return f"[Train-only KG {label}] {100 * bug_count / total:.1f}% original-label BUG rate ({bug_count}/{total} training issues; target issue excluded)."

    def retrieve_evidence_for_issue(self, issue: dict[str, Any]) -> list[str]:
        project = str(issue.get("project") or "").casefold()
        component = str(issue.get("component") or "").casefold()
        source = str(issue.get("source") or "").casefold()
        issue_id = str(issue.get("issue_id") or "")
        rows = self.df[
            self.df["_project_key"].eq(project)
            & self.df["_source_key"].eq(source)
            & self.df["issue_id"].eq(issue_id)
        ]
        current_is_bug = bool(rows.iloc[0]["_is_bug"]) if len(rows) else False
        evidence: list[str] = []

        bugs, total = self._loo(self.project_stats, project, current_is_bug if len(rows) else False)
        line = self._format_prior("project context", bugs, total)
        if line:
            evidence.append(line)

        if component:
            bugs, total = self._loo(self.component_stats, (project, component), current_is_bug if len(rows) else False)
            line = self._format_prior("project-component context", bugs, total)
            if line:
                evidence.append(line)

        severity = str(issue.get("severity") or "").casefold()
        if severity:
            # Resolve the current training-group count and leave the target record out.
            key = (source, severity)
            stat_bugs, stat_count = self.severity_stats.loc[key].tolist() if key in self.severity_stats.index else (0, 0)
            if len(rows):
                stat_bugs, stat_count = int(stat_bugs) - int(current_is_bug), int(stat_count) - 1
            line = self._format_prior("source-severity context", int(stat_bugs), int(stat_count))
            if line:
                evidence.append(line)

        evidence.append(f"[KG identity] {source}::{project}::{issue_id}; evidence source: frozen training split only.")
        return evidence


_kg_retriever_instance: KnowledgeGraphRetriever | None = None


def get_kg_retriever() -> KnowledgeGraphRetriever:
    global _kg_retriever_instance
    if _kg_retriever_instance is None:
        _kg_retriever_instance = KnowledgeGraphRetriever()
    return _kg_retriever_instance
