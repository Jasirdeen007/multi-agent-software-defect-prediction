from __future__ import annotations

import json
from pathlib import Path
from typing import Any
import networkx as nx
import pandas as pd


class KnowledgeGraphRetriever:
    """Retrieves structured relational evidence from the Knowledge Graph for agent reasoning."""

    def __init__(self, dataset_path: str = "data/processed/audit_sample.parquet"):
        self.dataset_path = Path(dataset_path)
        self.df = None
        self._load_data()

    def _load_data(self):
        if self.dataset_path.exists():
            cols = ["source", "project", "issue_id", "component", "severity", "priority", "original_label"]
            self.df = pd.read_parquet(self.dataset_path, columns=cols)
            self.df["issue_id"] = self.df["issue_id"].astype(str)

    def retrieve_evidence_for_issue(self, issue: dict[str, Any]) -> list[str]:
        """Returns structured relational evidence strings for an issue from the Knowledge Graph."""
        if self.df is None:
            return ["Knowledge Graph index not available."]

        evidence = []
        source = str(issue.get("source", "")).strip()
        project = str(issue.get("project", "")).strip()
        issue_id = str(issue.get("issue_id", "")).strip()
        comp = issue.get("component")
        sev = issue.get("severity")
        pri = issue.get("priority")

        # 1. Project-level defect prior
        proj_matches = self.df[self.df["project"].str.lower() == project.lower()]
        if len(proj_matches) > 0:
            bug_count = (proj_matches["original_label"].str.lower() == "bug").sum()
            rate = bug_count / len(proj_matches) * 100.0
            evidence.append(
                f"[KG Project Context] Project '{project}' has a historical defect rate of {rate:.1f}% "
                f"({bug_count}/{len(proj_matches)} sample issues classified as bugs)."
            )

        # 2. Component-level defect prior
        if comp and pd.notna(comp) and str(comp).strip():
            comp_name = str(comp).strip()
            comp_matches = self.df[
                (self.df["project"].str.lower() == project.lower()) &
                (self.df["component"].astype(str).str.lower() == comp_name.lower())
            ]
            if len(comp_matches) > 0:
                c_bugs = (comp_matches["original_label"].str.lower() == "bug").sum()
                c_rate = c_bugs / len(comp_matches) * 100.0
                evidence.append(
                    f"[KG Component Cluster] Component '{comp_name}' exhibits {c_rate:.1f}% defect density "
                    f"({c_bugs}/{len(comp_matches)} issues are confirmed defects)."
                )

        # 3. Source-level Severity / Priority co-occurrence
        if sev and pd.notna(sev) and str(sev).strip():
            sev_name = str(sev).strip().lower()
            sev_matches = self.df[
                (self.df["source"].str.lower() == source.lower()) &
                (self.df["severity"].astype(str).str.lower() == sev_name)
            ]
            if len(sev_matches) > 0:
                s_bugs = (sev_matches["original_label"].str.lower() == "bug").sum()
                s_rate = s_bugs / len(sev_matches) * 100.0
                evidence.append(
                    f"[KG Metadata Rule] Severity '{sev_name}' in {source} correlates with defects in {s_rate:.1f}% of cases."
                )

        # 4. Topology provenance
        evidence.append(
            f"[KG Topology Node] Composite Entity URI: '{source.upper()}::{project}::{issue_id}'."
        )

        return evidence


# Singleton instance for fast retrieval
_kg_retriever_instance = None

def get_kg_retriever() -> KnowledgeGraphRetriever:
    global _kg_retriever_instance
    if _kg_retriever_instance is None:
        _kg_retriever_instance = KnowledgeGraphRetriever()
    return _kg_retriever_instance
