from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any
import networkx as nx
import pandas as pd

from src.knowledge_graph.schema import EdgeType, NodeType


class KnowledgeGraphBuilder:
    """Constructs a typed Knowledge Graph from the canonical audit sample (AGENTS.md Section 24)."""

    def __init__(self, dataset_path: str = "data/processed/audit_sample.parquet"):
        self.dataset_path = Path(dataset_path)
        self.graph = nx.DiGraph()

    def build(self) -> nx.DiGraph:
        """Parses the audit sample and constructs nodes and relational edges."""
        if not self.dataset_path.exists():
            raise FileNotFoundError(f"Audit sample parquet not found at: {self.dataset_path}")

        print(f"Loading audit sample from {self.dataset_path}...")
        df = pd.read_parquet(self.dataset_path)
        total_issues = len(df)
        print(f"Constructing Knowledge Graph from {total_issues} issue records...")

        # Track entity nodes to avoid duplicate insertions
        projects_seen = set()
        sources_seen = set()
        components_seen = set()
        severities_seen = set()
        priorities_seen = set()
        statuses_seen = set()
        resolutions_seen = set()

        for _, row in df.iterrows():
            source = str(row.get("source", "")).strip()
            project = str(row.get("project", "")).strip()
            issue_id = str(row.get("issue_id", "")).strip()
            orig_label = str(row.get("original_label", "")).strip().lower()

            if not source or not project or not issue_id:
                continue

            # Composite issue node ID: (source, project, issue_id)
            issue_node_id = f"issue:{source}:{project}:{issue_id}"
            self.graph.add_node(
                issue_node_id,
                node_type=NodeType.ISSUE.value,
                source=source,
                project=project,
                issue_id=issue_id,
                original_label=orig_label,
                text_len=len(str(row.get("model_text", ""))),
            )

            # Project Node
            project_node_id = f"project:{project}"
            if project_node_id not in projects_seen:
                self.graph.add_node(project_node_id, node_type=NodeType.PROJECT.value, name=project)
                projects_seen.add(project_node_id)
            self.graph.add_edge(issue_node_id, project_node_id, edge_type=EdgeType.BELONGS_TO.value)

            # Source Node
            source_node_id = f"source:{source}"
            if source_node_id not in sources_seen:
                self.graph.add_node(source_node_id, node_type=NodeType.SOURCE.value, name=source)
                sources_seen.add(source_node_id)
            self.graph.add_edge(issue_node_id, source_node_id, edge_type=EdgeType.TRACKED_IN.value)

            # Component Node (scoped by project to maintain domain boundaries)
            comp = row.get("component")
            if pd.notna(comp) and str(comp).strip():
                clean_comp = str(comp).strip()
                comp_node_id = f"component:{project}:{clean_comp}"
                if comp_node_id not in components_seen:
                    self.graph.add_node(comp_node_id, node_type=NodeType.COMPONENT.value, name=clean_comp, project=project)
                    components_seen.add(comp_node_id)
                self.graph.add_edge(issue_node_id, comp_node_id, edge_type=EdgeType.HAS_COMPONENT.value)

            # Severity Node
            sev = row.get("severity")
            if pd.notna(sev) and str(sev).strip():
                clean_sev = str(sev).strip().lower()
                sev_node_id = f"severity:{clean_sev}"
                if sev_node_id not in severities_seen:
                    self.graph.add_node(sev_node_id, node_type=NodeType.SEVERITY.value, name=clean_sev)
                    severities_seen.add(sev_node_id)
                self.graph.add_edge(issue_node_id, sev_node_id, edge_type=EdgeType.HAS_SEVERITY.value)

            # Priority Node
            pri = row.get("priority")
            if pd.notna(pri) and str(pri).strip():
                clean_pri = str(pri).strip().lower()
                pri_node_id = f"priority:{clean_pri}"
                if pri_node_id not in priorities_seen:
                    self.graph.add_node(pri_node_id, node_type=NodeType.PRIORITY.value, name=clean_pri)
                    priorities_seen.add(pri_node_id)
                self.graph.add_edge(issue_node_id, pri_node_id, edge_type=EdgeType.HAS_PRIORITY.value)

            # Status Node
            st = row.get("status")
            if pd.notna(st) and str(st).strip():
                clean_st = str(st).strip().lower()
                st_node_id = f"status:{clean_st}"
                if st_node_id not in statuses_seen:
                    self.graph.add_node(st_node_id, node_type=NodeType.STATUS.value, name=clean_st)
                    statuses_seen.add(st_node_id)
                self.graph.add_edge(issue_node_id, st_node_id, edge_type=EdgeType.HAS_STATUS.value)

            # Resolution Node
            res = row.get("resolution")
            if pd.notna(res) and str(res).strip():
                clean_res = str(res).strip().lower()
                res_node_id = f"resolution:{clean_res}"
                if res_node_id not in resolutions_seen:
                    self.graph.add_node(res_node_id, node_type=NodeType.RESOLUTION.value, name=clean_res)
                    resolutions_seen.add(res_node_id)
                self.graph.add_edge(issue_node_id, res_node_id, edge_type=EdgeType.HAS_RESOLUTION.value)

        print(f"Graph construction complete! Nodes: {self.graph.number_of_nodes()}, Edges: {self.graph.number_of_edges()}")
        return self.graph

    def compute_summary_statistics(self) -> dict[str, Any]:
        """Calculates node and edge counts by category."""
        node_counts: dict[str, int] = {}
        for _, data in self.graph.nodes(data=True):
            nt = data.get("node_type", "Unknown")
            node_counts[nt] = node_counts.get(nt, 0) + 1

        edge_counts: dict[str, int] = {}
        for _, _, data in self.graph.edges(data=True):
            et = data.get("edge_type", "Unknown")
            edge_counts[et] = edge_counts.get(et, 0) + 1

        return {
            "total_nodes": self.graph.number_of_nodes(),
            "total_edges": self.graph.number_of_edges(),
            "node_distribution": node_counts,
            "edge_distribution": edge_counts,
        }

    def save(self, output_dir: str = "outputs/knowledge_graph") -> None:
        """Serializes the graph and summary statistics to disk."""
        out_path = Path(output_dir)
        out_path.mkdir(parents=True, exist_ok=True)

        summary = self.compute_summary_statistics()
        with open(out_path / "kg_summary.json", "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)

        # Export node-link graph data for fast query loading
        data = nx.node_link_data(self.graph)
        with open(out_path / "kg_audit_sample.json", "w", encoding="utf-8") as f:
            json.dump(data, f)

        print(f"Saved Knowledge Graph artifacts to {out_path}:")
        print(f"  - Summary:    {out_path / 'kg_summary.json'}")
        print(f"  - Graph JSON: {out_path / 'kg_audit_sample.json'}")


def run_kg_builder(dataset_path: str = "data/processed/audit_sample.parquet", output_dir: str = "outputs/knowledge_graph"):
    builder = KnowledgeGraphBuilder(dataset_path=dataset_path)
    builder.build()
    builder.save(output_dir=output_dir)
    stats = builder.compute_summary_statistics()
    print("\n--- Knowledge Graph Summary ---")
    print(f"  Total Nodes: {stats['total_nodes']}")
    print(f"  Total Edges: {stats['total_edges']}")
    print(f"  Nodes by Type: {stats['node_distribution']}")
    print(f"  Edges by Type: {stats['edge_distribution']}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Construct Software Defect Prediction Knowledge Graph")
    parser.add_argument("--dataset-path", default="data/processed/audit_sample.parquet")
    parser.add_argument("--output-dir", default="outputs/knowledge_graph")
    args = parser.parse_args()

    run_kg_builder(dataset_path=args.dataset_path, output_dir=args.output_dir)
