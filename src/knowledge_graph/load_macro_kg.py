from __future__ import annotations

import csv
import json
from pathlib import Path
import re
from typing import Any, Dict, List, Optional, Tuple

import networkx as nx


def parse_cypher_entity(text: str) -> Tuple[Optional[str], Dict[str, Any]]:
    """Parses a Cypher node string e.g. (:Component {name: 'Search', bug_rate: 0.705})."""
    m = re.match(r"\(:([A-Za-z_]+)\s*\{(.*)\}\)", text.strip())
    if not m:
        return None, {}
    label, props_str = m.group(1), m.group(2)
    props: Dict[str, Any] = {}
    for match in re.finditer(r"([A-Za-z_]+):\s*(\[[^\]]*\]|[^,]+)", props_str):
        k = match.group(1).strip()
        v = match.group(2).strip()
        if v.startswith("[") and v.endswith("]"):
            props[k] = [x.strip() for x in v[1:-1].split(",") if x.strip()]
        else:
            try:
                if "." in v:
                    props[k] = float(v)
                else:
                    props[k] = int(v)
            except ValueError:
                props[k] = v
    return label, props


def parse_cypher_rel(text: str) -> Tuple[str, Dict[str, Any]]:
    """Parses a Cypher relationship string e.g. [:DEFECT_CLUSTER {avg_bug_rate: 0.78}]."""
    m = re.match(r"\[:([A-Za-z_]+)(?:\s*\{(.*)\})?\]", text.strip())
    if not m:
        return "RELATED_TO", {}
    rel, props_str = m.group(1), m.group(2)
    props: Dict[str, Any] = {}
    if props_str:
        for match in re.finditer(r"([A-Za-z_]+):\s*(\[[^\]]*\]|[^,]+)", props_str):
            k = match.group(1).strip()
            v = match.group(2).strip()
            if v.startswith("[") and v.endswith("]"):
                props[k] = [x.strip() for x in v[1:-1].split(",") if x.strip()]
            else:
                try:
                    if "." in v:
                        props[k] = float(v)
                    else:
                        props[k] = int(v)
                except ValueError:
                    props[k] = v
    return rel, props


class MacroKnowledgeGraph:
    """Ingests and queries the macro-ontology knowledge graph exported from Neo4j/Cypher.
    
    Contains 5,000 statistical triples computed across 926,461 BugHub records and 79 projects.
    """

    def __init__(self, csv_path: str = "data/knowledge_graph/kg_macro_triples.csv"):
        self.csv_path = Path(csv_path)
        self.graph = nx.MultiDiGraph()
        self.projects: Dict[str, Dict[str, Any]] = {}
        self.components: Dict[str, Dict[str, Any]] = {}
        self.sources: Dict[str, Dict[str, Any]] = {}
        self.priorities: Dict[str, Dict[str, Any]] = {}
        if self.csv_path.exists():
            self.load()

    def load(self) -> None:
        """Parses the CSV and populates the NetworkX graph and lookup caches."""
        if not self.csv_path.exists():
            raise FileNotFoundError(f"Macro KG CSV not found at {self.csv_path}")

        print(f"Loading Macro Knowledge Graph from {self.csv_path}...")
        triple_count = 0
        with open(self.csv_path, "r", encoding="utf-8") as f:
            reader = csv.reader(f)
            _ = next(reader)  # Header: n, r, m
            for row in reader:
                if len(row) < 3:
                    continue
                n_raw, r_raw, m_raw = row[0], row[1], row[2]
                s_label, s_props = parse_cypher_entity(n_raw)
                rel_type, r_props = parse_cypher_rel(r_raw)
                t_label, t_props = parse_cypher_entity(m_raw)

                if not s_label or not t_label:
                    continue

                s_name = str(s_props.get("name", s_label))
                t_name = str(t_props.get("name", t_label))

                s_id = f"{s_label}::{s_name}"
                t_id = f"{t_label}::{t_name}"

                self.graph.add_node(s_id, label=s_label, **s_props)
                self.graph.add_node(t_id, label=t_label, **t_props)
                self.graph.add_edge(s_id, t_id, type=rel_type, **r_props)
                triple_count += 1

                # Update lookup indexes
                if s_label == "Project":
                    self.projects[s_name.lower()] = s_props
                elif s_label == "Component":
                    self.components[s_name.lower()] = s_props
                elif s_label == "Source":
                    self.sources[s_name.lower()] = s_props
                elif s_label == "Priority":
                    self.priorities[s_name.lower()] = s_props

                if t_label == "Project":
                    self.projects[t_name.lower()] = t_props
                elif t_label == "Component":
                    self.components[t_name.lower()] = t_props
                elif t_label == "Source":
                    self.sources[t_name.lower()] = t_props
                elif t_label == "Priority":
                    self.priorities[t_name.lower()] = t_props

        print(
            f"Macro KG loaded: {self.graph.number_of_nodes()} nodes, "
            f"{self.graph.number_of_edges()} edges ({triple_count} triples parsed)."
        )

    def get_project_prior(self, project_name: str) -> Optional[Dict[str, Any]]:
        """Returns macro defect stats for a project (bug_rate, bug_count, total count)."""
        return self.projects.get(str(project_name).strip().lower())

    def get_component_prior(self, component_name: str) -> Optional[Dict[str, Any]]:
        """Returns macro defect stats for a component (bug_rate, bug_count, total count)."""
        return self.components.get(str(component_name).strip().lower())

    def get_defect_clusters(self, component_name: str) -> List[Dict[str, Any]]:
        """Finds components connected via DEFECT_CLUSTER edges."""
        comp_id = f"Component::{component_name}"
        clusters = []
        if comp_id in self.graph:
            for _, target, data in self.graph.out_edges(comp_id, data=True):
                if data.get("type") == "DEFECT_CLUSTER":
                    target_props = self.graph.nodes[target]
                    clusters.append({
                        "component": target_props.get("name"),
                        "bug_rate": target_props.get("bug_rate"),
                        "avg_bug_rate": data.get("avg_bug_rate"),
                        "project_count": data.get("project_count"),
                    })
        return clusters

    def get_co_occurring_components(self, component_name: str) -> List[Dict[str, Any]]:
        """Finds components connected via CO_OCCURS_WITH edges."""
        comp_id = f"Component::{component_name}"
        co_occurring = []
        if comp_id in self.graph:
            for _, target, data in self.graph.out_edges(comp_id, data=True):
                if data.get("type") == "CO_OCCURS_WITH":
                    target_props = self.graph.nodes[target]
                    co_occurring.append({
                        "component": target_props.get("name"),
                        "bug_rate": target_props.get("bug_rate"),
                        "shared_projects": data.get("shared_projects"),
                        "project_count": data.get("project_count"),
                    })
        return co_occurring

    def get_shared_components(self, project_name: str) -> List[Dict[str, Any]]:
        """Finds other projects connected via SHARES_COMPONENT edges."""
        proj_id = f"Project::{project_name}"
        shared = []
        if proj_id in self.graph:
            for _, target, data in self.graph.out_edges(proj_id, data=True):
                if data.get("type") == "SHARES_COMPONENT":
                    target_props = self.graph.nodes[target]
                    shared.append({
                        "peer_project": target_props.get("name"),
                        "peer_bug_rate": target_props.get("bug_rate"),
                        "shared_count": data.get("shared_count"),
                        "shared_components": data.get("shared_components"),
                    })
        return shared

    def export_summary(self, output_path: str = "outputs/knowledge_graph/kg_macro_summary.json") -> Dict[str, Any]:
        """Exports statistical summary of the macro knowledge graph."""
        rel_counts: Dict[str, int] = {}
        for _, _, data in self.graph.edges(data=True):
            r_type = data.get("type", "UNKNOWN")
            rel_counts[r_type] = rel_counts.get(r_type, 0) + 1

        summary = {
            "source_file": str(self.csv_path),
            "total_nodes": self.graph.number_of_nodes(),
            "total_edges": self.graph.number_of_edges(),
            "projects_count": len(self.projects),
            "components_count": len(self.components),
            "sources_count": len(self.sources),
            "priorities_count": len(self.priorities),
            "relationship_distribution": rel_counts,
        }
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)
        print(f"Saved Macro KG summary to {out}")
        return summary


_MACRO_KG_INSTANCE: Optional[MacroKnowledgeGraph] = None


def get_macro_kg() -> MacroKnowledgeGraph:
    """Singleton getter for the MacroKnowledgeGraph."""
    global _MACRO_KG_INSTANCE
    if _MACRO_KG_INSTANCE is None:
        _MACRO_KG_INSTANCE = MacroKnowledgeGraph()
    return _MACRO_KG_INSTANCE


if __name__ == "__main__":
    kg = get_macro_kg()
    summary = kg.export_summary()
    print("\n--- Macro KG Summary ---")
    print(json.dumps(summary, indent=2))

    # Test query
    sample_proj = "terraform"
    prior = kg.get_project_prior(sample_proj)
    print(f"\nSample Project Prior ('{sample_proj}'): {prior}")

    sample_comp = "Search"
    clusters = kg.get_defect_clusters(sample_comp)
    print(f"\nDefect Clusters for Component '{sample_comp}': {len(clusters)} clusters found")
    if clusters:
        print(f"  Example: {clusters[0]}")
