from __future__ import annotations

import argparse
import json
from pathlib import Path

import networkx as nx
import pandas as pd


def build_training_kg(dataset_path: str, output_dir: str) -> dict[str, int]:
    """Build a compact, label-statistics KG from the frozen training split only."""
    frame = pd.read_parquet(
        dataset_path,
        columns=["source", "project", "component", "severity", "original_label"],
    )
    graph = nx.DiGraph()
    graph.add_node("dataset:train", node_type="dataset", split="train", records=len(frame))
    frame["_bug"] = frame["original_label"].astype(str).str.casefold().eq("bug")

    def add_group_nodes(keys: list[str], kind: str, edge: str) -> None:
        usable = frame.dropna(subset=keys).copy()
        usable = usable[usable[keys].astype(str).apply(lambda col: col.str.strip().ne("")).all(axis=1)]
        stats = usable.groupby(keys, dropna=False)["_bug"].agg(["sum", "count"])
        for key, row in stats.iterrows():
            values = key if isinstance(key, tuple) else (key,)
            node_id = kind + ":" + ":".join(str(v).strip() for v in values)
            bug_count, count = int(row["sum"]), int(row["count"])
            graph.add_node(
                node_id,
                node_type=kind,
                values=[str(v) for v in values],
                training_count=count,
                original_bug_count=bug_count,
                original_bug_rate=bug_count / count,
            )
            graph.add_edge("dataset:train", node_id, edge_type=edge)

    add_group_nodes(["source"], "source", "HAS_SOURCE")
    add_group_nodes(["source", "project"], "project", "HAS_PROJECT")
    add_group_nodes(["source", "project", "component"], "component", "HAS_COMPONENT")
    add_group_nodes(["source", "severity"], "severity", "HAS_SEVERITY")

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    graph_path = out / "training_kg.json"
    summary_path = out / "training_kg_summary.json"
    graph_path.write_text(json.dumps(nx.node_link_data(graph), ensure_ascii=False), encoding="utf-8")
    summary = {
        "source_dataset": dataset_path,
        "split": "frozen training only",
        "records": len(frame),
        "nodes": graph.number_of_nodes(),
        "edges": graph.number_of_edges(),
        "node_types": {},
    }
    for _, attrs in graph.nodes(data=True):
        node_type = attrs["node_type"]
        summary["node_types"][node_type] = summary["node_types"].get(node_type, 0) + 1
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="data/processed/fixed_splits/train.parquet")
    parser.add_argument("--output-dir", default="outputs/knowledge_graph/train_only")
    args = parser.parse_args()
    print(json.dumps(build_training_kg(args.dataset, args.output_dir), indent=2))
