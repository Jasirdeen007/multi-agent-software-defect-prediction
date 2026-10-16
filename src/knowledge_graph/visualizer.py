from __future__ import annotations

import argparse
from pathlib import Path
import matplotlib.pyplot as plt
import networkx as nx

from src.knowledge_graph.builder import KnowledgeGraphBuilder


def visualize_kg_subgraph(
    dataset_path: str = "data/processed/audit_sample.parquet",
    output_dir: str = "outputs/figures/knowledge_graph",
    sample_issues_count: int = 4,
) -> None:
    """Renders a multi-project ego-network subgraph from the Knowledge Graph."""
    builder = KnowledgeGraphBuilder(dataset_path=dataset_path)
    G = builder.build()

    # Find sample issues from distinct projects
    sample_nodes = []
    projects_seen = set()

    for node, data in G.nodes(data=True):
        if data.get("node_type") == "Issue":
            proj = data.get("project")
            if proj not in projects_seen:
                sample_nodes.append(node)
                projects_seen.add(proj)
            if len(sample_nodes) >= sample_issues_count:
                break

    # Extract 1-hop ego networks around the sample issues
    subgraph_nodes = set(sample_nodes)
    for n in sample_nodes:
        # Successors (entities the issue points to: Project, Component, Severity, etc.)
        subgraph_nodes.update(G.successors(n))
        # Find 1 neighbor issue sharing the same component if available
        for target in G.successors(n):
            if G.nodes[target].get("node_type") == "Component":
                in_issues = [u for u in G.predecessors(target) if G.nodes[u].get("node_type") == "Issue" and u != n]
                if in_issues:
                    subgraph_nodes.add(in_issues[0])

    subG = G.subgraph(subgraph_nodes)
    print(f"Sub-network contains {subG.number_of_nodes()} nodes and {subG.number_of_edges()} edges.")

    # Color mapping by node type
    color_map = {
        "Issue": "#3b82f6",       # Blue
        "Project": "#10b981",     # Green
        "Source": "#8b5cf6",      # Purple
        "Component": "#f59e0b",   # Amber
        "Severity": "#ef4444",    # Red
        "Priority": "#ec4899",    # Pink
        "Status": "#6b7280",      # Gray
        "Resolution": "#14b8a6",  # Teal
    }

    node_colors = []
    node_sizes = []
    node_labels = {}

    for n, d in subG.nodes(data=True):
        nt = d.get("node_type", "Issue")
        node_colors.append(color_map.get(nt, "#94a3b8"))
        if nt == "Issue":
            node_sizes.append(450)
            node_labels[n] = f"Issue #{d.get('issue_id', '')[:8]}"
        elif nt in ("Project", "Source"):
            node_sizes.append(1200)
            node_labels[n] = d.get("name", n)
        else:
            node_sizes.append(700)
            name = d.get("name", n.split(":")[-1])
            node_labels[n] = f"{nt}:\n{name[:15]}"

    plt.figure(figsize=(14, 10))
    pos = nx.spring_layout(subG, k=1.2, iterations=100, seed=42)

    nx.draw_networkx_nodes(
        subG, pos,
        node_color=node_colors,
        node_size=node_sizes,
        alpha=0.92,
        edgecolors="#1e293b",
        linewidths=1.5,
    )

    nx.draw_networkx_edges(
        subG, pos,
        edge_color="#94a3b8",
        width=1.5,
        alpha=0.65,
        arrows=True,
        arrowsize=14,
        connectionstyle="arc3,rad=0.08",
    )

    nx.draw_networkx_labels(
        subG, pos,
        labels=node_labels,
        font_size=8,
        font_family="sans-serif",
        font_weight="bold",
    )

    # Legend handles
    import matplotlib.patches as mpatches
    legend_handles = [
        mpatches.Patch(color=c, label=nt)
        for nt, c in color_map.items()
    ]
    plt.legend(handles=legend_handles, loc="upper right", fontsize=10, framealpha=0.9)
    plt.title("Multi-Agent SDP: Knowledge Graph Ego-Subnetwork Evidence Structure", fontsize=14, fontweight="bold", pad=20)
    plt.axis("off")
    plt.tight_layout()

    out_p = Path(output_dir)
    out_p.mkdir(parents=True, exist_ok=True)
    out_file = out_p / "kg_ego_subgraph.png"
    plt.savefig(out_file, dpi=300)
    plt.close()
    print(f"Saved Knowledge Graph visualization to: {out_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Render Knowledge Graph Subgraph")
    parser.add_argument("--dataset-path", default="data/processed/audit_sample.parquet")
    parser.add_argument("--output-dir", default="outputs/figures/knowledge_graph")
    args = parser.parse_args()

    visualize_kg_subgraph(dataset_path=args.dataset_path, output_dir=args.output_dir)
