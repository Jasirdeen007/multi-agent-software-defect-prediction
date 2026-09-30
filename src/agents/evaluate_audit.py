from __future__ import annotations

import json
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def evaluate_audit_results(
    audited_parquet: str = "data/processed/audited_labels.parquet",
    figures_dir: str = "outputs/figures/audited",
    results_dir: str = "outputs/results/audited",
) -> None:
    """Analyzes audit outcomes: flip rate, agreement distribution, and confidence calibration."""
    p_path = Path(audited_parquet)
    if not p_path.exists():
        print(f"Audited parquet not found at {p_path}")
        return

    df = pd.read_parquet(p_path)
    total = len(df)
    print(f"=== Multi-Agent Audit Analysis (N = {total}) ===")

    # 1. Flip Rate Analysis
    flips = df["label_flipped"].sum()
    flip_rate = flips / total if total > 0 else 0.0

    orig_dist = df["original_label"].value_counts(normalize=True).to_dict()
    judge_dist = df["judge_label"].value_counts(normalize=True).to_dict()

    # Flip matrix
    flip_matrix = pd.crosstab(df["original_label"], df["judge_label"], margins=True)
    print("\n--- Label Revision (Flip) Matrix ---")
    print(flip_matrix)

    # 2. Agreement Statistics
    mean_agreement = df["agent_agreement"].mean()
    unanimous_rate = (df["agent_agreement"] == 1.0).mean()
    majority_rate = (df["agent_agreement"] < 1.0).mean()

    # 3. Average Confidences
    policy_conf = df["policy_confidence"].mean()
    data_conf = df["data_confidence"].mean()
    pattern_conf = df["pattern_confidence"].mean()
    judge_conf = df["judge_confidence"].mean()

    summary_metrics = {
        "total_audited": int(total),
        "total_flips": int(flips),
        "flip_rate": round(float(flip_rate), 4),
        "mean_agent_agreement": round(float(mean_agreement), 4),
        "unanimous_consensus_rate": round(float(unanimous_rate), 4),
        "majority_disagreement_rate": round(float(majority_rate), 4),
        "mean_confidences": {
            "policy_agent": round(float(policy_conf), 4),
            "data_agent": round(float(data_conf), 4),
            "pattern_agent": round(float(pattern_conf), 4),
            "judge_agent": round(float(judge_conf), 4),
        },
        "original_distribution": {k: round(float(v), 4) for k, v in orig_dist.items()},
        "audited_distribution": {k: round(float(v), 4) for k, v in judge_dist.items()},
    }

    out_res = Path(results_dir)
    out_res.mkdir(parents=True, exist_ok=True)
    with open(out_res / "audit_metrics.json", "w", encoding="utf-8") as f:
        json.dump(summary_metrics, f, indent=2)

    # 4. Generate Diagnostic Figures
    out_fig = Path(figures_dir)
    out_fig.mkdir(parents=True, exist_ok=True)

    # Figure 1: Agreement Breakdown
    plt.figure(figsize=(6, 5))
    agreement_counts = df["agent_agreement"].value_counts().sort_index()
    labels = ["Majority (2/3)" if x < 1.0 else "Unanimous (3/3)" for x in agreement_counts.index]
    plt.bar(labels, agreement_counts.values, color=["#ED8936", "#38A169"], edgecolor="black", width=0.5)
    plt.title("Multi-Agent Agreement Distribution", fontsize=13, pad=12)
    plt.ylabel("Number of Issues")
    plt.grid(axis="y", linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(out_fig / "agent_agreement_distribution.png", dpi=300)
    plt.close()

    # Figure 2: Label Transition (Original vs Audited)
    plt.figure(figsize=(7, 5))
    categories = ["Bug", "Non-Bug"]
    orig_vals = [orig_dist.get("bug", 0) * 100, orig_dist.get("non-bug", 0) * 100]
    audit_vals = [judge_dist.get("bug", 0) * 100, judge_dist.get("non-bug", 0) * 100]
    
    x = np.arange(len(categories))
    width = 0.35
    plt.bar(x - width/2, orig_vals, width, label="Original BugHub Labels", color="#3182CE")
    plt.bar(x + width/2, audit_vals, width, label="Agent-Audited Labels", color="#805AD5")
    plt.ylabel("Percentage (%)")
    plt.title("Class Distribution: Original vs. Agent-Audited", fontsize=13, pad=12)
    plt.xticks(x, categories)
    plt.legend()
    plt.grid(axis="y", linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(out_fig / "label_transition_comparison.png", dpi=300)
    plt.close()

    print(f"\n--> Saved audit metrics to: {out_res / 'audit_metrics.json'}")
    print(f"--> Saved audit figures to: {out_fig}")


if __name__ == "__main__":
    evaluate_audit_results()
