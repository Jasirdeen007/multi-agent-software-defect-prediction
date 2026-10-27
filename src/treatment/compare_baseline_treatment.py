from __future__ import annotations

import argparse
import json
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np


def compare_baseline_and_treatment(
    baseline_metrics_path: str = "outputs/results/baseline/full/metrics.json",
    treatment_metrics_path: str = "outputs/results/treatment/metrics.json",
    output_dir: str = "outputs/results/comparison",
    figures_dir: str = "outputs/figures/comparison",
) -> dict:
    """Compares baseline model (BugHub labels) vs treatment model (Multi-Agent audited labels).

    Generates comparison summary tables, delta calculations, and comparative figures.
    """
    print("\n" + "=" * 60)
    print("=== Multi-Agent SDP: Baseline vs Treatment Comparison ===")
    print("=" * 60)

    base_p = Path(baseline_metrics_path)
    treat_p = Path(treatment_metrics_path)

    if not base_p.exists():
        raise FileNotFoundError(f"Baseline metrics not found at: {base_p}")
    if not treat_p.exists():
        raise FileNotFoundError(f"Treatment metrics not found at: {treat_p}")

    with open(base_p, "r", encoding="utf-8") as f:
        baseline = json.load(f)
    with open(treat_p, "r", encoding="utf-8") as f:
        treatment = json.load(f)

    baseline_test_hash = baseline.get("test_identity_sha256")
    treatment_test_hash = treatment.get("test_identity_sha256")
    if not baseline_test_hash or baseline_test_hash != treatment_test_hash:
        raise ValueError("Baseline and treatment metrics do not prove use of the same fixed test identities; refusing comparison.")
    if baseline.get("test_records") != treatment.get("test_records"):
        raise ValueError("Baseline and treatment evaluated different test-set sizes; refusing comparison.")

    # Compute metric deltas
    comparison_table = []
    comparison_dict = {}

    metric_keys = [
        ("accuracy", "Accuracy", False),
        ("balanced_accuracy", "Balanced Accuracy", False),
        ("macro_f1", "Macro F1", False),
        ("roc_auc", "ROC-AUC", False),
        ("matthews_corrcoef", "Matthews Corr (MCC)", False),
        ("precision_bug", "Precision (Bug)", False),
        ("recall_bug", "Recall (Bug)", False),
        ("f1_bug", "F1 (Bug)", False),
        ("precision_non_bug", "Precision (Non-Bug)", False),
        ("recall_non_bug", "Recall (Non-Bug)", False),
        ("f1_non_bug", "F1 (Non-Bug)", False),
        ("brier_score", "Brier Score", True),  # lower is better
        ("overfitting_gap", "Overfitting Gap", True),  # lower is better
    ]

    for key, display_name, lower_is_better in metric_keys:
        base_val = baseline.get(key)
        treat_val = treatment.get(key)

        if base_val is not None and treat_val is not None:
            delta = treat_val - base_val
            pct_change = (delta / abs(base_val) * 100.0) if base_val != 0 else 0.0
            improved = (delta < 0) if lower_is_better else (delta > 0)

            comparison_dict[key] = {
                "display_name": display_name,
                "baseline": round(float(base_val), 4),
                "treatment": round(float(treat_val), 4),
                "delta": round(float(delta), 4),
                "pct_change": round(float(pct_change), 2),
                "improved": bool(improved),
                "lower_is_better": lower_is_better,
            }

            sign = "+" if delta >= 0 else ""
            status = "[UP]" if improved else ("[SAME]" if delta == 0 else "[DOWN]")
            comparison_table.append(
                f"{display_name:<25} | Base: {base_val:.4f} | Treat: {treat_val:.4f} | Delta: {sign}{delta:.4f} ({sign}{pct_change:.1f}%) {status}"
            )


    print("\n--- Comparative Metrics Breakdown ---")
    for row in comparison_table:
        print(row)

    # Save comparison JSON
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    summary_file = out_dir / "comparison_summary.json"
    with open(summary_file, "w", encoding="utf-8") as f:
        # Attach audit scale and actual observed API usage when available.
        audit_metrics_path = Path("outputs/results/audited_train_1000_independent_calls/audit_metrics.json")
        audit_usage_path = Path("outputs/results/audited_train_1000_independent_calls/api_usage.json")
        treatment_summary_path = Path("outputs/results/treatment/treatment_dataset_summary_current.json")
        context = {}
        for name, path in (("audit", audit_metrics_path), ("api_usage", audit_usage_path), ("treatment_dataset", treatment_summary_path)):
            if path.exists():
                with open(path, "r", encoding="utf-8") as source:
                    context[name] = json.load(source)
        comparison_dict["experiment_context"] = context
        json.dump(comparison_dict, f, indent=2)
    print(f"\nSaved comparison summary to: {summary_file}")

    # Plot Comparison Bar Chart
    fig_dir = Path(figures_dir)
    fig_dir.mkdir(parents=True, exist_ok=True)

    chart_metrics = ["accuracy", "balanced_accuracy", "macro_f1", "roc_auc", "matthews_corrcoef"]
    chart_names = ["Accuracy", "Bal Acc", "Macro F1", "ROC-AUC", "MCC"]
    base_scores = [baseline[m] for m in chart_metrics]
    treat_scores = [treatment[m] for m in chart_metrics]

    x = np.arange(len(chart_metrics))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 6))
    rects1 = ax.bar(x - width/2, base_scores, width, label="Baseline (BugHub Original)", color="#3b82f6", alpha=0.9)
    rects2 = ax.bar(x + width/2, treat_scores, width, label="Treatment (Agent-Audited)", color="#10b981", alpha=0.9)

    ax.set_ylabel("Score (0.0 - 1.0)", fontsize=12)
    ax.set_title("Software Defect Prediction: Baseline vs Agent-Audited Treatment", fontsize=14, fontweight="bold", pad=15)
    ax.set_xticks(x)
    ax.set_xticklabels(chart_names, fontsize=11)
    ax.set_ylim(0, 1.05)
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    ax.legend(loc="lower right", fontsize=11)

    # Add text labels on bars
    for rect in rects1:
        h = rect.get_height()
        ax.annotate(f"{h:.3f}", xy=(rect.get_x() + rect.get_width()/2, h), xytext=(0, 3),
                    textcoords="offset points", ha="center", va="bottom", fontsize=9)
    for rect in rects2:
        h = rect.get_height()
        ax.annotate(f"{h:.3f}", xy=(rect.get_x() + rect.get_width()/2, h), xytext=(0, 3),
                    textcoords="offset points", ha="center", va="bottom", fontsize=9, fontweight="bold")

    plt.tight_layout()
    chart_path = fig_dir / "baseline_vs_treatment_comparison.png"
    plt.savefig(chart_path, dpi=300)
    plt.close()
    print(f"Saved comparison figure to: {chart_path}")

    return comparison_dict


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compare Baseline vs Treatment Models")
    parser.add_argument("--baseline-metrics", default="outputs/results/baseline/full/metrics.json")
    parser.add_argument("--treatment-metrics", default="outputs/results/treatment/metrics.json")
    parser.add_argument("--output-dir", default="outputs/results/comparison")
    parser.add_argument("--figures-dir", default="outputs/figures/comparison")
    args = parser.parse_args()

    compare_baseline_and_treatment(
        baseline_metrics_path=args.baseline_metrics,
        treatment_metrics_path=args.treatment_metrics,
        output_dir=args.output_dir,
        figures_dir=args.figures_dir,
    )
