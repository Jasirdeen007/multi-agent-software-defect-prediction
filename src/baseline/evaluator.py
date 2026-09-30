from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    brier_score_loss,
    classification_report,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)


def evaluate_predictions(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: np.ndarray,
    target_names: list[str] | None = None,
) -> dict[str, Any]:
    if target_names is None:
        target_names = ["non-bug", "bug"]

    metrics: dict[str, Any] = {
        "accuracy": round(float(accuracy_score(y_true, y_pred)), 4),
        "balanced_accuracy": round(float(balanced_accuracy_score(y_true, y_pred)), 4),
        "precision_bug": round(float(precision_score(y_true, y_pred, pos_label=1, zero_division=0)), 4),
        "recall_bug": round(float(recall_score(y_true, y_pred, pos_label=1, zero_division=0)), 4),
        "f1_bug": round(float(f1_score(y_true, y_pred, pos_label=1, zero_division=0)), 4),
        "precision_non_bug": round(float(precision_score(y_true, y_pred, pos_label=0, zero_division=0)), 4),
        "recall_non_bug": round(float(recall_score(y_true, y_pred, pos_label=0, zero_division=0)), 4),
        "f1_non_bug": round(float(f1_score(y_true, y_pred, pos_label=0, zero_division=0)), 4),
        "macro_f1": round(float(f1_score(y_true, y_pred, average="macro", zero_division=0)), 4),
        "weighted_f1": round(float(f1_score(y_true, y_pred, average="weighted", zero_division=0)), 4),
        "roc_auc": round(float(roc_auc_score(y_true, y_proba)), 4),
        "matthews_corrcoef": round(float(matthews_corrcoef(y_true, y_pred)), 4),
        "brier_score": round(float(brier_score_loss(y_true, y_proba)), 4),
    }

    report_str = classification_report(y_true, y_pred, target_names=target_names, digits=4)
    return {"metrics": metrics, "classification_report": report_str}


def plot_and_save_figures(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: np.ndarray,
    figures_dir: str | Path,
) -> None:
    fig_path = Path(figures_dir)
    fig_path.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid")

    # 1. Confusion Matrix
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["Non-Bug (0)", "Bug (1)"],
        yticklabels=["Non-Bug (0)", "Bug (1)"],
    )
    plt.title("Confusion Matrix - Baseline Model")
    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")
    plt.tight_layout()
    plt.savefig(fig_path / "confusion_matrix.png", dpi=300)
    plt.close()

    # 2. ROC Curve
    fpr, tpr, _ = roc_curve(y_true, y_proba)
    auc_val = roc_auc_score(y_true, y_proba)
    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, color="darkorange", lw=2, label=f"ROC curve (AUC = {auc_val:.4f})")
    plt.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--", label="Random Chance")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Receiver Operating Characteristic (ROC)")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(fig_path / "roc_curve.png", dpi=300)
    plt.close()

    # 3. Precision-Recall Curve
    precision, recall, _ = precision_recall_curve(y_true, y_proba)
    plt.figure(figsize=(6, 5))
    plt.plot(recall, precision, color="teal", lw=2, label="Precision-Recall curve")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve")
    plt.legend(loc="lower left")
    plt.tight_layout()
    plt.savefig(fig_path / "precision_recall_curve.png", dpi=300)
    plt.close()

    # 4. Calibration Curve
    prob_true, prob_pred = calibration_curve(y_true, y_proba, n_bins=10)
    plt.figure(figsize=(6, 5))
    plt.plot(prob_pred, prob_true, marker="o", lw=2, label="Baseline Model")
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Perfectly Calibrated")
    plt.xlabel("Mean Predicted Probability")
    plt.ylabel("Fraction of Positives")
    plt.title("Reliability / Calibration Curve")
    plt.legend(loc="upper left")
    plt.tight_layout()
    plt.savefig(fig_path / "calibration_curve.png", dpi=300)
    plt.close()

    print(f"Generated and saved 4 evaluation plots to: {fig_path}")


def save_evaluation_results(
    eval_dict: dict[str, Any],
    results_dir: str | Path,
) -> None:
    res_path = Path(results_dir)
    res_path.mkdir(parents=True, exist_ok=True)

    metrics_file = res_path / "metrics.json"
    with open(metrics_file, "w", encoding="utf-8") as f:
        json.dump(eval_dict["metrics"], f, indent=2)

    report_file = res_path / "classification_report.txt"
    with open(report_file, "w", encoding="utf-8") as f:
        f.write(eval_dict["classification_report"])

    print(f"Saved metrics to: {metrics_file}")
    print(f"Saved classification report to: {report_file}")
