"""Paired uncertainty and error-disagreement analysis for baseline vs treatment.

Regenerates test predictions with XGBoost on CUDA, checks them against the
retained metric artifacts, then writes a stratified paired-bootstrap report
and exact McNemar tests for the shared frozen test set.
"""

from __future__ import annotations

import argparse
from datetime import date
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from scipy.stats import binomtest
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    brier_score_loss,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)


ROOT = Path(__file__).resolve().parents[2]
BASELINE_MODEL = ROOT / "kaggle_outputs/outputs/models/baseline/full/xgb_baseline.json"
TREATMENT_MODEL = ROOT / "outputs/models/treatment/xgb_treatment.json"
BASELINE_METRICS = ROOT / "outputs/results/baseline/full/metrics.json"
TREATMENT_METRICS = ROOT / "outputs/results/treatment/metrics.json"
TEST_DATA = ROOT / "data/processed/fixed_splits/test.parquet"
TEST_EMBEDDINGS = ROOT / "data/embeddings/test_embeddings_138949.npy"
SPLIT_MANIFEST = ROOT / "data/processed/fixed_splits/split_manifest.json"
REPORT_PATH = ROOT / "outputs/results/comparison/paired_statistical_analysis.md"

METRIC_FUNCTIONS = {
    "accuracy": lambda y, p, pred: accuracy_score(y, pred),
    "balanced_accuracy": lambda y, p, pred: balanced_accuracy_score(y, pred),
    "recall_bug": lambda y, p, pred: recall_score(y, pred, pos_label=1, zero_division=0),
    "recall_non_bug": lambda y, p, pred: recall_score(y, pred, pos_label=0, zero_division=0),
    "precision_bug": lambda y, p, pred: precision_score(y, pred, pos_label=1, zero_division=0),
    "precision_non_bug": lambda y, p, pred: precision_score(y, pred, pos_label=0, zero_division=0),
    "f1_bug": lambda y, p, pred: f1_score(y, pred, pos_label=1, zero_division=0),
    "f1_non_bug": lambda y, p, pred: f1_score(y, pred, pos_label=0, zero_division=0),
    "macro_f1": lambda y, p, pred: f1_score(y, pred, average="macro", zero_division=0),
    "matthews_corrcoef": lambda y, p, pred: matthews_corrcoef(y, pred),
    "brier_score": lambda y, p, pred: brier_score_loss(y, p),
    "roc_auc": lambda y, p, pred: roc_auc_score(y, p),
}


def identity_digest(frame: pd.DataFrame) -> str:
    payload = frame[["source", "project", "issue_id"]].astype(str).to_csv(
        index=False, header=False
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _confusion_metrics(y: np.ndarray, pred: np.ndarray, prob: np.ndarray) -> dict[str, float]:
    tp = int(np.count_nonzero((y == 1) & (pred == 1)))
    fn = int(np.count_nonzero((y == 1) & (pred == 0)))
    tn = int(np.count_nonzero((y == 0) & (pred == 0)))
    fp = int(np.count_nonzero((y == 0) & (pred == 1)))

    def div(num: float, den: float) -> float:
        return float(num / den) if den else 0.0

    r_bug, r_non_bug = div(tp, tp + fn), div(tn, tn + fp)
    p_bug, p_non_bug = div(tp, tp + fp), div(tn, tn + fn)
    f1_bug = div(2 * p_bug * r_bug, p_bug + r_bug)
    f1_non_bug = div(2 * p_non_bug * r_non_bug, p_non_bug + r_non_bug)
    mcc_den = np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    return {
        "accuracy": div(tp + tn, len(y)),
        "balanced_accuracy": (r_bug + r_non_bug) / 2,
        "recall_bug": r_bug,
        "recall_non_bug": r_non_bug,
        "precision_bug": p_bug,
        "precision_non_bug": p_non_bug,
        "f1_bug": f1_bug,
        "f1_non_bug": f1_non_bug,
        "macro_f1": (f1_bug + f1_non_bug) / 2,
        "matthews_corrcoef": div((tp * tn) - (fp * fn), mcc_den),
        "brier_score": float(np.mean(np.square(prob - y))),
    }


def paired_bootstrap(
    y: np.ndarray,
    baseline_pred: np.ndarray,
    treatment_pred: np.ndarray,
    baseline_prob: np.ndarray,
    treatment_prob: np.ndarray,
    replicates: int,
    seed: int,
) -> dict[str, tuple[float, float]]:
    """Percentile intervals from class-stratified paired resampling."""
    bug_idx = np.flatnonzero(y == 1)
    non_bug_idx = np.flatnonzero(y == 0)
    rng = np.random.default_rng(seed)
    keys = tuple(METRIC_FUNCTIONS)
    samples = {key: np.empty(replicates, dtype=np.float64) for key in keys}
    _, baseline_auc_groups = np.unique(baseline_prob, return_inverse=True)
    _, treatment_auc_groups = np.unique(treatment_prob, return_inverse=True)

    def weighted_auc(group_ids: np.ndarray, group_count: int, weights: np.ndarray) -> float:
        pos_weights = np.bincount(
            group_ids, weights=weights * (y == 1), minlength=group_count
        )
        neg_weights = np.bincount(
            group_ids, weights=weights * (y == 0), minlength=group_count
        )
        neg_below_or_half_tied = np.cumsum(neg_weights) - 0.5 * neg_weights
        return float(
            np.sum(pos_weights * neg_below_or_half_tied)
            / (len(bug_idx) * len(non_bug_idx))
        )

    for i in range(replicates):
        idx = np.concatenate((
            bug_idx[rng.integers(0, len(bug_idx), size=len(bug_idx))],
            non_bug_idx[rng.integers(0, len(non_bug_idx), size=len(non_bug_idx))],
        ))
        yb = y[idx]
        mb = _confusion_metrics(yb, baseline_pred[idx], baseline_prob[idx])
        mt = _confusion_metrics(yb, treatment_pred[idx], treatment_prob[idx])
        multiplicities = np.bincount(idx, minlength=len(y))
        mb["roc_auc"] = weighted_auc(
            baseline_auc_groups, int(baseline_auc_groups.max()) + 1, multiplicities
        )
        mt["roc_auc"] = weighted_auc(
            treatment_auc_groups, int(treatment_auc_groups.max()) + 1, multiplicities
        )
        for key in keys:
            samples[key][i] = mt[key] - mb[key]
        if (i + 1) % 500 == 0:
            print(f"Bootstrap replicates: {i + 1}/{replicates}", flush=True)

    return {
        key: tuple(float(x) for x in np.percentile(values, [2.5, 97.5]))
        for key, values in samples.items()
    }


def mcnemar_exact(
    y: np.ndarray, baseline_pred: np.ndarray, treatment_pred: np.ndarray, mask: np.ndarray
) -> tuple[int, int, float]:
    """Return (baseline-only correct, treatment-only correct, exact two-sided p)."""
    b_correct = baseline_pred[mask] == y[mask]
    t_correct = treatment_pred[mask] == y[mask]
    baseline_only = int(np.count_nonzero(b_correct & ~t_correct))
    treatment_only = int(np.count_nonzero(~b_correct & t_correct))
    discordant = baseline_only + treatment_only
    pvalue = float(binomtest(min(baseline_only, treatment_only), discordant, 0.5).pvalue) if discordant else 1.0
    return baseline_only, treatment_only, pvalue


def holm_adjust(pvalues: list[float]) -> list[float]:
    order = np.argsort(pvalues)
    adjusted = [0.0] * len(pvalues)
    running = 0.0
    for rank, index in enumerate(order):
        running = max(running, min(1.0, (len(pvalues) - rank) * pvalues[index]))
        adjusted[index] = running
    return adjusted


def _load_model(path: Path, embeddings: np.ndarray) -> tuple[np.ndarray, int]:
    booster = xgb.Booster()
    booster.load_model(path)
    booster.set_param({"device": "cuda"})
    probabilities = booster.predict(xgb.DMatrix(embeddings, nthread=1))
    if not np.isfinite(probabilities).all():
        raise ValueError(f"Non-finite predictions from {path}")
    return probabilities.astype(np.float64), booster.num_boosted_rounds()


def _fmt_p(value: float) -> str:
    return "< 1e-300" if value == 0 else f"{value:.3g}"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bootstrap-replicates", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.bootstrap_replicates < 100:
        raise ValueError("Use at least 100 bootstrap replicates.")

    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required; refusing to run model inference on CPU.")
    gpu_name = torch.cuda.get_device_name(0)
    print(f"CUDA inference device: {gpu_name}", flush=True)

    baseline_metrics = json.loads(BASELINE_METRICS.read_text(encoding="utf-8"))
    treatment_metrics = json.loads(TREATMENT_METRICS.read_text(encoding="utf-8"))
    manifest = json.loads(SPLIT_MANIFEST.read_text(encoding="utf-8"))
    expected_hash = manifest["identity_sha256"]["test"]
    if baseline_metrics["test_identity_sha256"] != treatment_metrics["test_identity_sha256"]:
        raise ValueError("Saved baseline and treatment metric artifacts identify different test sets.")

    test = pd.read_parquet(TEST_DATA)
    actual_hash = identity_digest(test)
    if actual_hash != expected_hash or actual_hash != baseline_metrics["test_identity_sha256"]:
        raise ValueError("Frozen test identity does not match the saved split and model metrics.")
    mapped_labels = test["original_label"].map({"bug": 1, "non-bug": 0})
    if mapped_labels.isna().any():
        raise ValueError("Unexpected test label encountered.")
    y = mapped_labels.to_numpy(dtype=np.int8)

    embeddings = np.load(TEST_EMBEDDINGS, mmap_mode="r")
    if embeddings.shape[0] != len(test):
        raise ValueError(f"Test rows ({len(test)}) do not match embedding rows ({embeddings.shape[0]}).")
    baseline_prob, baseline_rounds = _load_model(BASELINE_MODEL, embeddings)
    treatment_prob, treatment_rounds = _load_model(TREATMENT_MODEL, embeddings)
    baseline_pred = (baseline_prob >= float(baseline_metrics["optimal_threshold"])).astype(np.int8)
    treatment_pred = (treatment_prob >= float(treatment_metrics["optimal_threshold"])).astype(np.int8)

    point = {}
    reproduction_deltas: dict[str, dict[str, float]] = {}
    for name, probs, pred, saved in (
        ("baseline", baseline_prob, baseline_pred, baseline_metrics),
        ("treatment", treatment_prob, treatment_pred, treatment_metrics),
    ):
        computed = {key: float(fn(y, probs, pred)) for key, fn in METRIC_FUNCTIONS.items()}
        reproduction_deltas[name] = {
            key: computed[key] - float(saved[key])
            for key in computed
            if key in saved
        }
        point[name] = computed
        max_delta = max(abs(value) for value in reproduction_deltas[name].values())
        print(
            f"{name}: {len(y)} predictions, {baseline_rounds if name == 'baseline' else treatment_rounds} boosting rounds, "
            f"max absolute saved-metric delta={max_delta:.6f}", flush=True
        )

    print(f"Running {args.bootstrap_replicates} paired, class-stratified bootstrap replicates (seed={args.seed})", flush=True)
    intervals = paired_bootstrap(
        y, baseline_pred, treatment_pred, baseline_prob, treatment_prob,
        args.bootstrap_replicates, args.seed,
    )

    non_bug_mask = y == 0
    all_mask = np.ones(len(y), dtype=bool)
    nb_base_only, nb_treatment_only, nb_p = mcnemar_exact(y, baseline_pred, treatment_pred, non_bug_mask)
    all_base_only, all_treatment_only, all_p = mcnemar_exact(y, baseline_pred, treatment_pred, all_mask)
    adjusted = holm_adjust([nb_p, all_p])
    nb_delta = point["treatment"]["recall_non_bug"] - point["baseline"]["recall_non_bug"]
    nb_ci = intervals["recall_non_bug"]
    bug_recall_delta = point["treatment"]["recall_bug"] - point["baseline"]["recall_bug"]
    bug_recall_ci = intervals["recall_bug"]
    auc_delta = point["treatment"]["roc_auc"] - point["baseline"]["roc_auc"]
    auc_ci = intervals["roc_auc"]
    accuracy_delta = point["treatment"]["accuracy"] - point["baseline"]["accuracy"]
    accuracy_ci = intervals["accuracy"]

    metric_labels = {
        "accuracy": "Accuracy",
        "balanced_accuracy": "Balanced accuracy",
        "recall_bug": "Bug recall",
        "recall_non_bug": "Non-Bug recall (focus endpoint)",
        "precision_bug": "Bug precision",
        "precision_non_bug": "Non-Bug precision",
        "f1_bug": "Bug F1",
        "f1_non_bug": "Non-Bug F1",
        "macro_f1": "Macro-F1",
        "matthews_corrcoef": "MCC",
        "brier_score": "Brier score",
    "roc_auc": "ROC-AUC",
    }
    rows = [
        "# Paired statistical comparison: baseline vs treatment",
        "",
        f"Generated by `src/treatment/paired_statistical_analysis.py` on {date.today().isoformat()}.",
        "",
        "## Analysis population and artifacts",
        "",
        f"- Shared frozen test set: **{len(y):,} records** ({int(np.sum(y == 1)):,} BUG; {int(np.sum(y == 0)):,} NON-BUG).",
        f"- Test identity SHA-256: `{actual_hash}` (matches the split manifest and both saved metric files).",
        f"- Baseline checkpoint: [`{BASELINE_MODEL.relative_to(ROOT).as_posix()}`]({BASELINE_MODEL.relative_to(ROOT).as_posix()}).",
        f"- Treatment checkpoint: [`{TREATMENT_MODEL.relative_to(ROOT).as_posix()}`]({TREATMENT_MODEL.relative_to(ROOT).as_posix()}).",
        f"- Cached test embeddings: [`{TEST_EMBEDDINGS.relative_to(ROOT).as_posix()}`]({TEST_EMBEDDINGS.relative_to(ROOT).as_posix()}); model predictions were regenerated on **{gpu_name}** using XGBoost CUDA.",
        f"- Thresholds retained from validation: baseline **{baseline_metrics['optimal_threshold']:.2f}**, treatment **{treatment_metrics['optimal_threshold']:.2f}**. The thresholds differ because each run selected its threshold on validation.",
        f"- Model rounds: baseline **{baseline_rounds}**, treatment **{treatment_rounds}**. Predictions were regenerated from both checkpoints and compared with their stored metric files; see the discrepancy table below.",
        "- The comparison summary identifies the treatment as using 10,000 one-call, four-perspective audited training records. The saved artifacts do not establish that the baseline and treatment XGBoost hyperparameters and threshold-selection policy were fully matched.",
        "",
        "## Paired stratified bootstrap",
        "",
        f"Resampling was paired by record and stratified by the true source-label class, preserving the test-set BUG/NON-BUG counts. The same sampled records were used for each model in every replicate. Percentile 95% confidence intervals are for treatment minus baseline. Replicates: **{args.bootstrap_replicates:,}**; RNG seed: **{args.seed}**. Intervals are descriptive uncertainty estimates for performance against the original BugHub source labels.",
        "",
        "| Metric | Baseline | Treatment | Difference (treatment - baseline) | Paired 95% CI for difference |",
        "|---|---:|---:|---:|---:|",
    ]
    for key in intervals:
        delta = point["treatment"][key] - point["baseline"][key]
        low, high = intervals[key]
        rows.append(
        f"| {metric_labels[key]} | {point['baseline'][key]:.4f} | {point['treatment'][key]:.4f} | {delta:+.4f} | [{low:+.6f}, {high:+.6f}] |"
        )
    rows.extend([
        "",
        "### Checkpoint re-inference vs stored metrics",
        "",
        "The table below reports re-inferred checkpoint metric minus the stored metric. Nonzero differences mean this report describes predictions regenerated in the current CUDA runtime; it does not silently substitute them for the originally saved comparison values.",
        "",
        "| Metric | Baseline delta | Treatment delta |",
        "|---|---:|---:|",
    ])
    for key in METRIC_FUNCTIONS:
        rows.append(
            f"| {metric_labels.get(key, key)} | {reproduction_deltas['baseline'][key]:+.6f} | {reproduction_deltas['treatment'][key]:+.6f} |"
        )
    rows.extend([
        "",
        "## Exact McNemar tests",
        "",
        "McNemar tests compare paired correctness disagreements, using two-sided exact binomial p-values. The NON-BUG test is restricted to actual NON-BUG test records and therefore tests the change in NON-BUG recall. Holm adjustment covers these two reported tests.",
        "",
        "| Endpoint | Baseline-only correct | Treatment-only correct | Exact two-sided p | Holm-adjusted p |",
        "|---|---:|---:|---:|---:|",
        f"| NON-BUG records (NON-BUG recall) | {nb_base_only:,} | {nb_treatment_only:,} | {_fmt_p(nb_p)} | {_fmt_p(adjusted[0])} |",
        f"| All test records (accuracy) | {all_base_only:,} | {all_treatment_only:,} | {_fmt_p(all_p)} | {_fmt_p(adjusted[1])} |",
        "",
        "## Interpretation and limits",
        "",
        f"- For the NON-BUG recall focus endpoint, treatment minus baseline is **{nb_delta:+.4f}** (95% paired CI [{nb_ci[0]:+.4f}, {nb_ci[1]:+.4f}]); the Holm-adjusted exact McNemar p-value is **{_fmt_p(adjusted[0])}**.",
        f"- BUG recall changed by **{bug_recall_delta:+.4f}** (95% CI [{bug_recall_ci[0]:+.4f}, {bug_recall_ci[1]:+.4f}]); ROC-AUC changed by **{auc_delta:+.4f}** (95% CI [{auc_ci[0]:+.4f}, {auc_ci[1]:+.4f}]). These exploratory metric intervals were not adjusted for testing multiple metrics.",
        f"- Accuracy changed by **{accuracy_delta:+.4f}** (95% paired CI [{accuracy_ci[0]:+.4f}, {accuracy_ci[1]:+.4f}]); the Holm-adjusted exact McNemar p-value is **{_fmt_p(adjusted[1])}**.",
        "- These tests compare predictions against the original BugHub source labels. They do not establish that source or agent-audited labels are independently correct, and they do not establish causal improvement beyond this experiment.",
        "- The models use different validation-selected thresholds (0.52 vs 0.55). Threshold-based differences therefore reflect each model's retained validation policy as well as learned model differences. ROC-AUC is threshold independent and shown descriptively.",
        "- The bootstrap and McNemar analyses are post-hoc analyses of one frozen test set; the focus endpoint was not preregistered in the retained artifacts. Intervals do not account for training-seed or audit-sample variability. Holm adjustment applies to the two McNemar tests, not to the exploratory bootstrap intervals across metrics.",
        "- Saved comparison artifacts: [`outputs/results/baseline/full/metrics.json`](outputs/results/baseline/full/metrics.json), [`outputs/results/treatment/metrics.json`](outputs/results/treatment/metrics.json), and [`outputs/results/comparison/comparison_summary.json`](outputs/results/comparison/comparison_summary.json).",
        "",
    ])
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text("\n".join(rows), encoding="utf-8")
    print(f"Report written: {REPORT_PATH.relative_to(ROOT)}", flush=True)


if __name__ == "__main__":
    main()
