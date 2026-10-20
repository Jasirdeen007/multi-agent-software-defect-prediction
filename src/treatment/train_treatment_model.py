from __future__ import annotations

import argparse
from pathlib import Path
import yaml
import pandas as pd
from sklearn.metrics import accuracy_score

from src.baseline.embedding_extractor import DebertaEmbeddingExtractor, get_or_extract_embeddings
from src.baseline.evaluator import evaluate_predictions, plot_and_save_figures, save_evaluation_results
from src.baseline.data_splitter import validate_split_disjointness
from src.baseline.xgboost_classifier import XGBoostBaselineClassifier


def train_treatment_pipeline(
    dataset_path: str = "data/processed/fixed_splits/treatment_train.parquet",
    config_path: str = "configs/baseline_config.yaml",
    force_recompute: bool = False,
) -> dict:
    """Train treatment with the baseline protocol; only training labels may differ."""
    cfg = yaml.safe_load(Path(config_path).read_text(encoding="utf-8"))
    data_cfg, model_cfg, xgb_cfg, out_cfg = (cfg[k] for k in ("data", "model", "xgboost", "output"))
    train = pd.read_parquet(dataset_path)
    val = pd.read_parquet(Path(data_cfg.get("split_dir", "data/processed/fixed_splits")) / "validation.parquet")
    test = pd.read_parquet(Path(data_cfg.get("split_dir", "data/processed/fixed_splits")) / "test.parquet")
    validate_split_disjointness(train, val, test)
    if not train["is_audited"].isin([True, False]).all():
        raise ValueError("Treatment training data is missing audit provenance.")
    identity = ["source", "project", "issue_id"]
    if set(map(tuple, train[identity].astype(str).itertuples(index=False, name=None))) != set(
        map(tuple, pd.read_parquet(Path(data_cfg.get("split_dir", "data/processed/fixed_splits")) / "train.parquet")[identity].astype(str).itertuples(index=False, name=None))
    ):
        raise ValueError("Treatment training identities differ from the frozen baseline training split.")

    label_map = data_cfg.get("label_map", {"bug": 1, "non-bug": 0})
    y_train = train["treatment_label"].map(label_map).astype(int).to_numpy()
    y_val = val["original_label"].map(label_map).astype(int).to_numpy()
    y_test = test["original_label"].map(label_map).astype(int).to_numpy()
    # Reuse baseline's frozen encoder and split-specific embedding caches.
    extractor = DebertaEmbeddingExtractor(
        model_name=model_cfg["name"], max_length=model_cfg.get("max_length", 256),
        batch_size=model_cfg.get("batch_size", 32), use_fp16=model_cfg.get("use_fp16", True),
        pooling=model_cfg.get("pooling", "cls"), device=model_cfg.get("device", "auto"),
    )
    emb_dir = Path(out_cfg["embeddings_dir"])
    arrays = []
    for part, name in ((train, "train"), (val, "val"), (test, "test")):
        arrays.append(get_or_extract_embeddings(
            extractor, part[data_cfg["text_column"]].astype(str).tolist(),
            emb_dir / f"{name}_embeddings_{len(part)}.npy", f"treatment_{name}", force_recompute,
        ))
    X_train, X_val, X_test = arrays

    classifier = XGBoostBaselineClassifier(
        n_estimators=xgb_cfg.get("n_estimators", 500), max_depth=xgb_cfg.get("max_depth", 4),
        learning_rate=xgb_cfg.get("learning_rate", 0.05), subsample=xgb_cfg.get("subsample", 0.7),
        colsample_bytree=xgb_cfg.get("colsample_bytree", 0.6), reg_alpha=xgb_cfg.get("reg_alpha", 2.0),
        reg_lambda=xgb_cfg.get("reg_lambda", 10.0), gamma=xgb_cfg.get("gamma", 1.0),
        min_child_weight=xgb_cfg.get("min_child_weight", 3.0), scale_pos_weight=xgb_cfg.get("scale_pos_weight", 1.0),
        eval_metric=xgb_cfg.get("eval_metric", "logloss"),
        early_stopping_rounds=xgb_cfg.get("early_stopping_rounds", 35),
        tree_method=xgb_cfg.get("tree_method", "hist"), device=xgb_cfg.get("device", "cuda"),
        random_state=xgb_cfg.get("random_state", 42),
    ).fit(X_train, y_train, X_val, y_val)

    model_dir = Path("outputs/models/treatment")
    model_dir.mkdir(parents=True, exist_ok=True)
    classifier.save_model(model_dir / "xgb_treatment.json")
    pred = classifier.predict(X_test)
    proba = classifier.predict_proba(X_test)[:, 1]
    result = evaluate_predictions(y_test, pred, proba)
    result["metrics"]["train_accuracy"] = round(float(accuracy_score(y_train, classifier.predict(X_train))), 4)
    result["metrics"]["optimal_threshold"] = classifier.optimal_threshold
    results_dir = Path("outputs/results/treatment")
    results_dir.mkdir(parents=True, exist_ok=True)
    save_evaluation_results(result, str(results_dir))
    plot_and_save_figures(y_test, pred, proba, "outputs/figures/treatment")
    return result["metrics"]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-path", default="data/processed/fixed_splits/treatment_train.parquet")
    parser.add_argument("--config", default="configs/baseline_config.yaml")
    parser.add_argument("--force-recompute", action="store_true")
    args = parser.parse_args()
    train_treatment_pipeline(args.dataset_path, args.config, args.force_recompute)
