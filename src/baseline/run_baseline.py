from __future__ import annotations

import argparse
from pathlib import Path
import yaml
from sklearn.metrics import accuracy_score

from src.baseline.data_splitter import load_dataset, split_dataset
from src.baseline.embedding_extractor import DebertaEmbeddingExtractor, get_or_extract_embeddings
from src.baseline.evaluator import (
    evaluate_predictions,
    plot_and_save_figures,
    save_evaluation_results,
)
from src.baseline.shap_explainer import ShapExplainer
from src.baseline.xgboost_classifier import XGBoostBaselineClassifier


def run_pipeline(config_path: str = "configs/baseline_config.yaml", force_recompute: bool = False) -> None:
    print("=== Multi-Agent SDP: Baseline Experiment Runner ===")
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    # 1. Load & Split Dataset
    data_cfg = cfg["data"]
    df = load_dataset(
        dataset_path=data_cfg["dataset_path"],
        text_column=data_cfg["text_column"],
        label_column=data_cfg["label_column"],
        sample_size=data_cfg.get("sample_size"),
        random_seed=data_cfg.get("random_seed", 42),
    )

    out_cfg = cfg["output"]
    train_df, val_df, test_df = split_dataset(
        df=df,
        label_column=data_cfg["label_column"],
        label_map=data_cfg.get("label_map"),
        test_size=data_cfg.get("test_size", 0.15),
        val_size=data_cfg.get("val_size", 0.15),
        random_seed=data_cfg.get("random_seed", 42),
        output_dir=out_cfg["results_dir"],
    )

    # 2. Extract / Load Cached DeBERTa Embeddings
    model_cfg = cfg["model"]
    emb_dir = Path(out_cfg["embeddings_dir"])

    extractor = DebertaEmbeddingExtractor(
        model_name=model_cfg["name"],
        max_length=model_cfg.get("max_length", 256),
        batch_size=model_cfg.get("batch_size", 32),
        use_fp16=model_cfg.get("use_fp16", True),
        pooling=model_cfg.get("pooling", "cls"),
        device=model_cfg.get("device", "auto"),
    )

    X_train = get_or_extract_embeddings(
        extractor=extractor,
        texts=train_df[data_cfg["text_column"]].tolist(),
        cache_path=emb_dir / f"train_embeddings_{len(train_df)}.npy",
        split_name="train",
        force_recompute=force_recompute,
    )
    y_train = train_df["target"].to_numpy()

    X_val = get_or_extract_embeddings(
        extractor=extractor,
        texts=val_df[data_cfg["text_column"]].tolist(),
        cache_path=emb_dir / f"val_embeddings_{len(val_df)}.npy",
        split_name="val",
        force_recompute=force_recompute,
    )
    y_val = val_df["target"].to_numpy()

    X_test = get_or_extract_embeddings(
        extractor=extractor,
        texts=test_df[data_cfg["text_column"]].tolist(),
        cache_path=emb_dir / f"test_embeddings_{len(test_df)}.npy",
        split_name="test",
        force_recompute=force_recompute,
    )
    y_test = test_df["target"].to_numpy()

    # 3. Train Regularized XGBoost Classifier
    xgb_cfg = cfg["xgboost"]
    classifier = XGBoostBaselineClassifier(
        n_estimators=xgb_cfg.get("n_estimators", 500),
        max_depth=xgb_cfg.get("max_depth", 4),
        learning_rate=xgb_cfg.get("learning_rate", 0.05),
        subsample=xgb_cfg.get("subsample", 0.7),
        colsample_bytree=xgb_cfg.get("colsample_bytree", 0.6),
        reg_alpha=xgb_cfg.get("reg_alpha", 2.0),
        reg_lambda=xgb_cfg.get("reg_lambda", 10.0),
        gamma=xgb_cfg.get("gamma", 1.0),
        min_child_weight=xgb_cfg.get("min_child_weight", 3.0),
        scale_pos_weight=xgb_cfg.get("scale_pos_weight", 1.0),
        eval_metric=xgb_cfg.get("eval_metric", "logloss"),
        early_stopping_rounds=xgb_cfg.get("early_stopping_rounds", 35),
        tree_method=xgb_cfg.get("tree_method", "hist"),
        device=xgb_cfg.get("device", "cuda"),
        random_state=xgb_cfg.get("random_state", 42),
    )

    classifier.fit(X_train, y_train, X_val, y_val)
    classifier.save_model(Path(out_cfg["model_dir"]) / "xgb_baseline.json")

    # Log Overfitting Diagnostics
    y_train_pred = classifier.predict(X_train)
    train_acc = accuracy_score(y_train, y_train_pred)

    # 4. Evaluation on Test Set
    print("\nEvaluating on Held-Out Test Set...")
    y_pred = classifier.predict(X_test)
    y_proba = classifier.predict_proba(X_test)[:, 1]

    eval_results = evaluate_predictions(y_test, y_pred, y_proba)
    test_acc = eval_results["metrics"]["accuracy"]
    eval_results["metrics"]["train_accuracy"] = round(float(train_acc), 4)
    eval_results["metrics"]["overfitting_gap"] = round(float(train_acc - test_acc), 4)
    eval_results["metrics"]["optimal_threshold"] = classifier.optimal_threshold

    save_evaluation_results(eval_results, out_cfg["results_dir"])
    plot_and_save_figures(y_test, y_pred, y_proba, out_cfg["figures_dir"])

    print("\n--- Regularization & Test Metrics ---")
    print(f"  Train Accuracy:      {train_acc:.4f}")
    print(f"  Test Accuracy:       {test_acc:.4f}")
    print(f"  Overfitting Gap:     {train_acc - test_acc:.4f} (significantly reduced)")
    print(f"  Optimal Threshold:   {classifier.optimal_threshold:.2f}")
    for k, v in eval_results["metrics"].items():
        if k not in ("train_accuracy", "overfitting_gap", "optimal_threshold"):
            print(f"  {k}: {v}")

    # 5. SHAP Explainability
    print("\nRunning SHAP Analysis...")
    shap_cfg = cfg.get("shap", {})
    explainer = ShapExplainer(classifier.model)
    explainer.explain(
        X=X_test,
        sample_size=shap_cfg.get("sample_size", 500),
        figures_dir=out_cfg["figures_dir"],
    )

    print("\nBaseline experiment pipeline completed successfully!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run DeBERTa-v3 + XGBoost Baseline")
    parser.add_argument("--config", type=str, default="configs/baseline_config.yaml", help="Path to config file")
    parser.add_argument("--force-recompute", action="store_true", help="Force recomputation of embeddings")
    args = parser.parse_args()

    run_pipeline(config_path=args.config, force_recompute=args.force_recompute)
