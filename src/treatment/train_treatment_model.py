from __future__ import annotations

import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import yaml

from src.baseline.embedding_extractor import DebertaEmbeddingExtractor, get_or_extract_embeddings
from src.baseline.evaluator import (
    evaluate_predictions,
    plot_and_save_figures,
    save_evaluation_results,
)
from src.baseline.shap_explainer import ShapExplainer
from src.baseline.xgboost_classifier import XGBoostBaselineClassifier


def train_treatment_pipeline(
    dataset_path: str = "data/processed/treatment_sample.parquet",
    output_dir: str = "outputs",
    force_recompute: bool = False,
    random_seed: int = 42,
) -> dict:
    """Trains the treatment defect prediction model on multi-agent audited labels.

    Uses the identical DeBERTa-v3 feature extractor and regularized XGBoost architecture
    to ensure strict baseline-vs-treatment experimental comparability.
    """
    print("\n" + "=" * 60)
    print("=== Multi-Agent SDP: Treatment Model Training (Audited Labels) ===")
    print("=" * 60)

    # 1. Load Dataset
    print(f"Loading treatment dataset: {dataset_path}...")
    df = pd.read_parquet(dataset_path)
    total_len = len(df)
    print(f"Loaded {total_len} records.")

    # Target mapping
    target_col = "treatment_label"
    label_map = {"bug": 1, "non-bug": 0}
    df["target"] = df[target_col].map(label_map)

    # 2. Stratified Split (70% train / 15% val / 15% test)
    train_val_df, test_df = train_test_split(
        df,
        test_size=0.15,
        random_state=random_seed,
        stratify=df["target"],
    )
    val_relative_ratio = 0.15 / (1.0 - 0.15)  # 0.17647
    train_df, val_df = train_test_split(
        train_val_df,
        test_size=val_relative_ratio,
        random_state=random_seed,
        stratify=train_val_df["target"],
    )

    print(f"Splits: Train = {len(train_df)}, Val = {len(val_df)}, Test = {len(test_df)}")

    # Save splits summary
    splits_summary = {
        "train_size": len(train_df),
        "val_size": len(val_df),
        "test_size": len(test_df),
        "train_bug_ratio": round(float(train_df["target"].mean()), 4),
        "val_bug_ratio": round(float(val_df["target"].mean()), 4),
        "test_bug_ratio": round(float(test_df["target"].mean()), 4),
        "random_seed": random_seed,
    }
    results_dir = Path(output_dir) / "results" / "treatment"
    results_dir.mkdir(parents=True, exist_ok=True)
    with open(results_dir / "splits_summary.json", "w", encoding="utf-8") as f:
        json.dump(splits_summary, f, indent=2)

    # 3. Extract / Load DeBERTa-v3 Embeddings with LoRA fine-tuned adapter
    emb_dir = Path("data/embeddings/treatment")
    emb_dir.mkdir(parents=True, exist_ok=True)

    adapter_path = Path("outputs/models/treatment_deberta_lora")
    lora_adapter = str(adapter_path) if adapter_path.exists() else None
    cache_prefix = "lora_" if lora_adapter else ""
    print(f"Embedding extractor adapter: {lora_adapter}")

    extractor = DebertaEmbeddingExtractor(
        model_name="microsoft/deberta-v3-small",
        adapter_path=lora_adapter,
        max_length=256,
        batch_size=32,
        use_fp16=True,
        pooling="cls",
        device="auto",
    )

    X_train_emb = get_or_extract_embeddings(
        extractor=extractor,
        texts=train_df["model_text"].tolist(),
        cache_path=emb_dir / f"{cache_prefix}train_embeddings_{len(train_df)}.npy",
        split_name="treatment_train",
        force_recompute=force_recompute,
    )
    y_train = train_df["target"].to_numpy()

    X_val_emb = get_or_extract_embeddings(
        extractor=extractor,
        texts=val_df["model_text"].tolist(),
        cache_path=emb_dir / f"{cache_prefix}val_embeddings_{len(val_df)}.npy",
        split_name="treatment_val",
        force_recompute=force_recompute,
    )
    y_val = val_df["target"].to_numpy()

    X_test_emb = get_or_extract_embeddings(
        extractor=extractor,
        texts=test_df["model_text"].tolist(),
        cache_path=emb_dir / f"{cache_prefix}test_embeddings_{len(test_df)}.npy",
        split_name="treatment_test",
        force_recompute=force_recompute,
    )
    y_test = test_df["target"].to_numpy()

    # Extract Knowledge Graph Prior & Context Features
    print("Extracting Knowledge Graph relational features...")
    from src.knowledge_graph.retriever import get_kg_retriever
    retriever = get_kg_retriever()

    def _extract_kg_features(df_split: pd.DataFrame) -> np.ndarray:
        feats = []
        for _, row in df_split.iterrows():
            proj = str(row.get("project", "")).lower()
            comp = str(row.get("component", "")).lower() if pd.notna(row.get("component")) else ""
            sev = str(row.get("severity", "")).lower() if pd.notna(row.get("severity")) else ""
            pri = str(row.get("priority", "")).lower() if pd.notna(row.get("priority")) else ""

            p_rate = 0.70
            if retriever.df is not None:
                p_sub = retriever.df[retriever.df["project"].str.lower() == proj]
                if len(p_sub) > 0:
                    p_rate = float((p_sub["original_label"].str.lower() == "bug").mean())

            c_rate = p_rate
            if comp and retriever.df is not None:
                c_sub = retriever.df[(retriever.df["project"].str.lower() == proj) & (retriever.df["component"].astype(str).str.lower() == comp)]
                if len(c_sub) > 0:
                    c_rate = float((c_sub["original_label"].str.lower() == "bug").mean())

            sev_weights = {"blocker": 1.0, "critical": 0.9, "major": 0.75, "normal": 0.5, "minor": 0.25, "trivial": 0.1}
            s_val = sev_weights.get(sev, 0.5)

            pri_weights = {"highest": 1.0, "high": 0.8, "p1": 1.0, "p2": 0.8, "p3": 0.5, "medium": 0.5, "p4": 0.3, "p5": 0.1, "low": 0.2}
            p_val = pri_weights.get(pri, 0.5)

            feats.append([p_rate, c_rate, s_val, p_val])
        return np.array(feats, dtype=np.float32)

    X_train = np.hstack([X_train_emb, _extract_kg_features(train_df)])
    X_val = np.hstack([X_val_emb, _extract_kg_features(val_df)])
    X_test = np.hstack([X_test_emb, _extract_kg_features(test_df)])
    print(f"Combined Feature Matrix Shape: Train {X_train.shape}, Test {X_test.shape} (768 text emb + 4 KG features)")


    # 4. Train Regularized XGBoost Classifier
    print("\nTraining Regularized XGBoost on Treatment Labels (Anti-Overfitting Configuration)...")
    classifier = XGBoostBaselineClassifier(
        n_estimators=300,
        max_depth=2,
        learning_rate=0.03,
        subsample=0.6,
        colsample_bytree=0.5,
        reg_alpha=10.0,
        reg_lambda=30.0,
        gamma=3.0,
        min_child_weight=8.0,
        scale_pos_weight=1.0,
        eval_metric="logloss",
        early_stopping_rounds=25,
        tree_method="hist",
        device="cuda",
        random_state=random_seed,
    )

    classifier.fit(X_train, y_train, X_val, y_val)

    # Save model and threshold
    model_dir = Path(output_dir) / "models" / "treatment"
    model_dir.mkdir(parents=True, exist_ok=True)
    classifier.save_model(model_dir / "xgb_treatment.json")

    # Overfitting Diagnostics
    y_train_pred = classifier.predict(X_train)
    train_acc = accuracy_score(y_train, y_train_pred)

    # 5. Evaluate on Held-Out Test Set
    print("\nEvaluating Treatment Model on Held-Out Test Set...")
    y_pred = classifier.predict(X_test)
    y_proba = classifier.predict_proba(X_test)[:, 1]

    eval_results = evaluate_predictions(y_test, y_pred, y_proba)
    test_acc = eval_results["metrics"]["accuracy"]
    eval_results["metrics"]["train_accuracy"] = round(float(train_acc), 4)
    eval_results["metrics"]["overfitting_gap"] = round(float(train_acc - test_acc), 4)
    eval_results["metrics"]["optimal_threshold"] = classifier.optimal_threshold

    figures_dir = Path(output_dir) / "figures" / "treatment"
    figures_dir.mkdir(parents=True, exist_ok=True)

    save_evaluation_results(eval_results, str(results_dir))
    plot_and_save_figures(y_test, y_pred, y_proba, str(figures_dir))

    print("\n--- Treatment Model Evaluation Metrics ---")
    print(f"  Train Accuracy:      {train_acc:.4f}")
    print(f"  Test Accuracy:       {test_acc:.4f}")
    print(f"  Overfitting Gap:     {train_acc - test_acc:.4f}")
    print(f"  Optimal Threshold:   {classifier.optimal_threshold:.2f}")
    for k, v in eval_results["metrics"].items():
        if k not in ("train_accuracy", "overfitting_gap", "optimal_threshold"):
            print(f"  {k}: {v}")

    # 6. SHAP Feature Attribution
    print("\nRunning SHAP Analysis on Treatment Model...")
    explainer = ShapExplainer(classifier.model)
    explainer.explain(
        X=X_test,
        sample_size=min(500, len(X_test)),
        figures_dir=str(figures_dir),
    )

    print("\nTreatment model training pipeline completed successfully!")
    return eval_results["metrics"]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Treatment Model on Audited Labels")
    parser.add_argument("--dataset-path", default="data/processed/treatment_sample.parquet")
    parser.add_argument("--output-dir", default="outputs")
    parser.add_argument("--force-recompute", action="store_true")
    parser.add_argument("--random-seed", type=int, default=42)
    args = parser.parse_args()

    train_treatment_pipeline(
        dataset_path=args.dataset_path,
        output_dir=args.output_dir,
        force_recompute=args.force_recompute,
        random_seed=args.random_seed,
    )
