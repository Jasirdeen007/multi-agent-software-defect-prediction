from __future__ import annotations

import argparse
import json
from pathlib import Path
import numpy as np
import optuna
import xgboost as xgb
from sklearn.metrics import accuracy_score, balanced_accuracy_score, roc_auc_score, f1_score
import yaml

from src.baseline.data_splitter import load_dataset, split_dataset
from src.baseline.evaluator import evaluate_predictions, plot_and_save_figures, save_evaluation_results


def tune_xgboost(
    embeddings_dir: str = "data/embeddings",
    n_trials: int = 30,
    output_subdir: str = "tuned_xgboost",
    config_path: str = "configs/baseline_config.yaml",
) -> None:
    print(f"=== Optuna Hyperparameter Optimization for XGBoost ===")
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    # 1. Load Data
    data_cfg = cfg["data"]
    df = load_dataset(data_cfg["dataset_path"], sample_size=data_cfg.get("sample_size"), random_seed=data_cfg.get("random_seed", 42))
    train_df, val_df, test_df = split_dataset(df, random_seed=data_cfg.get("random_seed", 42))

    y_train = train_df["target"].to_numpy()
    y_val = val_df["target"].to_numpy()
    y_test = test_df["target"].to_numpy()

    emb_path = Path(embeddings_dir)
    X_train = np.load(emb_path / f"train_embeddings_{len(train_df)}.npy")
    X_val = np.load(emb_path / f"val_embeddings_{len(val_df)}.npy")
    X_test = np.load(emb_path / f"test_embeddings_{len(test_df)}.npy")

    print(f"Loaded embeddings for tuning: Train={X_train.shape}, Val={X_val.shape}, Test={X_test.shape}")

    # 2. Define Objective Function
    def objective(trial: optuna.Trial) -> float:
        params = {
            "n_estimators": 450,
            "max_depth": trial.suggest_int("max_depth", 3, 5),
            "learning_rate": trial.suggest_float("learning_rate", 0.02, 0.08, log=True),
            "subsample": trial.suggest_float("subsample", 0.5, 0.8),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.4, 0.7),
            "reg_alpha": trial.suggest_float("reg_alpha", 0.5, 5.0),
            "reg_lambda": trial.suggest_float("reg_lambda", 2.0, 15.0),
            "gamma": trial.suggest_float("gamma", 0.5, 3.0),
            "min_child_weight": trial.suggest_float("min_child_weight", 1.0, 6.0),
            "scale_pos_weight": 1.0,
            "eval_metric": "logloss",
            "early_stopping_rounds": 25,
            "tree_method": "hist",
            "device": "cuda",
            "random_state": 42,
        }

        clf = xgb.XGBClassifier(**params)
        clf.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)

        val_proba = clf.predict_proba(X_val)[:, 1]
        val_auc = roc_auc_score(y_val, val_proba)
        return val_auc

    # 3. Run Optimization Study
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(direction="maximize")
    print(f"Running {n_trials} trials to find best generalization hyperparameters...")
    study.optimize(objective, n_trials=n_trials)

    print("\n--- Optuna Best Hyperparameters ---")
    best_params = study.best_params
    for k, v in best_params.items():
        print(f"  {k}: {v}")
    print(f"  Best Validation ROC-AUC: {study.best_value:.4f}")

    # 4. Train Final Model with Best Parameters
    full_params = {
        "n_estimators": 600,
        "eval_metric": "logloss",
        "early_stopping_rounds": 35,
        "tree_method": "hist",
        "device": "cuda",
        "random_state": 42,
        **best_params,
    }
    best_clf = xgb.XGBClassifier(**full_params)
    best_clf.fit(X_train, y_train, eval_set=[(X_train, y_train), (X_val, y_val)], verbose=50)

    # 5. Threshold Optimization
    val_proba = best_clf.predict_proba(X_val)[:, 1]
    best_th = 0.50
    best_val_acc = 0.0
    for th in np.arange(0.40, 0.70, 0.01):
        acc = accuracy_score(y_val, (val_proba >= th).astype(int))
        if acc > best_val_acc:
            best_val_acc = acc
            best_th = round(float(th), 2)

    print(f"Optimized Threshold: {best_th:.2f} (Val Acc: {best_val_acc:.4f})")

    # 6. Final Test Set Evaluation
    test_proba = best_clf.predict_proba(X_test)[:, 1]
    test_pred = (test_proba >= best_th).astype(int)
    train_pred = (best_clf.predict_proba(X_train)[:, 1] >= best_th).astype(int)
    train_acc = accuracy_score(y_train, train_pred)

    out_results_dir = Path(cfg["output"]["results_dir"]) / output_subdir
    out_figures_dir = Path(cfg["output"]["figures_dir"]) / output_subdir
    out_model_dir = Path(cfg["output"]["model_dir"]) / output_subdir

    eval_results = evaluate_predictions(y_test, test_pred, test_proba)
    eval_results["metrics"]["train_accuracy"] = round(float(train_acc), 4)
    eval_results["metrics"]["overfitting_gap"] = round(float(train_acc - eval_results["metrics"]["accuracy"]), 4)
    eval_results["metrics"]["optimal_threshold"] = best_th
    eval_results["metrics"]["best_params"] = best_params

    save_evaluation_results(eval_results, out_results_dir)
    plot_and_save_figures(y_test, test_pred, test_proba, out_figures_dir)
    out_model_dir.mkdir(parents=True, exist_ok=True)
    best_clf.save_model(str(out_model_dir / "xgb_tuned.json"))

    print("\n--- Final Tuned Model Test Metrics ---")
    print(f"  Train Accuracy:   {train_acc:.4f}")
    print(f"  Test Accuracy:    {eval_results['metrics']['accuracy']:.4f}")
    print(f"  Overfitting Gap:  {eval_results['metrics']['overfitting_gap']:.4f}")
    print(f"  Test ROC-AUC:     {eval_results['metrics']['roc_auc']:.4f}")
    print(f"  Test Macro F1:    {eval_results['metrics']['macro_f1']:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Tune XGBoost on embeddings")
    parser.add_argument("--n-trials", type=int, default=25)
    parser.add_argument("--embeddings-dir", type=str, default="data/embeddings")
    parser.add_argument("--output-subdir", type=str, default="tuned_xgboost")
    args = parser.parse_args()

    tune_xgboost(embeddings_dir=args.embeddings_dir, n_trials=args.n_trials, output_subdir=args.output_subdir)
