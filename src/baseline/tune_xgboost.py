from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import optuna
import pandas as pd
import torch
import xgboost as xgb
import yaml
from sklearn.metrics import accuracy_score, roc_auc_score

from src.baseline.data_splitter import load_frozen_split_metadata
from src.baseline.evaluator import evaluate_predictions, save_evaluation_results


def tune_xgboost(
    n_trials: int = 15,
    config_path: str = "configs/baseline_fixed_split_config.yaml",
    output_subdir: str = "tuned_xgboost",
) -> None:
    """Tune only on the frozen training/validation splits; use test once at the end."""
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this GPU-only tuning run; refusing CPU fallback.")
    build = xgb.build_info()
    if not build.get("USE_CUDA", False):
        raise RuntimeError("Installed XGBoost build lacks CUDA support; refusing CPU fallback.")

    cfg = yaml.safe_load(Path(config_path).read_text(encoding="utf-8"))
    data_cfg, xgb_cfg, out_cfg = cfg["data"], cfg["xgboost"], cfg["output"]
    split_dir = Path(data_cfg["split_dir"])
    train, validation, test = load_frozen_split_metadata(
        split_dir=split_dir,
        label_column=data_cfg["label_column"],
        random_seed=data_cfg.get("random_seed", 42),
        test_size=data_cfg.get("test_size", 0.15),
        val_size=data_cfg.get("val_size", 0.15),
    )
    label_map = data_cfg.get("label_map", {"bug": 1, "non-bug": 0})
    y_train = train[data_cfg["label_column"]].map(label_map).to_numpy(dtype=np.int8)
    y_val = validation[data_cfg["label_column"]].map(label_map).to_numpy(dtype=np.int8)

    embedding_dir = Path(out_cfg["embeddings_dir"])
    def load_embedding(name: str, rows: int) -> np.ndarray:
        path = embedding_dir / f"{name}_embeddings_{rows}.npy"
        if not path.is_file():
            raise FileNotFoundError(
                f"Missing {path}. Run the fixed-split baseline once to extract and cache DeBERTa embeddings."
            )
        matrix = np.load(path, mmap_mode="r")
        if len(matrix) != rows:
            raise ValueError(f"{path} has {len(matrix)} rows; expected {rows} frozen-split rows.")
        return matrix

    X_train = load_embedding("train", len(train))
    X_val = load_embedding("val", len(validation))
    print(f"GPU={torch.cuda.get_device_name(0)}; embedding shapes: {X_train.shape}, {X_val.shape}")
    del train

    tuning_results = Path(out_cfg["results_dir"]) / output_subdir
    model_dir = Path(out_cfg["model_dir"]) / output_subdir
    tuning_results.mkdir(parents=True, exist_ok=True)
    model_dir.mkdir(parents=True, exist_ok=True)
    study_db = (tuning_results / "optuna_study.sqlite3").resolve().as_posix()
    study = optuna.create_study(
        study_name="baseline_frozen_split_cuda_v1",
        storage=f"sqlite:///{study_db}",
        direction="maximize",
        load_if_exists=True,
    )
    # Include the previous 78.52%-accuracy configuration as the first explicit candidate.
    if len(study.trials) == 0:
        study.enqueue_trial({
            "n_estimators": xgb_cfg.get("n_estimators", 500),
            "max_bin": xgb_cfg.get("max_bin", 64),
            "max_depth": xgb_cfg.get("max_depth", 4),
            "learning_rate": xgb_cfg.get("learning_rate", 0.05),
            "subsample": xgb_cfg.get("subsample", 0.7),
            "colsample_bytree": xgb_cfg.get("colsample_bytree", 0.6),
            "reg_alpha": xgb_cfg.get("reg_alpha", 2.0),
            "reg_lambda": xgb_cfg.get("reg_lambda", 10.0),
            "gamma": xgb_cfg.get("gamma", 1.0),
            "min_child_weight": xgb_cfg.get("min_child_weight", 3.0),
        })

    def objective(trial: optuna.Trial) -> float:
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 300, 800, step=100),
            "max_bin": trial.suggest_categorical("max_bin", [32, 64]),
            "max_depth": trial.suggest_int("max_depth", 3, 6),
            "learning_rate": trial.suggest_float("learning_rate", 0.02, 0.08, log=True),
            "subsample": trial.suggest_float("subsample", 0.55, 0.85),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.4, 0.8),
            "reg_alpha": trial.suggest_float("reg_alpha", 0.1, 5.0),
            "reg_lambda": trial.suggest_float("reg_lambda", 2.0, 20.0, log=True),
            "gamma": trial.suggest_float("gamma", 0.0, 3.0),
            "min_child_weight": trial.suggest_float("min_child_weight", 1.0, 8.0),
        }
        classifier = xgb.XGBClassifier(
            **params,
            scale_pos_weight=xgb_cfg.get("scale_pos_weight", 1.0),
            eval_metric="logloss",
            early_stopping_rounds=xgb_cfg.get("early_stopping_rounds", 35),
            tree_method="hist",
            device="cuda",
            random_state=data_cfg.get("random_seed", 42),
        )
        classifier.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
        val_auc = float(roc_auc_score(y_val, classifier.predict_proba(X_val)[:, 1]))
        trial.set_user_attr("best_iteration", int(classifier.best_iteration))
        print(f"Trial {trial.number}: validation ROC-AUC={val_auc:.5f}, params={params}", flush=True)
        return val_auc

    pending = max(0, n_trials - sum(t.state == optuna.trial.TrialState.COMPLETE for t in study.trials))
    print(f"Running {pending} additional validation-only trials; completed={len(study.trials)}", flush=True)
    study.optimize(objective, n_trials=pending, gc_after_trial=True)
    best_params = dict(study.best_params)
    (tuning_results / "best_params.json").write_text(json.dumps({
        "objective": "validation_roc_auc",
        "best_validation_roc_auc": study.best_value,
        "params": best_params,
        "trials_completed": len(study.trials),
        "seed": data_cfg.get("random_seed", 42),
        "device": "cuda",
    }, indent=2), encoding="utf-8")
    study.trials_dataframe().to_csv(tuning_results / "trials.csv", index=False)

    # Refit the selected configuration on training data; validation controls early stopping/threshold.
    final_model = xgb.XGBClassifier(
        **best_params,
        scale_pos_weight=xgb_cfg.get("scale_pos_weight", 1.0),
        eval_metric="logloss",
        early_stopping_rounds=xgb_cfg.get("early_stopping_rounds", 35),
        tree_method="hist",
        device="cuda",
        random_state=data_cfg.get("random_seed", 42),
    )
    final_model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=50)
    val_probability = final_model.predict_proba(X_val)[:, 1]
    thresholds = np.arange(0.40, 0.70, 0.01)
    threshold = round(float(max(thresholds, key=lambda t: accuracy_score(y_val, val_probability >= t))), 2)

    X_test = load_embedding("test", len(test))
    del validation
    y_test = test[data_cfg["label_column"]].map(label_map).to_numpy(dtype=np.int8)
    test_probability = final_model.predict_proba(X_test)[:, 1]
    test_prediction = (test_probability >= threshold).astype(np.int8)
    report = evaluate_predictions(y_test, test_prediction, test_probability)
    report["metrics"].update({
        "optimal_threshold": threshold,
        "train_records": len(y_train),
        "validation_records": len(y_val),
        "test_records": len(y_test),
        "test_identity_sha256": _identity_digest(test),
        "tuning_objective": "validation_roc_auc",
        "best_validation_roc_auc": float(study.best_value),
        "tuning_trials": len(study.trials),
        "device": "cuda",
    })
    final_model.save_model(str(model_dir / "xgb_baseline_tuned.json"))
    (model_dir / "threshold_meta.json").write_text(json.dumps({"optimal_threshold": threshold}, indent=2), encoding="utf-8")
    save_evaluation_results(report, tuning_results)
    print("Final held-out test metrics:", json.dumps(report["metrics"], indent=2), flush=True)


def _identity_digest(frame: pd.DataFrame) -> str:
    payload = frame[["source", "project", "issue_id"]].astype(str).to_csv(index=False, header=False).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Validation-only XGBoost tuning on frozen CUDA baseline splits")
    parser.add_argument("--n-trials", type=int, default=15)
    parser.add_argument("--config", default="configs/baseline_fixed_split_config.yaml")
    parser.add_argument("--output-subdir", default="tuned_xgboost")
    args = parser.parse_args()
    tune_xgboost(n_trials=args.n_trials, config_path=args.config, output_subdir=args.output_subdir)
