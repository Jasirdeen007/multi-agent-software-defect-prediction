from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import xgboost as xgb
from sklearn.metrics import accuracy_score


class XGBoostBaselineClassifier:
    """XGBoost classifier with strong regularization (L1/L2, gamma, feature subsampling)
    and validation threshold tuning to prevent overfitting and boost accuracy."""

    def __init__(
        self,
        n_estimators: int = 500,
        max_depth: int = 4,
        learning_rate: float = 0.05,
        subsample: float = 0.7,
        colsample_bytree: float = 0.6,
        reg_alpha: float = 2.0,
        reg_lambda: float = 10.0,
        gamma: float = 1.0,
        min_child_weight: float = 3.0,
        scale_pos_weight: float | str = 1.0,
        eval_metric: str = "logloss",
        early_stopping_rounds: int = 35,
        tree_method: str = "hist",
        device: str = "cuda",
        random_state: int = 42,
    ):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.subsample = subsample
        self.colsample_bytree = colsample_bytree
        self.reg_alpha = reg_alpha
        self.reg_lambda = reg_lambda
        self.gamma = gamma
        self.min_child_weight = min_child_weight
        self.scale_pos_weight = scale_pos_weight
        self.eval_metric = eval_metric
        self.early_stopping_rounds = early_stopping_rounds
        self.tree_method = tree_method
        self.device = device
        self.random_state = random_state
        self.model: xgb.XGBClassifier | None = None
        self.optimal_threshold: float = 0.50

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
    ) -> XGBoostBaselineClassifier:
        if self.scale_pos_weight == "auto":
            neg_count = int((y_train == 0).sum())
            pos_count = int((y_train == 1).sum())
            computed_weight = neg_count / max(pos_count, 1)
            print(f"Computed scale_pos_weight (neg/pos = {neg_count}/{pos_count}): {computed_weight:.4f}")
            actual_scale_weight = float(computed_weight)
        else:
            actual_scale_weight = float(self.scale_pos_weight)

        self.model = xgb.XGBClassifier(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            learning_rate=self.learning_rate,
            subsample=self.subsample,
            colsample_bytree=self.colsample_bytree,
            reg_alpha=self.reg_alpha,
            reg_lambda=self.reg_lambda,
            gamma=self.gamma,
            min_child_weight=self.min_child_weight,
            scale_pos_weight=actual_scale_weight,
            eval_metric=self.eval_metric,
            early_stopping_rounds=self.early_stopping_rounds,
            tree_method=self.tree_method,
            device=self.device,
            random_state=self.random_state,
        )

        print(f"Training Regularized XGBoost with {X_train.shape[0]:,} samples, {X_train.shape[1]} features (device={self.device})...")
        print(f"Regularization: max_depth={self.max_depth}, alpha={self.reg_alpha}, lambda={self.reg_lambda}, gamma={self.gamma}, colsample={self.colsample_bytree}")
        
        self.model.fit(
            X_train,
            y_train,
            eval_set=[(X_train, y_train), (X_val, y_val)],
            verbose=50,
        )
        print(f"Training completed. Best iteration: {self.model.best_iteration}")

        # Optimize classification probability threshold on validation set
        val_proba = self.model.predict_proba(X_val)[:, 1]
        best_th = 0.50
        best_acc = 0.0
        for th in np.arange(0.40, 0.70, 0.01):
            acc = accuracy_score(y_val, (val_proba >= th).astype(int))
            if acc > best_acc:
                best_acc = acc
                best_th = round(float(th), 2)

        self.optimal_threshold = best_th
        print(f"Optimized validation threshold: {self.optimal_threshold:.2f} (Validation Accuracy: {best_acc:.4f})")
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("Model is not fitted yet.")
        return self.model.predict_proba(X)

    def predict(self, X: np.ndarray, threshold: float | None = None) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("Model is not fitted yet.")
        th = self.optimal_threshold if threshold is None else threshold
        probas = self.predict_proba(X)[:, 1]
        return (probas >= th).astype(int)

    def save_model(self, model_path: str | Path) -> None:
        if self.model is None:
            raise RuntimeError("Model is not fitted yet.")
        path = Path(model_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self.model.save_model(str(path))
        # Save threshold metadata
        meta_path = path.parent / "threshold_meta.json"
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump({"optimal_threshold": self.optimal_threshold}, f, indent=2)
        print(f"Saved XGBoost model to: {path}")

    def load_model(self, model_path: str | Path) -> None:
        self.model = xgb.XGBClassifier()
        self.model.load_model(str(model_path))
        meta_path = Path(model_path).parent / "threshold_meta.json"
        if meta_path.exists():
            with open(meta_path, "r", encoding="utf-8") as f:
                self.optimal_threshold = json.load(f).get("optimal_threshold", 0.50)
        print(f"Loaded XGBoost model from: {model_path} (threshold={self.optimal_threshold})")
