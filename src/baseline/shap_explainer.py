from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import shap
import xgboost as xgb


class ShapExplainer:
    """Computes SHAP feature importance for the XGBoost baseline on DeBERTa embeddings."""

    def __init__(self, model: xgb.XGBClassifier):
        self.model = model
        self.explainer = shap.TreeExplainer(model)

    def explain(
        self,
        X: np.ndarray,
        sample_size: int = 500,
        figures_dir: str | Path | None = None,
    ) -> np.ndarray:
        if len(X) > sample_size:
            np.random.seed(42)
            indices = np.random.choice(len(X), size=sample_size, replace=False)
            X_sample = X[indices]
        else:
            X_sample = X

        print(f"Computing SHAP values for {len(X_sample)} samples...")
        shap_values = self.explainer.shap_values(X_sample)

        if figures_dir is not None:
            fig_path = Path(figures_dir)
            fig_path.mkdir(parents=True, exist_ok=True)

            feature_names = [f"emb_dim_{i}" for i in range(X.shape[1])]

            # 1. Beeswarm / summary plot
            plt.figure(figsize=(10, 8))
            shap.summary_plot(
                shap_values,
                X_sample,
                feature_names=feature_names,
                max_display=20,
                show=False,
            )
            plt.title("SHAP Feature Importance (Top 20 DeBERTa Dimensions)", fontsize=14, pad=15)
            plt.tight_layout()
            plt.savefig(fig_path / "shap_summary.png", dpi=300, bbox_inches="tight")
            plt.close()

            # 2. Bar plot of global importance
            plt.figure(figsize=(10, 6))
            shap.summary_plot(
                shap_values,
                X_sample,
                feature_names=feature_names,
                plot_type="bar",
                max_display=20,
                show=False,
            )
            plt.title("Global Feature Importance (Mean |SHAP Value|)", fontsize=14, pad=15)
            plt.tight_layout()
            plt.savefig(fig_path / "shap_importance_bar.png", dpi=300, bbox_inches="tight")
            plt.close()

            print(f"Generated SHAP plots at: {fig_path}")

        return shap_values
