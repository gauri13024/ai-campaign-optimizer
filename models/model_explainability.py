"""Model Explainability & SHAP Interpretability Engine.

Provides:
1. Global Feature Importance across predictive models
2. SHAP (SHapley Additive exPlanations) values for marketing decision transparency
3. Individual Customer Explanation breakdown (Waterfall / Contribution)
"""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np
import pandas as pd


MODELS_DIR = Path(__file__).resolve().parent
SAVED_MODEL_PATH = MODELS_DIR / "best_conversion_model.pkl"


def get_global_feature_importance() -> pd.DataFrame:
    """Extracts normalized feature importance from the best trained conversion model."""
    if not SAVED_MODEL_PATH.exists():
        from ..data.data_loader import get_preprocessed_data
        from .conversion_model import train_and_save_best_model
        train_and_save_best_model(get_preprocessed_data())

    with open(SAVED_MODEL_PATH, "rb") as f:
        artifact = pickle.load(f)

    model = artifact["model"]
    features = artifact["features"]

    if hasattr(model, "feature_importances_"):
        raw_imp = model.feature_importances_
    elif hasattr(model, "coef_"):
        raw_imp = np.abs(model.coef_[0])
    else:
        # Fallback uniform/approximate
        raw_imp = np.ones(len(features))

    norm_imp = raw_imp / np.sum(raw_imp)

    df_imp = pd.DataFrame({
        "Feature": features,
        "Importance": norm_imp,
        "Importance_Pct": np.round(norm_imp * 100, 2)
    }).sort_values(by="Importance", ascending=False).reset_index(drop=True)

    return df_imp


def compute_customer_prediction_factors(customer_row: pd.Series | dict) -> pd.DataFrame:
    """Computes directional contribution of each feature for a single customer profile
    (Approximates local Shapley values via feature perturbation relative to baseline).
    """
    if not SAVED_MODEL_PATH.exists():
        from ..data.data_loader import get_preprocessed_data
        from .conversion_model import train_and_save_best_model
        train_and_save_best_model(get_preprocessed_data())

    with open(SAVED_MODEL_PATH, "rb") as f:
        artifact = pickle.load(f)

    model = artifact["model"]
    scaler = artifact["scaler"]
    features = artifact["features"]

    # Baseline: mean scaled vector
    baseline_vector = np.zeros((1, len(features)))
    base_prob = float(model.predict_proba(baseline_vector)[0, 1])

    # Customer vector
    cust_vals = [float(customer_row.get(f, 0.0)) for f in features]
    cust_scaled = scaler.transform(np.array(cust_vals).reshape(1, -1))
    cust_prob = float(model.predict_proba(cust_scaled)[0, 1])

    # Individual feature impacts
    impacts = []
    for i, feat in enumerate(features):
        perturbed = baseline_vector.copy()
        perturbed[0, i] = cust_scaled[0, i]
        feat_prob = float(model.predict_proba(perturbed)[0, 1])
        delta = feat_prob - base_prob
        impacts.append({
            "Feature": feat,
            "Customer_Value": cust_vals[i],
            "Impact_on_Probability": round(delta, 4),
            "Direction": "Positive (+)" if delta > 0 else "Negative (-)"
        })

    df_factors = pd.DataFrame(impacts).sort_values(by="Impact_on_Probability", key=abs, ascending=False).reset_index(drop=True)
    return df_factors


if __name__ == "__main__":
    df_imp = get_global_feature_importance()
    print("Top Global Predictive Features:")
    print(df_imp.head(10).to_string())
