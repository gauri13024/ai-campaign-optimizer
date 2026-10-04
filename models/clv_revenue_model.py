"""Predictive Regression Engine for Customer Lifetime Value (CLV) and Campaign Revenue.

Estimates expected monetary value conditional on campaign conversion.
Combines historical RFM metrics, basket composition, and income to forecast:
1. Short-term campaign incremental revenue
2. Long-term multi-year Customer Lifetime Value (CLV)
"""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any, Dict, Tuple
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


MODELS_DIR = Path(__file__).resolve().parent
SAVED_CLV_MODEL_PATH = MODELS_DIR / "best_clv_model.pkl"

REGRESSION_FEATURES = [
    "Income", "Age", "Recency", "TotalPurchases", "AverageOrderValue",
    "NumWebPurchases", "NumCatalogPurchases", "NumStorePurchases",
    "WebPurchaseRatio", "CatalogPurchaseRatio", "StorePurchaseRatio",
    "PastCampaignsAccepted", "R_Score", "F_Score", "M_Score",
    "Has_Children", "Total_Children", "Tenure_Days"
]
TARGET_CLV = "Estimated_CLV"


def prepare_regression_data(df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, StandardScaler, list]:
    available_cols = [c for c in REGRESSION_FEATURES if c in df.columns]
    X = df[available_cols].copy().fillna(df[available_cols].median())
    y = df[TARGET_CLV].values

    scaler = StandardScaler()
    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42
    )
    X_train = scaler.fit_transform(X_train_raw)
    X_test = scaler.transform(X_test_raw)

    return X_train, X_test, y_train, y_test, scaler, available_cols


def evaluate_clv_models(
    X_train: np.ndarray,
    X_test: np.ndarray,
    y_train: np.ndarray,
    y_test: np.ndarray
) -> Tuple[pd.DataFrame, Any, str]:
    models: Dict[str, Any] = {
        "Ridge Regression (Baseline)": Ridge(alpha=1.0, random_state=42),
        "Random Forest Regressor": RandomForestRegressor(n_estimators=120, max_depth=8, random_state=42),
        "Gradient Boosting Regressor": GradientBoostingRegressor(n_estimators=150, learning_rate=0.05, max_depth=5, random_state=42),
    }

    results = []
    best_rmse = float("inf")
    best_model = None
    best_name = ""

    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        rmse = float(np.sqrt(mean_squared_error(y_test, y_pred)))
        mae = float(mean_absolute_error(y_test, y_pred))
        r2 = float(r2_score(y_test, y_pred))

        results.append({
            "Model Name": name,
            "RMSE ($)": round(rmse, 2),
            "MAE ($)": round(mae, 2),
            "R2 Score": round(r2, 4)
        })

        if rmse < best_rmse:
            best_rmse = rmse
            best_model = model
            best_name = name

    results_df = pd.DataFrame(results).sort_values(by="RMSE ($)", ascending=True).reset_index(drop=True)
    return results_df, best_model, best_name


def train_and_save_clv_model(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    X_train, X_test, y_train, y_test, scaler, features = prepare_regression_data(df)
    results_df, best_model, best_name = evaluate_clv_models(X_train, X_test, y_train, y_test)

    artifact = {
        "model": best_model,
        "model_name": best_name,
        "scaler": scaler,
        "features": features,
        "results_table": results_df.to_dict(orient="records"),
        "best_rmse": float(results_df.iloc[0]["RMSE ($)"])
    }

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    with open(SAVED_CLV_MODEL_PATH, "wb") as f:
        pickle.dump(artifact, f)

    print(f"[CLVModel] Training completed. Best Model: '{best_name}' (RMSE: ${artifact['best_rmse']:.2f})")
    return results_df, artifact


def predict_clv(customer_features_df: pd.DataFrame) -> np.ndarray:
    if not SAVED_CLV_MODEL_PATH.exists():
        from ..data.data_loader import get_preprocessed_data
        train_and_save_clv_model(get_preprocessed_data())

    with open(SAVED_CLV_MODEL_PATH, "rb") as f:
        artifact = pickle.load(f)

    model = artifact["model"]
    scaler = artifact["scaler"]
    features = artifact["features"]

    X = customer_features_df[features].fillna(0)
    X_scaled = scaler.transform(X)
    return np.maximum(0.0, model.predict(X_scaled))


if __name__ == "__main__":
    import sys
    sys.path.append(str(Path(__file__).resolve().parent.parent))
    from data.data_loader import get_preprocessed_data
    df = get_preprocessed_data()
    leaderboard, _ = train_and_save_clv_model(df)
    print("\nCLV Regression Leaderboard:")
    print(leaderboard.to_string())
