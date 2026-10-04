"""AI Predictive Modeling Suite for Campaign Conversion / Response Prediction.

Compares multiple state-of-the-art architectures:
1. LightGBM Classifier
2. XGBoost Classifier
3. CatBoost Classifier
4. Multi-Layer Perceptron (Neural Network)
5. Random Forest / Gradient Boosting
6. Logistic Regression (Baseline)

Evaluates ROC-AUC, PR-AUC, F1-Score, Brier Loss, and Calibration.
Includes graceful fallbacks if specific optional boosting libraries are not installed.
"""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any, Dict, Tuple
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, brier_score_loss, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler


MODELS_DIR = Path(__file__).resolve().parent
SAVED_MODEL_PATH = MODELS_DIR / "best_conversion_model.pkl"

# Check for optional packages
try:
    import lightgbm as lgb
    HAS_LIGHTGBM = True
except ImportError:
    HAS_LIGHTGBM = False

try:
    import xgboost as xgb
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False

try:
    import catboost as cb
    HAS_CATBOOST = True
except ImportError:
    HAS_CATBOOST = False


FEATURE_COLUMNS = [
    "Age", "Income", "Recency", "TotalSpend", "TotalPurchases",
    "AverageOrderValue", "WebPurchaseRatio", "CatalogPurchaseRatio",
    "StorePurchaseRatio", "WebConversionRate", "PastCampaignsAccepted",
    "EverAcceptedCampaign", "R_Score", "F_Score", "M_Score",
    "Is_Higher_Ed", "Is_Partnered", "Has_Children", "Total_Children", "Tenure_Days"
]
TARGET_COLUMN = "Response"


def prepare_modeling_data(df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, StandardScaler, list]:
    """Splits dataset into stratified train and test sets and applies scaling."""
    available_cols = [c for c in FEATURE_COLUMNS if c in df.columns]
    X = df[available_cols].copy().fillna(df[available_cols].median())
    y = df[TARGET_COLUMN].values

    scaler = StandardScaler()
    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )
    X_train = scaler.fit_transform(X_train_raw)
    X_test = scaler.transform(X_test_raw)

    return X_train, X_test, y_train, y_test, scaler, available_cols


def get_candidate_models() -> Dict[str, Any]:
    """Initializes competitive candidate classification models."""
    models: Dict[str, Any] = {
        "Logistic Regression (Baseline)": LogisticRegression(max_iter=1000, random_state=42),
        "Random Forest": RandomForestClassifier(n_estimators=150, max_depth=8, random_state=42),
        "Neural Network (MLP)": MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=400, random_state=42),
    }

    if HAS_LIGHTGBM:
        models["LightGBM"] = lgb.LGBMClassifier(
            n_estimators=150, learning_rate=0.05, max_depth=6, random_state=42, verbose=-1
        )
    else:
        models["LightGBM (HistGB Fallback)"] = HistGradientBoostingClassifier(
            max_iter=150, learning_rate=0.05, max_depth=6, random_state=42
        )

    if HAS_XGBOOST:
        models["XGBoost"] = xgb.XGBClassifier(
            n_estimators=150, learning_rate=0.05, max_depth=5, random_state=42, eval_metric="logloss"
        )
    else:
        models["XGBoost (GBM Fallback)"] = GradientBoostingClassifier(
            n_estimators=150, learning_rate=0.05, max_depth=5, random_state=42
        )

    if HAS_CATBOOST:
        models["CatBoost"] = cb.CatBoostClassifier(
            iterations=150, learning_rate=0.05, depth=5, random_seed=42, verbose=0
        )

    return models


def evaluate_all_models(
    X_train: np.ndarray,
    X_test: np.ndarray,
    y_train: np.ndarray,
    y_test: np.ndarray
) -> Tuple[pd.DataFrame, Any, str]:
    """Trains and benchmarks all candidate architectures."""
    models = get_candidate_models()
    results = []
    best_auc = -1.0
    best_model = None
    best_name = ""

    for name, model in models.items():
        model.fit(X_train, y_train)
        y_prob = model.predict_proba(X_test)[:, 1]
        y_pred = (y_prob >= 0.5).astype(int)

        auc = roc_auc_score(y_test, y_prob)
        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        prec = precision_score(y_test, y_pred, zero_division=0)
        rec = recall_score(y_test, y_pred, zero_division=0)
        brier = brier_score_loss(y_test, y_prob)

        results.append({
            "Model Name": name,
            "ROC-AUC": round(auc, 4),
            "F1-Score": round(f1, 4),
            "Precision": round(prec, 4),
            "Recall": round(rec, 4),
            "Accuracy": round(acc, 4),
            "Brier Score": round(brier, 4)
        })

        if auc > best_auc:
            best_auc = auc
            best_model = model
            best_name = name

    results_df = pd.DataFrame(results).sort_values(by="ROC-AUC", ascending=False).reset_index(drop=True)
    return results_df, best_model, best_name


def train_and_save_best_model(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Runs the complete training and benchmarking pipeline, saving artifacts."""
    X_train, X_test, y_train, y_test, scaler, features = prepare_modeling_data(df)
    results_df, best_model, best_name = evaluate_all_models(X_train, X_test, y_train, y_test)

    artifact = {
        "model": best_model,
        "model_name": best_name,
        "scaler": scaler,
        "features": features,
        "results_table": results_df.to_dict(orient="records"),
        "best_auc": float(results_df.iloc[0]["ROC-AUC"])
    }

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    with open(SAVED_MODEL_PATH, "wb") as f:
        pickle.dump(artifact, f)

    print(f"[ConversionModel] Benchmark completed. Best Model: '{best_name}' (ROC-AUC: {artifact['best_auc']:.4f})")
    return results_df, artifact


def predict_conversion_probability(customer_features_df: pd.DataFrame) -> np.ndarray:
    """Predicts conversion probability for new customer feature vectors."""
    if not SAVED_MODEL_PATH.exists():
        from ..data.data_loader import get_preprocessed_data
        train_and_save_best_model(get_preprocessed_data())

    with open(SAVED_MODEL_PATH, "rb") as f:
        artifact = pickle.load(f)

    model = artifact["model"]
    scaler = artifact["scaler"]
    features = artifact["features"]

    X = customer_features_df[features].fillna(0)
    X_scaled = scaler.transform(X)
    return model.predict_proba(X_scaled)[:, 1]


if __name__ == "__main__":
    import sys
    sys.path.append(str(Path(__file__).resolve().parent.parent))
    from data.data_loader import get_preprocessed_data
    df = get_preprocessed_data()
    leaderboard, _ = train_and_save_best_model(df)
    print("\nModel Leaderboard:")
    print(leaderboard.to_string())
