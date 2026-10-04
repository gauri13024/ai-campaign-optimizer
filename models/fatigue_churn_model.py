"""Ad Fatigue & Churn Risk Prediction Engine.

Models the negative externalities of marketing over-exposure:
- Unsubscribe / Spam complaints
- Negative brand sentiment
- Churn hazard rate as a function of contact frequency across channels

Provides the critical 4th objective in our Multi-Objective Optimization:
Minimizing Customer Fatigue & Brand Dissatisfaction.
"""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any, Dict
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler


MODELS_DIR = Path(__file__).resolve().parent
SAVED_FATIGUE_MODEL_PATH = MODELS_DIR / "best_fatigue_model.pkl"


def train_fatigue_risk_model(touchpoints_df: pd.DataFrame) -> Dict[str, Any]:
    """Trains a classifier predicting customer churn/complaint probability
    conditioned on channel, contact frequency, customer age, and spend.
    """
    df = touchpoints_df.copy()
    
    # Target: High complaint / churn hazard indicator
    # Threshold at median or defined risk
    df["High_Risk_Target"] = (df["Complaint_Churn_Risk"] > 0.05).astype(int)

    # Encode channel
    channel_dummies = pd.get_dummies(df["Channel"], prefix="Ch")
    X = pd.concat([
        df[["Contact_Frequency", "Customer_Income", "Customer_Age", "Customer_TotalSpend"]],
        channel_dummies
    ], axis=1)
    y = df["High_Risk_Target"].values

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    clf = RandomForestClassifier(n_estimators=100, max_depth=6, random_state=42)
    clf.fit(X_scaled, y)

    artifact = {
        "model": clf,
        "scaler": scaler,
        "features": list(X.columns),
        "channel_columns": list(channel_dummies.columns)
    }

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    with open(SAVED_FATIGUE_MODEL_PATH, "wb") as f:
        pickle.dump(artifact, f)

    print(f"[FatigueModel] Trained fatigue prediction model successfully.")
    return artifact


def predict_fatigue_hazard(
    contact_frequencies: np.ndarray | list,
    channels: list[str],
    incomes: np.ndarray | list,
    ages: np.ndarray | list,
    spends: np.ndarray | list
) -> np.ndarray:
    """Predicts the continuous fatigue hazard score in [0, 1]."""
    if not SAVED_FATIGUE_MODEL_PATH.exists():
        from ..data.data_loader import get_preprocessed_data
        from ..data.channel_touchpoint_generator import get_channel_touchpoint_data
        df = get_preprocessed_data()
        tp_df = get_channel_touchpoint_data(df)
        train_fatigue_risk_model(tp_df)

    with open(SAVED_FATIGUE_MODEL_PATH, "rb") as f:
        artifact = pickle.load(f)

    clf = artifact["model"]
    scaler = artifact["scaler"]
    feature_names = artifact["features"]

    n = len(contact_frequencies)
    df_eval = pd.DataFrame({
        "Contact_Frequency": contact_frequencies,
        "Customer_Income": incomes,
        "Customer_Age": ages,
        "Customer_TotalSpend": spends
    })

    # One-hot encode channel match
    for ch_col in artifact["channel_columns"]:
        ch_name = ch_col.replace("Ch_", "")
        df_eval[ch_col] = [1 if ch == ch_name else 0 for ch in channels]

    X = df_eval[feature_names].fillna(0)
    X_scaled = scaler.transform(X)
    return clf.predict_proba(X_scaled)[:, 1]


if __name__ == "__main__":
    import sys
    sys.path.append(str(Path(__file__).resolve().parent.parent))
    from data.data_loader import get_preprocessed_data
    from data.channel_touchpoint_generator import get_channel_touchpoint_data
    df = get_preprocessed_data()
    tp_df = get_channel_touchpoint_data(df)
    train_fatigue_risk_model(tp_df)
    
    # Test sample hazard
    test_hazards = predict_fatigue_hazard(
        contact_frequencies=[1, 4, 8],
        channels=["Email", "SMS", "SMS"],
        incomes=[50000, 50000, 50000],
        ages=[35, 35, 35],
        spends=[400, 400, 400]
    )
    print("Test Fatigue Hazard Scores:", test_hazards)
