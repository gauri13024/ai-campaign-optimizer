"""Multi-Channel Touchpoint & Ad Fatigue Simulation Engine.

Enriches customer data with multi-channel attributes across:
- Email
- Paid Search
- Social Media
- Display Ads
- Direct SMS

Models realistic marketing physics:
1. Channel Unit Costs (CPC, CPM, CPA)
2. Channel Affinity based on demographics and past behavior
3. Non-linear Ad Wear-in and Wear-out (Fatigue & Annoyance curves)
Grounded in Marketing Science (Rafieian et al., 2025) and MDPI (Abildaeva et al., 2026).
"""

from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd


DATA_DIR = Path(__file__).resolve().parent
TOUCHPOINTS_FILE_PATH = DATA_DIR / "multi_channel_touchpoints.csv"

# Channel Economics Configuration (Cost in USD/EUR)
CHANNEL_CONFIG = {
    "Email": {
        "cost_per_contact": 0.05,
        "base_ctr": 0.035,
        "base_cvr": 0.045,
        "fatigue_threshold": 4,       # contacts before wear-out steepens
        "fatigue_decay_rate": 0.22,
        "complaint_risk_base": 0.005,
    },
    "Paid_Search": {
        "cost_per_contact": 1.20,
        "base_ctr": 0.060,
        "base_cvr": 0.090,
        "fatigue_threshold": 6,
        "fatigue_decay_rate": 0.10,
        "complaint_risk_base": 0.001,
    },
    "Social_Media": {
        "cost_per_contact": 0.45,
        "base_ctr": 0.025,
        "base_cvr": 0.038,
        "fatigue_threshold": 5,
        "fatigue_decay_rate": 0.18,
        "complaint_risk_base": 0.008,
    },
    "Display_Ads": {
        "cost_per_contact": 0.15,
        "base_ctr": 0.012,
        "base_cvr": 0.020,
        "fatigue_threshold": 7,
        "fatigue_decay_rate": 0.25,
        "complaint_risk_base": 0.003,
    },
    "SMS": {
        "cost_per_contact": 0.25,
        "base_ctr": 0.090,
        "base_cvr": 0.075,
        "fatigue_threshold": 2,       # SMS wears out very quickly
        "fatigue_decay_rate": 0.40,
        "complaint_risk_base": 0.025,
    },
}


def compute_ad_fatigue_multiplier(frequency: int | np.ndarray, channel: str) -> np.ndarray:
    """Computes the response degradation multiplier f(frequency) in [0, 1.3].
    Models wear-in (initial repetition increases awareness) followed by
    exponential wear-out (ad saturation and cognitive fatigue).
    """
    cfg = CHANNEL_CONFIG[channel]
    thresh = cfg["fatigue_threshold"]
    decay = cfg["fatigue_decay_rate"]

    freq = np.asarray(frequency, dtype=float)
    # Wear-in curve up to threshold (rises slightly up to ~1.15x)
    wear_in = 1.0 + 0.15 * np.minimum(freq / max(thresh, 1), 1.0)
    # Wear-out penalty beyond threshold
    excess = np.maximum(freq - thresh, 0.0)
    wear_out = np.exp(-decay * excess)

    return np.clip(wear_in * wear_out, 0.05, 1.25)


def compute_churn_complaint_risk(frequency: int | np.ndarray, channel: str) -> np.ndarray:
    """Computes the probability of customer annoyance / complaint / unsubscribe
    as a function of contact frequency and channel intrusion level.
    """
    cfg = CHANNEL_CONFIG[channel]
    base_risk = cfg["complaint_risk_base"]
    freq = np.asarray(frequency, dtype=float)
    thresh = cfg["fatigue_threshold"]

    # Annoyance grows non-linearly when exceeding threshold
    excess = np.maximum(freq - thresh, 0.0)
    risk = base_risk * (1.0 + 1.8 * (excess ** 1.6))
    return np.clip(risk, 0.0, 0.65)


def generate_multi_channel_dataset(
    processed_df: pd.DataFrame,
    random_state: int = 42
) -> pd.DataFrame:
    """Transforms customer records into candidate customer-channel-offer allocations.
    For each customer and each potential marketing channel, calculates:
    - Channel Affinity Score
    - Contact Frequency
    - Channel Unit Cost
    - Fatigue Multiplier
    - Effective Conversion Probability
    - Expected Revenue
    - Predicted Churn / Complaint Penalty
    """
    rng = np.random.default_rng(random_state)
    records = []

    for _, row in processed_df.iterrows():
        cust_id = int(row["ID"])
        income = float(row["Income"])
        age = float(row["Age"])
        web_purchases = float(row["NumWebPurchases"])
        total_spend = float(row["TotalSpend"])
        ever_accepted = int(row["EverAcceptedCampaign"])
        estimated_clv = float(row["Estimated_CLV"])

        for channel, cfg in CHANNEL_CONFIG.items():
            # Channel affinity weighting based on demographics
            if channel == "Social_Media":
                affinity = 1.3 if age < 45 else 0.8
            elif channel == "Paid_Search":
                affinity = 1.2 if income > 55000 else 0.9
            elif channel == "Email":
                affinity = 1.2 if web_purchases > 3 else 0.85
            elif channel == "SMS":
                affinity = 1.1 if age < 50 else 0.7
            else:  # Display Ads
                affinity = 1.0

            # Historical baseline conversion
            base_prob = cfg["base_cvr"] * affinity * (1.3 if ever_accepted else 0.9)
            
            # Simulate historical exposure frequency (1 to 8 contacts)
            freq = int(rng.integers(1, 9))
            
            fatigue_mult = compute_ad_fatigue_multiplier(freq, channel)
            churn_risk = compute_churn_complaint_risk(freq, channel)
            effective_cvr = float(np.clip(base_prob * fatigue_mult, 0.005, 0.40))
            
            # Simulated outcome
            converted = int(rng.random() < effective_cvr)
            cost = cfg["cost_per_contact"] * freq
            # Value realized if converted
            revenue_generated = float(np.round(converted * rng.normal(loc=total_spend * 0.15 + 40, scale=15), 2)) if converted else 0.0

            records.append({
                "Customer_ID": cust_id,
                "Channel": channel,
                "Contact_Frequency": freq,
                "Channel_Cost": round(cost, 2),
                "Base_CVR": round(base_prob, 4),
                "Fatigue_Multiplier": round(fatigue_mult, 3),
                "Effective_CVR": round(effective_cvr, 4),
                "Converted": converted,
                "Revenue_Generated": max(0.0, revenue_generated),
                "Complaint_Churn_Risk": round(churn_risk, 4),
                "Customer_CLV": estimated_clv,
                "Customer_Income": income,
                "Customer_Age": age,
                "Customer_TotalSpend": total_spend,
            })

    touchpoint_df = pd.DataFrame(records)
    return touchpoint_df


def get_channel_touchpoint_data(processed_df: pd.DataFrame, force_recompute: bool = False) -> pd.DataFrame:
    """Returns the multi-channel touchpoint dataset, caching to disk."""
    if not force_recompute and TOUCHPOINTS_FILE_PATH.exists():
        return pd.read_csv(TOUCHPOINTS_FILE_PATH)

    touchpoint_df = generate_multi_channel_dataset(processed_df)
    touchpoint_df.to_csv(TOUCHPOINTS_FILE_PATH, index=False)
    print(f"[TouchpointEngine] Generated {len(touchpoint_df)} customer-channel touchpoints.")
    return touchpoint_df


if __name__ == "__main__":
    from data_loader import get_preprocessed_data
    df = get_preprocessed_data()
    tp_df = get_channel_touchpoint_data(df, force_recompute=True)
    print(f"Sample multi-channel touchpoints:\n{tp_df.head(10)}")
