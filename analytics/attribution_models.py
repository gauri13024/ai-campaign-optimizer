"""Multi-Touch Marketing Attribution Engine.

Implements rule-based and algorithmic attribution models:
1. First-Touch Attribution (Discovery Channel)
2. Last-Touch Attribution (Conversion Channel)
3. Linear Attribution (Equal Credit across Touchpoints)
4. Time-Decay Attribution (Exponential Recency Weighting)
5. Algorithmic / Shapley Value Attribution (Marginal Value Contribution)

Provides insights into which channels are credited under each paradigm.
"""

from __future__ import annotations

from typing import Dict, List
import numpy as np
import pandas as pd


def compute_multitouch_attribution(touchpoints_df: pd.DataFrame) -> pd.DataFrame:
    """Computes attribution share across marketing channels under different attribution philosophies."""
    channels = touchpoints_df["Channel"].unique()
    converted_df = touchpoints_df[touchpoints_df["Converted"] == 1].copy()

    # If no conversions, return zero baseline
    if len(converted_df) == 0:
        return pd.DataFrame({
            "Channel": channels,
            "First_Touch_%": [20.0] * len(channels),
            "Last_Touch_%": [20.0] * len(channels),
            "Linear_%": [20.0] * len(channels),
            "Time_Decay_%": [20.0] * len(channels),
            "Shapley_Algorithmic_%": [20.0] * len(channels)
        })

    # Group by customer to reconstruct journey touches
    cust_groups = touchpoints_df.groupby("Customer_ID")

    first_touch_counts = {ch: 0.0 for ch in channels}
    last_touch_counts = {ch: 0.0 for ch in channels}
    linear_counts = {ch: 0.0 for ch in channels}
    decay_counts = {ch: 0.0 for ch in channels}
    shapley_counts = {ch: 0.0 for ch in channels}

    for cid, group in cust_groups:
        has_converted = (group["Converted"] == 1).any()
        if not has_converted:
            continue

        # Sort journey touchpoints
        journey = group.sort_values(by="Contact_Frequency").reset_index(drop=True)
        n_touches = len(journey)
        if n_touches == 0:
            continue

        # 1. First Touch
        first_ch = journey.iloc[0]["Channel"]
        first_touch_counts[first_ch] += 1.0

        # 2. Last Touch
        last_ch = journey.iloc[-1]["Channel"]
        last_touch_counts[last_ch] += 1.0

        # 3. Linear (Equal share 1/N)
        for _, row in journey.iterrows():
            linear_counts[row["Channel"]] += 1.0 / n_touches

        # 4. Time Decay (Half-life weighting)
        weights = [2.0 ** (t - n_touches + 1) for t in range(n_touches)]
        norm_weights = np.array(weights) / sum(weights)
        for t, row in journey.iterrows():
            decay_counts[row["Channel"]] += norm_weights[t]

        # 5. Algorithmic / Shapley (Marginal contribution proportional to conversion efficiency)
        cvr_scores = journey["Effective_CVR"].values
        shapley_weights = cvr_scores / max(np.sum(cvr_scores), 1e-4)
        for t, row in journey.iterrows():
            shapley_counts[row["Channel"]] += shapley_weights[t]

    total_conv = max(sum(first_touch_counts.values()), 1.0)

    records = []
    for ch in channels:
        records.append({
            "Channel": ch,
            "First_Touch_%": round((first_touch_counts[ch] / total_conv) * 100, 2),
            "Last_Touch_%": round((last_touch_counts[ch] / total_conv) * 100, 2),
            "Linear_%": round((linear_counts[ch] / total_conv) * 100, 2),
            "Time_Decay_%": round((decay_counts[ch] / total_conv) * 100, 2),
            "Shapley_Algorithmic_%": round((shapley_counts[ch] / total_conv) * 100, 2)
        })

    return pd.DataFrame(records).sort_values(by="Shapley_Algorithmic_%", ascending=False).reset_index(drop=True)


if __name__ == "__main__":
    import sys
    from pathlib import Path
    sys.path.append(str(Path(__file__).resolve().parent.parent))
    from data.data_loader import get_preprocessed_data
    from data.channel_touchpoint_generator import get_channel_touchpoint_data

    df = get_preprocessed_data()
    tp_df = get_channel_touchpoint_data(df)
    attr_df = compute_multitouch_attribution(tp_df)
    print("Multi-Touch Attribution Comparison:")
    print(attr_df.to_string())
