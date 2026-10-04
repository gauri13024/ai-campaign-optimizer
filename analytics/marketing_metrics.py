"""Marketing Metrics Engine for Campaign Performance Evaluation.

Implements standard digital marketing metrics and KPIs:
- Click-Through Rate (CTR)
- Cost-Per-Click (CPC)
- Cost-Per-Acquisition (CPA / CAC)
- Conversion Rate (CVR)
- Return on Ad Spend (ROAS)
- Return on Investment (ROI %)
- Customer Lifetime Value (CLV)
- CLV-to-CAC Ratio (Unit Economics Health)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Any
import numpy as np
import pandas as pd


@dataclass
class CampaignPerformanceMetrics:
    impressions: int
    clicks: int
    conversions: int
    spend: float
    revenue: float
    total_customers: int

    @property
    def ctr(self) -> float:
        """Click-Through Rate (%)"""
        return (self.clicks / max(self.impressions, 1)) * 100.0

    @property
    def cpc(self) -> float:
        """Cost Per Click ($)"""
        return self.spend / max(self.clicks, 1)

    @property
    def cvr(self) -> float:
        """Conversion Rate (%)"""
        return (self.conversions / max(self.clicks, 1)) * 100.0

    @property
    def cpa(self) -> float:
        """Cost Per Acquisition / CPA ($)"""
        return self.spend / max(self.conversions, 1)

    @property
    def roas(self) -> float:
        """Return On Ad Spend (Ratio)"""
        return self.revenue / max(self.spend, 1e-4)

    @property
    def roi(self) -> float:
        """Return On Investment (%)"""
        return ((self.revenue - self.spend) / max(self.spend, 1e-4)) * 100.0

    @property
    def average_order_value(self) -> float:
        """AOV ($)"""
        return self.revenue / max(self.conversions, 1)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "Impressions": f"{self.impressions:,}",
            "Clicks": f"{self.clicks:,}",
            "Conversions": f"{self.conversions:,}",
            "Total Spend": f"${self.spend:,.2f}",
            "Total Revenue": f"${self.revenue:,.2f}",
            "CTR (%)": f"{self.ctr:.2f}%",
            "CPC ($)": f"${self.cpc:.2f}",
            "CVR (%)": f"{self.cvr:.2f}%",
            "CPA ($)": f"${self.cpa:.2f}",
            "ROAS": f"{self.roas:.2f}x",
            "ROI (%)": f"{self.roi:.1f}%",
            "AOV ($)": f"${self.average_order_value:.2f}"
        }


def compute_channel_performance_breakdown(touchpoints_df: pd.DataFrame) -> pd.DataFrame:
    """Computes comparative performance breakdown across marketing channels."""
    records = []
    for channel, group in touchpoints_df.groupby("Channel"):
        total_contacts = int(group["Contact_Frequency"].sum())
        total_cost = float(group["Channel_Cost"].sum())
        conversions = int(group["Converted"].sum())
        revenue = float(group["Revenue_Generated"].sum())
        avg_fatigue = float(group["Complaint_Churn_Risk"].mean())

        # Estimated clicks assuming CTR ~ 1.5x of CVR
        estimated_clicks = int(conversions * 1.8 + total_contacts * 0.03)

        ctr = (estimated_clicks / max(total_contacts, 1)) * 100.0
        cpc = total_cost / max(estimated_clicks, 1)
        cpa = total_cost / max(conversions, 1)
        roas = revenue / max(total_cost, 1e-4)
        roi = ((revenue - total_cost) / max(total_cost, 1e-4)) * 100.0

        records.append({
            "Channel": channel,
            "Total Contacts": total_contacts,
            "Estimated Clicks": estimated_clicks,
            "Conversions": conversions,
            "Spend ($)": round(total_cost, 2),
            "Revenue ($)": round(revenue, 2),
            "CTR (%)": round(ctr, 2),
            "CPC ($)": round(cpc, 2),
            "CPA ($)": round(cpa, 2),
            "ROAS": round(roas, 2),
            "ROI (%)": round(roi, 1),
            "Avg Fatigue Risk": round(avg_fatigue, 3)
        })

    return pd.DataFrame(records).sort_values(by="ROAS", ascending=False).reset_index(drop=True)


if __name__ == "__main__":
    import sys
    from pathlib import Path
    sys.path.append(str(Path(__file__).resolve().parent.parent))
    from data.data_loader import get_preprocessed_data
    from data.channel_touchpoint_generator import get_channel_touchpoint_data

    df = get_preprocessed_data()
    tp_df = get_channel_touchpoint_data(df)
    breakdown = compute_channel_performance_breakdown(tp_df)
    print("Channel Performance Breakdown:")
    print(breakdown.to_string())
