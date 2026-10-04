"""Dataset Comparison & Survey Module for Digital Marketing Campaign Analytics.

Implements the comparative evaluation across four standard benchmark datasets:
1. Criteo Display Advertising Challenge (Kaggle/Criteo Labs)
2. Avazu Click-Through Rate Prediction (Kaggle/Avazu)
3. Bank Marketing Dataset (UCI Machine Learning Repository)
4. Marketing Campaign Dataset (Kaggle / Rodolfo Saldanha)

Evaluates:
- Dimensionality & Sparsity
- Target Variables (Binary CTR vs. Conversion vs. Multi-Objective Value)
- Campaign Context (Channel, Ad format, Bid)
- Customer Demographics & Historical Behavior
- Sequential & Temporal Dynamics
- Suitability for Multi-Objective Decision Making
"""

from __future__ import annotations

import pandas as pd


DATASET_SURVEY_DATA = [
    {
        "Dataset Name": "Criteo Display Advertising",
        "Source": "Criteo Labs / Kaggle (2014)",
        "Scale": "~45 Million records",
        "Features": "39 features (13 integer count features, 26 anonymized categorical hashed features)",
        "Target Variable": "Binary Click indicator (Click = 0 or 1)",
        "Campaign Info": "Implicit display impression bid context; anonymized publisher & ad exchange IDs",
        "Customer Info": "High-dimensional anonymized hashed user cookies; no explicit demographics",
        "Sequential Info": "Chronological ad request logs over 7 days",
        "Key Strengths": "Benchmark gold standard for massive-scale deep CTR models (DLRM, DeepFM, xDeepFM)",
        "Limitations for Multi-Objective": "Purely single-objective (CTR only). Lacks post-click conversion, revenue, margin, and customer satisfaction."
    },
    {
        "Dataset Name": "Avazu Mobile CTR",
        "Source": "Avazu Inc. / Kaggle (2014)",
        "Scale": "~40 Million mobile ad impressions",
        "Features": "24 features (id, click, hour, C1, banner_pos, site_id, site_domain, app_id, device_id, etc.)",
        "Target Variable": "Binary Click indicator (Click = 0 or 1)",
        "Campaign Info": "Ad banner position, mobile site/app category, ad placement context",
        "Customer Info": "Device ID, device IP hash, device model, device connection type",
        "Sequential Info": "Explicit timestamp (YYMMDDHH format) across 10 consecutive days",
        "Key Strengths": "Captures mobile in-app and mobile web ad dynamics; temporal daily/hourly seasonality",
        "Limitations for Multi-Objective": "No economic cost/revenue figures. Cannot model customer lifetime value, multi-channel synergy, or ad wear-out."
    },
    {
        "Dataset Name": "Bank Marketing Dataset",
        "Source": "UCI ML Repository / Portuguese Banking Institution (Moro et al., 2014)",
        "Scale": "45,211 phone contact instances",
        "Features": "16 input attributes (age, job, marital, education, balance, housing, loan, contact, day, month, duration, campaign, pdays, previous, poutcome)",
        "Target Variable": "Term Deposit Subscription ('yes' / 'no')",
        "Campaign Info": "Contact communication type (cellular/telephone), number of contacts during campaign, days passed",
        "Customer Info": "Rich socio-demographic profile (job, education, housing loan, credit default)",
        "Sequential Info": "Previous campaign outcome (poutcome) and elapsed days (pdays)",
        "Key Strengths": "Contains contact frequency counters and explicit negative outcomes (wear-out/rejection)",
        "Limitations for Multi-Objective": "Call duration leak (`duration` heavily affects target but unknown before call); single channel (telemarketing)."
    },
    {
        "Dataset Name": "Marketing Campaign Dataset (Selected)",
        "Source": "Kaggle (Rodolfo Saldanha / Customer Personality Analysis)",
        "Scale": "2,240 detailed customer profiles across multiple campaign waves",
        "Features": "29 features (Demographics, RFM attributes, product category spends, deal/channel purchase counts, past campaign responses)",
        "Target Variable": "Multi-Dimensional: Latest Campaign Response, Product Spend Breakdown, Past Acceptance (Cmp1-5), Churn/Complaints",
        "Campaign Info": "Multi-campaign historical response indicators (AcceptedCmp1..5), contact costs (Z_CostContact), unit revenue (Z_Revenue)",
        "Customer Info": "Year of birth, Education, Marital status, Household income, Children/Teens, Customer tenure date",
        "Sequential Info": "Recency (days since last purchase), customer tenure, multi-wave campaign history",
        "Key Strengths": "Ideal for Multi-Objective Optimization: simultaneously evaluates Conversion, Monetary Spend/CLV, Channel Efficiency, and Customer Fatigue.",
        "Limitations for Multi-Objective": "Moderate scale; requires channel-level touchpoint synthesis for multi-channel programmatic simulations."
    }
]


def get_dataset_survey_dataframe() -> pd.DataFrame:
    """Returns the comparative survey table as a pandas DataFrame."""
    return pd.DataFrame(DATASET_SURVEY_DATA)


def generate_markdown_survey_report() -> str:
    """Generates a structured markdown comparison matrix for the project documentation."""
    df = get_dataset_survey_dataframe()
    md = "# Digital Marketing Datasets: Comparative Survey & Evaluation\n\n"
    md += "A comprehensive comparison across the four candidate datasets identified for campaign optimization.\n\n"
    md += df.to_markdown(index=False)
    md += "\n\n### Justification for Selected Primary Dataset:\n"
    md += "- While **Criteo** and **Avazu** are industry benchmarks for raw binary CTR prediction at internet scale, they are fundamentally **single-objective** and lack financial metrics, customer demographics, and post-click lifetime value.\n"
    md += "- **Bank Marketing** captures contact fatigue well but is limited to a single telemarketing channel.\n"
    md += "- The **Kaggle Marketing Campaign Dataset** provides the rich multi-dimensional granularity (demographics, multi-category spend, multi-wave responses, and RFM attributes) necessary to execute **multi-objective Pareto optimization** (maximizing CLV and engagement while minimizing campaign costs and ad fatigue).\n"
    return md


if __name__ == "__main__":
    df = get_dataset_survey_dataframe()
    print("Dataset Comparison Survey:")
    print(df[["Dataset Name", "Scale", "Target Variable", "Key Strengths"]])
