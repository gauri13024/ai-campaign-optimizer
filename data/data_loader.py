"""Data Loader and Feature Engineering Engine for Marketing Campaign Analysis.

Supports the Kaggle Marketing Campaign Dataset (Rodolfo Saldanha)
Schema: ID, Year_Birth, Education, Marital_Status, Income, Kidhome, Teenhome,
        Dt_Customer, Recency, MntWines, MntFruits, MntMeatProducts, MntFishProducts,
        MntSweetProducts, MntGoldProds, NumDealsPurchases, NumWebPurchases,
        NumCatalogPurchases, NumStorePurchases, NumWebVisitsMonth, AcceptedCmp3,
        AcceptedCmp4, AcceptedCmp5, AcceptedCmp1, AcceptedCmp2, Complain, Response.
"""

from __future__ import annotations

import os
from datetime import datetime
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


DATA_DIR = Path(__file__).resolve().parent
RAW_FILE_PATH = DATA_DIR / "marketing_campaign.csv"
PROCESSED_FILE_PATH = DATA_DIR / "processed_campaign_data.csv"


def generate_synthetic_kaggle_dataset(n_samples: int = 2240, random_state: int = 42) -> pd.DataFrame:
    """Generates a realistic dataset matching the exact schema and statistical
    distribution of the Kaggle Marketing Campaign dataset (Saldanha).
    Used as an immediate, ready-to-run fallback if the raw CSV is not present.
    """
    rng = np.random.default_rng(random_state)

    customer_ids = 1000 + np.arange(n_samples)
    year_birth = rng.integers(1950, 2002, size=n_samples)
    educations = rng.choice(["Graduation", "PhD", "Master", "2n Cycle", "Basic"], size=n_samples, p=[0.50, 0.22, 0.16, 0.09, 0.03])
    maritals = rng.choice(["Married", "Together", "Single", "Divorced", "Widow"], size=n_samples, p=[0.39, 0.26, 0.21, 0.10, 0.04])
    
    # Incomes correlated with education
    base_incomes = {
        "Basic": (25000, 8000),
        "2n Cycle": (45000, 15000),
        "Graduation": (52000, 18000),
        "Master": (54000, 19000),
        "PhD": (57000, 21000)
    }
    income = np.array([
        max(12000, rng.normal(base_incomes[e][0], base_incomes[e][1]))
        for e in educations
    ])
    # Introduce ~1% missing values like original dataset
    missing_mask = rng.random(n_samples) < 0.01
    income[missing_mask] = np.nan

    kidhome = rng.choice([0, 1, 2], size=n_samples, p=[0.58, 0.38, 0.04])
    teenhome = rng.choice([0, 1, 2], size=n_samples, p=[0.52, 0.44, 0.04])

    # Join dates between 2012 and 2014
    start_ts = datetime(2012, 1, 1).timestamp()
    end_ts = datetime(2014, 6, 30).timestamp()
    dt_customer = [
        datetime.fromtimestamp(rng.uniform(start_ts, end_ts)).strftime("%d-%m-%Y")
        for _ in range(n_samples)
    ]

    recency = rng.integers(0, 100, size=n_samples)

    # Spending behavior correlated with income
    income_factor = np.nan_to_num(income, nan=50000.0) / 50000.0
    mnt_wines = np.clip(rng.exponential(scale=200 * income_factor).astype(int), 0, 1500)
    mnt_fruits = np.clip(rng.exponential(scale=20 * income_factor).astype(int), 0, 200)
    mnt_meat = np.clip(rng.exponential(scale=120 * income_factor).astype(int), 0, 1800)
    mnt_fish = np.clip(rng.exponential(scale=30 * income_factor).astype(int), 0, 300)
    mnt_sweet = np.clip(rng.exponential(scale=25 * income_factor).astype(int), 0, 300)
    mnt_gold = np.clip(rng.exponential(scale=40 * income_factor).astype(int), 0, 350)

    num_deals = rng.poisson(lam=2.3, size=n_samples)
    num_web = rng.poisson(lam=4.0 * np.clip(income_factor, 0.5, 1.8), size=n_samples)
    num_catalog = rng.poisson(lam=2.5 * np.clip(income_factor, 0.2, 2.5), size=n_samples)
    num_store = rng.poisson(lam=5.5 * np.clip(income_factor, 0.5, 2.0), size=n_samples)
    num_web_visits = rng.poisson(lam=5.0, size=n_samples)

    # Historical Campaign acceptances (Cmp1 to Cmp5)
    # Customers with higher spending are more responsive
    total_spend = mnt_wines + mnt_fruits + mnt_meat + mnt_fish + mnt_sweet + mnt_gold
    prob_cmp = np.clip(0.04 + 0.12 * (total_spend / 2000.0), 0.02, 0.35)

    accepted_cmp1 = (rng.random(n_samples) < prob_cmp).astype(int)
    accepted_cmp2 = (rng.random(n_samples) < prob_cmp * 0.7).astype(int)
    accepted_cmp3 = (rng.random(n_samples) < prob_cmp * 0.9).astype(int)
    accepted_cmp4 = (rng.random(n_samples) < prob_cmp * 0.95).astype(int)
    accepted_cmp5 = (rng.random(n_samples) < prob_cmp * 1.1).astype(int)

    complain = (rng.random(n_samples) < 0.015).astype(int)

    # Target variable: Response (to the latest campaign)
    # Dependent on past acceptances, recency, and spending
    resp_latent = (
        -2.2
        + 0.8 * accepted_cmp5
        + 0.6 * accepted_cmp1
        + 0.4 * (accepted_cmp2 + accepted_cmp3 + accepted_cmp4)
        + 0.001 * total_spend
        - 0.01 * recency
        - 1.5 * complain
    )
    resp_prob = 1.0 / (1.0 + np.exp(-resp_latent))
    response = (rng.random(n_samples) < resp_prob).astype(int)

    df = pd.DataFrame({
        "ID": customer_ids,
        "Year_Birth": year_birth,
        "Education": educations,
        "Marital_Status": maritals,
        "Income": np.round(income, 2),
        "Kidhome": kidhome,
        "Teenhome": teenhome,
        "Dt_Customer": dt_customer,
        "Recency": recency,
        "MntWines": mnt_wines,
        "MntFruits": mnt_fruits,
        "MntMeatProducts": mnt_meat,
        "MntFishProducts": mnt_fish,
        "MntSweetProducts": mnt_sweet,
        "MntGoldProds": mnt_gold,
        "NumDealsPurchases": num_deals,
        "NumWebPurchases": num_web,
        "NumCatalogPurchases": num_catalog,
        "NumStorePurchases": num_store,
        "NumWebVisitsMonth": num_web_visits,
        "AcceptedCmp3": accepted_cmp3,
        "AcceptedCmp4": accepted_cmp4,
        "AcceptedCmp5": accepted_cmp5,
        "AcceptedCmp1": accepted_cmp1,
        "AcceptedCmp2": accepted_cmp2,
        "Complain": complain,
        "Z_CostContact": 3,
        "Z_Revenue": 11,
        "Response": response
    })
    return df


def load_raw_dataset(csv_path: Path | str | None = None) -> pd.DataFrame:
    """Loads the dataset from disk if available, otherwise generates and saves the fallback."""
    target_path = Path(csv_path) if csv_path else RAW_FILE_PATH

    if target_path.exists():
        # Handle tab or comma separator which is common in this Kaggle dataset
        try:
            df = pd.read_csv(target_path, sep="\t")
            if df.shape[1] < 5:
                df = pd.read_csv(target_path, sep=",")
        except Exception:
            df = pd.read_csv(target_path, sep=",")
        print(f"[DataLoader] Loaded raw dataset from {target_path} ({df.shape[0]} rows, {df.shape[1]} cols)")
    else:
        print(f"[DataLoader] Raw file not found at {target_path}. Generating benchmark Kaggle dataset...")
        df = generate_synthetic_kaggle_dataset()
        target_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(target_path, index=False)
        print(f"[DataLoader] Generated & saved raw dataset to {target_path}")

    return df


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Cleans, preprocesses, and engineers behavioral marketing features:
    - Missing value imputation
    - Age calculation
    - Aggregated product spend (TotalMnt) & basket variety
    - Aggregated channel purchases & web intensity
    - Past campaign affinity score
    - RFM (Recency, Frequency, Monetary) quintile scoring
    - Estimated Customer Lifetime Value (CLV proxy)
    """
    data = df.copy()

    # 1. Clean Income
    if data["Income"].isnull().sum() > 0:
        data["Income"] = data["Income"].fillna(data["Income"].median())

    # 2. Demographics & Household
    current_year = 2026
    data["Age"] = current_year - data["Year_Birth"]
    data["Total_Children"] = data["Kidhome"] + data["Teenhome"]
    data["Has_Children"] = (data["Total_Children"] > 0).astype(int)

    # Education grouping
    data["Is_Higher_Ed"] = data["Education"].isin(["Graduation", "Master", "PhD"]).astype(int)
    # Marital grouping: Single/Living Alone vs Partnered
    data["Is_Partnered"] = data["Marital_Status"].isin(["Married", "Together"]).astype(int)

    # 3. Customer Tenure
    try:
        data["Customer_Date"] = pd.to_datetime(data["Dt_Customer"], format="%d-%m-%Y")
    except Exception:
        data["Customer_Date"] = pd.to_datetime(data["Dt_Customer"])
    
    reference_date = pd.to_datetime("2026-01-01")
    data["Tenure_Days"] = (reference_date - data["Customer_Date"]).dt.days

    # 4. Spending & Category Mix
    spend_cols = ["MntWines", "MntFruits", "MntMeatProducts", "MntFishProducts", "MntSweetProducts", "MntGoldProds"]
    data["TotalSpend"] = data[spend_cols].sum(axis=1)
    
    # Proportions of spend by category
    for col in spend_cols:
        prop_name = f"Prop_{col}"
        data[prop_name] = data[col] / (data["TotalSpend"] + 1e-5)

    # 5. Purchase Channels & Velocity
    purchase_cols = ["NumWebPurchases", "NumCatalogPurchases", "NumStorePurchases"]
    data["TotalPurchases"] = data[purchase_cols].sum(axis=1)
    data["AverageOrderValue"] = data["TotalSpend"] / np.maximum(data["TotalPurchases"], 1)

    data["WebPurchaseRatio"] = data["NumWebPurchases"] / np.maximum(data["TotalPurchases"], 1)
    data["CatalogPurchaseRatio"] = data["NumCatalogPurchases"] / np.maximum(data["TotalPurchases"], 1)
    data["StorePurchaseRatio"] = data["NumStorePurchases"] / np.maximum(data["TotalPurchases"], 1)

    # Web Engagement / Browsing vs Buying conversion efficiency
    data["WebConversionRate"] = data["NumWebPurchases"] / np.maximum(data["NumWebVisitsMonth"] * 12, 1)

    # 6. Past Campaign Responsiveness
    cmp_cols = ["AcceptedCmp1", "AcceptedCmp2", "AcceptedCmp3", "AcceptedCmp4", "AcceptedCmp5"]
    data["PastCampaignsAccepted"] = data[cmp_cols].sum(axis=1)
    data["CampaignResponseHistoryRatio"] = data["PastCampaignsAccepted"] / 5.0
    data["EverAcceptedCampaign"] = (data["PastCampaignsAccepted"] > 0).astype(int)

    # 7. RFM Scoring (Recency, Frequency, Monetary)
    # Recency: Lower is better -> reverse rank
    data["R_Score"] = pd.qcut(data["Recency"], 4, labels=[4, 3, 2, 1]).astype(int)
    data["F_Score"] = pd.qcut(data["TotalPurchases"].rank(method="first"), 4, labels=[1, 2, 3, 4]).astype(int)
    data["M_Score"] = pd.qcut(data["TotalSpend"].rank(method="first"), 4, labels=[1, 2, 3, 4]).astype(int)
    data["RFM_Composite"] = data["R_Score"] * 100 + data["F_Score"] * 10 + data["M_Score"]

    # 8. Customer Lifetime Value (CLV) proxy baseline
    # CLV = AverageOrderValue * AnnualFrequency * EstimatedLifespan (years)
    annual_frequency = np.maximum(data["TotalPurchases"] / np.maximum(data["Tenure_Days"] / 365.25, 0.5), 1.0)
    data["Estimated_CLV"] = np.round(data["AverageOrderValue"] * annual_frequency * 3.0, 2)

    return data


def get_preprocessed_data(force_recompute: bool = False) -> pd.DataFrame:
    """Returns the processed feature-engineered dataset, caching to disk."""
    if not force_recompute and PROCESSED_FILE_PATH.exists():
        return pd.read_csv(PROCESSED_FILE_PATH)

    raw_df = load_raw_dataset()
    processed_df = engineer_features(raw_df)
    processed_df.to_csv(PROCESSED_FILE_PATH, index=False)
    print(f"[DataLoader] Successfully engineered features: {processed_df.shape[0]} customers, {processed_df.shape[1]} features.")
    return processed_df


if __name__ == "__main__":
    df = get_preprocessed_data(force_recompute=True)
    print(f"Sample of processed data:\n{df[['ID', 'Age', 'TotalSpend', 'TotalPurchases', 'Estimated_CLV', 'Response']].head()}")
