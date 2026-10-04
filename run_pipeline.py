"""Master Execution Pipeline for AI-Driven Intelligent Campaign Optimization.

Orchestrates all components:
1. Ingestion & Preprocessing (Kaggle Dataset + RFM + Multi-Channel Generation)
2. AI Predictive Training (LightGBM, XGBoost, CatBoost, Neural Networks, CLV Regressors)
3. Multi-Objective Optimization Benchmarking (NSGA-II, Linear Programming, Greedy Baseline, Bandits)
4. Marketing Metrics & Multi-Touch Attribution Computation
5. Summary Generation & Streamlit Launch Instructions
"""

from __future__ import annotations

import sys
import time
from pathlib import Path
import pandas as pd

ROOT_DIR = Path(__file__).resolve().parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from data.data_loader import get_preprocessed_data
from data.channel_touchpoint_generator import get_channel_touchpoint_data
from data.dataset_survey import get_dataset_survey_dataframe
from models.conversion_model import train_and_save_best_model
from models.clv_revenue_model import train_and_save_clv_model
from models.fatigue_churn_model import train_fatigue_risk_model
from models.model_explainability import get_global_feature_importance
from optimization.optimization_benchmark import run_full_optimization_benchmark
from analytics.marketing_metrics import compute_channel_performance_breakdown
from analytics.attribution_models import compute_multitouch_attribution


def run_full_pipeline(budget: float = 2500.0) -> None:
    print("=" * 80)
    print("🚀 AI-DRIVEN INTELLIGENT CAMPAIGN OPTIMIZATION PLATFORM")
    print("=" * 80)

    start_time = time.time()

    # Step 1: Data Engineering
    print("\n[STEP 1/6] Ingesting Data & Engineering Behavioral Features...")
    cust_df = get_preprocessed_data(force_recompute=True)
    print(f" -> Processed {len(cust_df)} customer profiles with RFM & demographic features.")

    print("\n[STEP 2/6] Generating Multi-Channel Touchpoints & Ad Fatigue Curves...")
    tp_df = get_channel_touchpoint_data(cust_df, force_recompute=True)
    print(f" -> Generated {len(tp_df)} candidate touchpoints across 5 channels.")

    # Step 2: AI Predictive Modeling
    print("\n[STEP 3/6] Training AI Predictive Models (Conversion, CLV, Ad Fatigue)...")
    conv_leaderboard, best_conv = train_and_save_best_model(cust_df)
    print(" -> Conversion Classification Leaderboard:")
    print(conv_leaderboard.to_string(index=False))

    clv_leaderboard, best_clv = train_and_save_clv_model(cust_df)
    print(" -> CLV Monetary Regression Leaderboard:")
    print(clv_leaderboard.to_string(index=False))

    train_fatigue_risk_model(tp_df)
    print(" -> Ad Fatigue & Churn Risk Model trained.")

    df_imp = get_global_feature_importance()
    print("\n -> Top 5 Predictive Drivers (Global Importance):")
    for _, row in df_imp.head(5).iterrows():
        print(f"    * {row['Feature']}: {row['Importance_Pct']:.2f}%")

    # Step 3: Multi-Objective Optimization
    print(f"\n[STEP 4/6] Executing Multi-Objective Optimization Benchmark (Budget: ${budget:,.2f})...")
    comp_df, extra_opt = run_full_optimization_benchmark(tp_df, total_budget=budget)
    print("\n" + "=" * 80)
    print("OPTIMIZATION PARADIGM COMPARISON TABLE:")
    print("=" * 80)
    print(comp_df.to_string(index=False))

    pareto_df = extra_opt["pareto_front"]
    print(f"\n -> Discovered {len(pareto_df)} non-dominated Pareto optimal solutions.")
    if "Strategy_Archetype" in pareto_df.columns:
        print(" -> Pareto Strategy Clusters:")
        for arch, grp in pareto_df.groupby("Strategy_Archetype"):
            print(f"    * {arch}: Avg ROI = {grp['ROI_Pct'].mean():.1f}%, Avg Spend = ${grp['Campaign_Cost'].mean():.2f}")

    # Step 4: Marketing Analytics & Attribution
    print("\n[STEP 5/6] Computing Marketing Economics & Multi-Touch Attribution...")
    channel_perf = compute_channel_performance_breakdown(tp_df)
    print(" -> Channel Performance Summary:")
    print(channel_perf[["Channel", "Spend ($)", "Revenue ($)", "ROAS", "ROI (%)", "CPA ($)"]].to_string(index=False))

    attr_df = compute_multitouch_attribution(tp_df)
    print(" -> Attribution Share by Model:")
    print(attr_df.to_string(index=False))

    # Step 5: Dataset Survey
    print("\n[STEP 6/6] Generating Benchmark Dataset Survey Report...")
    survey_df = get_dataset_survey_dataframe()
    print(f" -> Surveyed {len(survey_df)} benchmark datasets (Criteo, Avazu, Bank Marketing, Kaggle).")

    elapsed = time.time() - start_time
    print("\n" + "=" * 80)
    print(f"✅ PIPELINE COMPLETED SUCCESSFULLY in {elapsed:.2f} seconds.")
    print("=" * 80)
    print("\nTo launch the Interactive Decision-Support Dashboard, execute:")
    print("    streamlit run dashboard/app.py\n")


if __name__ == "__main__":
    budget_arg = 2500.0
    if len(sys.argv) > 1:
        try:
            budget_arg = float(sys.argv[1])
        except ValueError:
            pass
    run_full_pipeline(budget=budget_arg)
