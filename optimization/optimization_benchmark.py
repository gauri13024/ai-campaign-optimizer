"""Comparative Optimization Benchmarking Suite.

Conducts a rigorous experimental comparison across 4 distinct optimization paradigms:
1. Heuristic Greedy Policy (Single-Objective CTR / Conversion greedy)
2. Constrained Integer Programming (PuLP / SciPy Operations Research)
3. Evolutionary Multi-Objective Optimization (NSGA-II Pareto Frontier)
4. Contextual Multi-Armed Bandit (Thompson Sampling RL Policy)

Evaluates:
- Return on Investment (ROI %)
- Total Revenue Generated ($)
- Total Marketing Spend ($)
- Conversion Volume
- Cost Per Acquisition (CPA)
- Customer Ad Fatigue / Churn Hazard
- Strategic Strengths & Limitations
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Any, Tuple
import numpy as np
import pandas as pd

from .problem_formulation import CampaignConstraints, evaluate_solution_objectives
from .nsga2_optimizer import NSGA2CampaignOptimizer
from .linear_programming_optimizer import LinearProgrammingOptimizer
from .contextual_bandit_optimizer import ContextualThompsonSamplingBandit


def run_greedy_ctr_baseline(
    candidate_df: pd.DataFrame,
    constraints: CampaignConstraints
) -> Dict[str, Any]:
    """Simulates standard single-objective industry practice:
    Sort candidates strictly by predicted conversion rate (ignoring cost, CLV, and fatigue)
    and allocate until budget depletion.
    """
    df_sorted = candidate_df.sort_values(by="Effective_CVR", ascending=False).reset_index(drop=True)
    n = len(df_sorted)
    sol = np.zeros(n, dtype=np.int8)
    accum_cost = 0.0
    seen_cust = set()

    for i, row in df_sorted.iterrows():
        cid = int(row["Customer_ID"])
        c = float(row["Channel_Cost"])
        if cid not in seen_cust and (accum_cost + c) <= constraints.total_budget:
            sol[i] = 1
            seen_cust.add(cid)
            accum_cost += c

    rev, conv, cost, fatigue = evaluate_solution_objectives(sol, df_sorted)
    roi = round((rev - cost) / max(cost, 1e-4) * 100.0, 2)

    return {
        "Optimization Strategy": "Greedy Single-Objective (CTR/CVR Only)",
        "Paradigm": "Heuristic Ranking",
        "Campaign_Cost": cost,
        "Expected_Revenue": rev,
        "Expected_Conversions": conv,
        "ROI_Pct": roi,
        "CPA": round(cost / max(conv, 1e-4), 2),
        "Customer_Fatigue": fatigue,
        "Targeted_Customers": int(np.sum(sol)),
        "Trade-Off Insight": "Suffers from budget exhaustion on high-cost channels; high ad fatigue penalty."
    }


def run_full_optimization_benchmark(
    candidate_df: pd.DataFrame,
    total_budget: float = 2000.0
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Runs all 4 optimization engines and returns a comparative evaluation table."""
    constraints = CampaignConstraints(total_budget=total_budget)

    # 1. Greedy Single-Objective Baseline
    res_greedy = run_greedy_ctr_baseline(candidate_df, constraints)

    # 2. Linear / Integer Programming
    lp_solver = LinearProgrammingOptimizer(candidate_df, constraints)
    res_lp_raw = lp_solver.run_optimization()
    res_lp = {
        "Optimization Strategy": "Integer Programming (PuLP / Knapsack)",
        "Paradigm": "Operations Research / Exact",
        "Campaign_Cost": res_lp_raw["Campaign_Cost"],
        "Expected_Revenue": res_lp_raw["Expected_Revenue"],
        "Expected_Conversions": res_lp_raw["Expected_Conversions"],
        "ROI_Pct": res_lp_raw["ROI_Pct"],
        "CPA": round(res_lp_raw["Campaign_Cost"] / max(res_lp_raw["Expected_Conversions"], 1e-4), 2),
        "Customer_Fatigue": res_lp_raw["Customer_Fatigue"],
        "Targeted_Customers": res_lp_raw["Targeted_Customers"],
        "Trade-Off Insight": "Fast exact solution for scalarized weighted sum, but requires predetermined objective weights."
    }

    # 3. Evolutionary Multi-Objective (NSGA-II)
    # Using candidate sample if dataset is huge for instant benchmark responsiveness
    sample_df = candidate_df.head(1000) if len(candidate_df) > 1000 else candidate_df
    nsga_solver = NSGA2CampaignOptimizer(
        candidate_df=sample_df,
        constraints=constraints,
        population_size=40,
        n_generations=20
    )
    res_nsga_raw = nsga_solver.run_optimization()
    pareto_df = res_nsga_raw["pareto_dataframe"]

    # Pick the Balanced Pareto solution
    if "Balanced Growth" in pareto_df["Strategy_Archetype"].values:
        balanced_row = pareto_df[pareto_df["Strategy_Archetype"] == "Balanced Growth"].iloc[0]
    else:
        balanced_row = pareto_df.iloc[len(pareto_df) // 2]

    res_nsga = {
        "Optimization Strategy": "Multi-Objective Genetic Algorithm (NSGA-II)",
        "Paradigm": "Evolutionary Metaheuristic",
        "Campaign_Cost": balanced_row["Campaign_Cost"],
        "Expected_Revenue": balanced_row["Expected_Revenue"],
        "Expected_Conversions": balanced_row["Expected_Conversions"],
        "ROI_Pct": balanced_row["ROI_Pct"],
        "CPA": round(balanced_row["Campaign_Cost"] / max(balanced_row["Expected_Conversions"], 1e-4), 2),
        "Customer_Fatigue": balanced_row["Customer_Fatigue"],
        "Targeted_Customers": balanced_row["Targeted_Customers"],
        "Trade-Off Insight": "Reveals the true Pareto frontier without weight tuning; best multi-objective compromise."
    }

    # 4. Contextual Multi-Armed Bandit
    unique_cust = candidate_df[["Customer_ID", "Customer_Income", "Customer_Age", "Customer_TotalSpend"]].drop_duplicates()
    channels = ["Email", "Paid_Search", "Social_Media", "Display_Ads", "SMS"]
    bandit = ContextualThompsonSamplingBandit(channels)
    res_bandit_raw = bandit.simulate_campaign_rollout(
        unique_cust.rename(columns={"Customer_ID": "ID", "Customer_Income": "Income", "Customer_Age": "Age", "Customer_TotalSpend": "TotalSpend"}),
        total_budget=total_budget
    )
    res_bandit = {
        "Optimization Strategy": "Contextual Bandit (Thompson Sampling RL)",
        "Paradigm": "Reinforcement Learning / Online",
        "Campaign_Cost": res_bandit_raw["Campaign_Cost"],
        "Expected_Revenue": res_bandit_raw["Expected_Revenue"],
        "Expected_Conversions": res_bandit_raw["Expected_Conversions"],
        "ROI_Pct": res_bandit_raw["ROI_Pct"],
        "CPA": round(res_bandit_raw["Campaign_Cost"] / max(res_bandit_raw["Expected_Conversions"], 1e-4), 2),
        "Customer_Fatigue": res_bandit_raw["Customer_Fatigue"],
        "Targeted_Customers": res_bandit_raw["Targeted_Customers"],
        "Trade-Off Insight": "Continuously balances exploration vs exploitation; robust to non-stationary environments."
    }

    comparison_df = pd.DataFrame([res_greedy, res_lp, res_nsga, res_bandit])
    return comparison_df, {
        "pareto_front": pareto_df,
        "bandit_details": res_bandit_raw
    }


if __name__ == "__main__":
    import sys
    sys.path.append(str(Path(__file__).resolve().parent.parent))
    from data.data_loader import get_preprocessed_data
    from data.channel_touchpoint_generator import get_channel_touchpoint_data

    df = get_preprocessed_data()
    tp_df = get_channel_touchpoint_data(df)
    comp_df, extra = run_full_optimization_benchmark(tp_df, total_budget=2000.0)
    print("\nOptimization Benchmark Results:")
    print(comp_df.to_string())
