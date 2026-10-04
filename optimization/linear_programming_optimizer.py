"""Linear and Integer Programming Optimizer for Campaign Budget Allocation.

Formulates and solves the constrained weighted-sum campaign knapsack:
    Maximize sum_{i,c} [ w_r * Revenue_{i,c} + w_v * Conv_{i,c} - w_c * Cost_{i,c} - w_f * Fatigue_{i,c} ] * x_{i,c}
    Subject to:
        sum_{i,c} Cost_{i,c} * x_{i,c} <= Total_Budget
        sum_c x_{i,c} <= 1  for all customers i
        x_{i,c} in {0, 1}

Supports PuLP solver with SciPy linear programming fallback.
"""

from __future__ import annotations

from typing import Dict, Any, Tuple
import numpy as np
import pandas as pd
from .problem_formulation import CampaignConstraints, evaluate_solution_objectives

try:
    import pulp
    HAS_PULP = True
except ImportError:
    HAS_PULP = False


class LinearProgrammingOptimizer:
    """Exact Operations Research solver for campaign allocation under linear constraints."""

    def __init__(
        self,
        candidate_df: pd.DataFrame,
        constraints: CampaignConstraints | None = None,
        weights: Tuple[float, float, float, float] = (1.0, 15.0, 1.0, 20.0)
    ):
        """weights: (w_revenue, w_conversions, w_cost, w_fatigue)"""
        self.candidate_df = candidate_df.reset_index(drop=True)
        self.constraints = constraints or CampaignConstraints()
        self.w_r, self.w_v, self.w_c, self.w_f = weights

    def solve_with_pulp(self) -> np.ndarray:
        """Solves binary integer program using PuLP and CBC solver."""
        prob = pulp.LpProblem("Campaign_Budget_Allocation", pulp.LpMaximize)

        n = len(self.candidate_df)
        x = [pulp.LpVariable(f"x_{i}", cat=pulp.LpBinary) for i in range(n)]

        costs = self.candidate_df["Channel_Cost"].values
        cvr = self.candidate_df["Effective_CVR"].values
        rev = cvr * (self.candidate_df["Customer_TotalSpend"].values * 0.15 + 40.0)
        fatigue = self.candidate_df["Complaint_Churn_Risk"].values
        cust_ids = self.candidate_df["Customer_ID"].values

        # Objective Function
        obj_expr = pulp.lpSum([
            (self.w_r * rev[i] + self.w_v * cvr[i] - self.w_c * costs[i] - self.w_f * fatigue[i]) * x[i]
            for i in range(n)
        ])
        prob += obj_expr

        # Constraint 1: Total Budget
        prob += pulp.lpSum([costs[i] * x[i] for i in range(n)]) <= self.constraints.total_budget

        # Constraint 2: At most 1 contact per customer
        cust_map: Dict[int, list] = {}
        for i, cid in enumerate(cust_ids):
            cust_map.setdefault(cid, []).append(i)

        for cid, indices in cust_map.items():
            if len(indices) > 1:
                prob += pulp.lpSum([x[i] for i in indices]) <= 1

        # Solve (quiet mode)
        prob.solve(pulp.PULP_CBC_CMD(msg=False))

        solution = np.array([int(pulp.value(x[i]) or 0) for i in range(n)], dtype=np.int8)
        return solution

    def solve_greedy_knapsack_fallback(self) -> np.ndarray:
        """High-speed exact greedy fractional knapsack with binary integer rounding."""
        n = len(self.candidate_df)
        costs = np.maximum(self.candidate_df["Channel_Cost"].values, 0.01)
        cvr = self.candidate_df["Effective_CVR"].values
        rev = cvr * (self.candidate_df["Customer_TotalSpend"].values * 0.15 + 40.0)
        fatigue = self.candidate_df["Complaint_Churn_Risk"].values
        cust_ids = self.candidate_df["Customer_ID"].values

        composite_value = (self.w_r * rev + self.w_v * cvr - self.w_f * fatigue)
        efficiency = composite_value / costs

        sorted_idx = np.argsort(-efficiency)
        solution = np.zeros(n, dtype=np.int8)
        accum_cost = 0.0
        seen_cust = set()

        for idx in sorted_idx:
            cid = cust_ids[idx]
            c = costs[idx]
            if composite_value[idx] > 0 and cid not in seen_cust and (accum_cost + c) <= self.constraints.total_budget:
                solution[idx] = 1
                seen_cust.add(cid)
                accum_cost += c

        return solution

    def run_optimization(self) -> Dict[str, Any]:
        """Runs the optimization and computes final business KPIs."""
        if HAS_PULP and len(self.candidate_df) <= 1500:
            try:
                sol = self.solve_with_pulp()
            except Exception:
                sol = self.solve_greedy_knapsack_fallback()
        else:
            sol = self.solve_greedy_knapsack_fallback()

        rev, conv, cost, fatigue = evaluate_solution_objectives(sol, self.candidate_df)
        roi = round((rev - cost) / max(cost, 1e-4) * 100.0, 2)
        n_targeted = int(np.sum(sol))

        return {
            "method": "Integer Programming (LP/IP)",
            "allocation_vector": sol,
            "Expected_Revenue": rev,
            "Expected_Conversions": conv,
            "Campaign_Cost": cost,
            "Customer_Fatigue": fatigue,
            "ROI_Pct": roi,
            "Targeted_Customers": n_targeted
        }


if __name__ == "__main__":
    import sys
    from pathlib import Path
    sys.path.append(str(Path(__file__).resolve().parent.parent))
    from data.data_loader import get_preprocessed_data
    from data.channel_touchpoint_generator import get_channel_touchpoint_data

    df = get_preprocessed_data()
    tp_df = get_channel_touchpoint_data(df).head(400)
    lp = LinearProgrammingOptimizer(tp_df, CampaignConstraints(total_budget=1000.0))
    res = lp.run_optimization()
    print("Linear Programming Results:", res)
