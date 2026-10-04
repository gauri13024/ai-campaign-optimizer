"""Mathematical Formulation of Multi-Objective Campaign Strategy Optimization.

Formalization:
Given N candidate customer touchpoints and C channels:
Decision variable:
    x_{i,c} in {0, 1} : Whether customer i is contacted via channel c

Objectives:
    Maximize f_1(X) = sum_{i,c} x_{i,c} * P(Conversion_{i,c}) * Revenue_{i}  [Expected Revenue]
    Maximize f_2(X) = sum_{i,c} x_{i,c} * P(Conversion_{i,c})                [Total Conversions]
    Minimize f_3(X) = sum_{i,c} x_{i,c} * UnitCost_{i,c}                    [Total Campaign Budget Spent]
    Minimize f_4(X) = sum_{i,c} x_{i,c} * FatigueHazard_{i,c}               [Ad Fatigue / Churn Risk]

Constraints:
    1. Budget Constraint: sum_{i,c} x_{i,c} * UnitCost_{i,c} <= Total_Budget
    2. Contact Frequency Cap: sum_c x_{i,c} <= 1  (at most one channel per customer per cycle)
    3. Minimum Channel Allocation: sum_i x_{i,c} * UnitCost_{i,c} >= Min_Channel_Budget
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple
import numpy as np
import pandas as pd


@dataclass
class CampaignConstraints:
    total_budget: float = 2500.0
    max_contacts_per_customer: int = 1
    min_roi_target: float = 1.20   # 120% target return on ad spend
    max_acceptable_fatigue_rate: float = 0.08  # at most 8% high-fatigue rate


def evaluate_solution_objectives(
    selection_vector: np.ndarray,
    candidate_df: pd.DataFrame
) -> Tuple[float, float, float, float]:
    """Computes the 4 business objectives for a given allocation vector x in {0, 1}^N:
    Returns:
        (total_revenue, total_conversions, total_cost, total_fatigue_risk)
    """
    selected = candidate_df.iloc[selection_vector.astype(bool)]
    
    if len(selected) == 0:
        return 0.0, 0.0, 0.0, 0.0

    # f1: Expected Revenue
    exp_rev = float(np.sum(selected["Effective_CVR"] * (selected["Customer_TotalSpend"] * 0.15 + 40.0)))
    # f2: Expected Conversions
    exp_conv = float(np.sum(selected["Effective_CVR"]))
    # f3: Total Cost
    cost = float(np.sum(selected["Channel_Cost"]))
    # f4: Total Fatigue Risk
    fatigue = float(np.sum(selected["Complaint_Churn_Risk"]))

    return round(exp_rev, 2), round(exp_conv, 1), round(cost, 2), round(fatigue, 2)


def check_solution_feasibility(
    selection_vector: np.ndarray,
    candidate_df: pd.DataFrame,
    constraints: CampaignConstraints
) -> bool:
    """Checks whether the candidate allocation satisfies budget and frequency caps."""
    selected = candidate_df.iloc[selection_vector.astype(bool)]
    cost = selected["Channel_Cost"].sum()
    if cost > constraints.total_budget:
        return False

    # Check max 1 contact per customer
    cust_counts = selected["Customer_ID"].value_counts()
    if (cust_counts > constraints.max_contacts_per_customer).any():
        return False

    return True


def compute_hypervolume_2d(pareto_front_points: np.ndarray, reference_point: Tuple[float, float]) -> float:
    """Computes 2D hypervolume indicator for a set of Pareto objective vectors
    (e.g., Revenue vs. Cost) relative to a reference point.
    """
    if len(pareto_front_points) == 0:
        return 0.0

    # Sort by first objective
    sorted_pts = pareto_front_points[pareto_front_points[:, 0].argsort()]
    ref_x, ref_y = reference_point
    hv = 0.0
    current_y = ref_y

    for x, y in sorted_pts[::-1]:
        if x > ref_x and y > current_y:
            hv += (x - ref_x) * (y - current_y)
            current_y = y

    return float(hv)
