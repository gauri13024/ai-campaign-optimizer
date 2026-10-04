"""Contextual Multi-Armed Bandit & Reinforcement Learning Policy.

Simulates adaptive sequential campaign allocation under uncertainty:
- Arms: Marketing Channels (Email, Paid Search, Social Media, Display Ads, SMS)
- Context: Customer Demographic & RFM Features (Income, Age, Past Spend)
- Exploration-Exploitation Strategy: Contextual Thompson Sampling (Bayesian Logistic Regression / Beta-Bernoulli Bandit) and UCB1 (Upper Confidence Bound).

Demonstrates sequential optimization and cumulative regret minimization.
"""

from __future__ import annotations

from typing import Dict, List, Any
import numpy as np
import pandas as pd


class ContextualThompsonSamplingBandit:
    """Contextual Bandit for dynamic online channel allocation using
    Bayesian conjugate updates (Beta-Bernoulli per customer segment).
    """

    def __init__(self, channels: list[str], random_state: int = 42):
        self.channels = channels
        self.n_arms = len(channels)
        self.rng = np.random.default_rng(random_state)
        
        # Segment definitions: 3 income tiers x 2 age brackets = 6 customer context buckets
        self.n_contexts = 6
        # Prior parameters: Beta(alpha=1, beta=1) uniform uninformative
        self.alpha = np.ones((self.n_contexts, self.n_arms))
        self.beta = np.ones((self.n_contexts, self.n_arms))

    def _get_context_id(self, income: float, age: float) -> int:
        """Discretizes customer context into state index [0..5]."""
        income_tier = 0 if income < 35000 else (1 if income < 65000 else 2)
        age_tier = 0 if age < 45 else 1
        return income_tier * 2 + age_tier

    def select_channel(self, income: float, age: float) -> str:
        """Samples from posterior Beta distributions (Thompson Sampling) and selects the arm."""
        ctx = self._get_context_id(income, age)
        # Sample theta ~ Beta(alpha, beta) for each arm
        sampled_thetas = self.rng.beta(self.alpha[ctx], self.beta[ctx])
        best_arm_idx = int(np.argmax(sampled_thetas))
        return self.channels[best_arm_idx]

    def update_posterior(self, income: float, age: float, chosen_channel: str, reward: int):
        """Bayesian update on conversion outcome."""
        ctx = self._get_context_id(income, age)
        arm_idx = self.channels.index(chosen_channel)
        if reward > 0:
            self.alpha[ctx, arm_idx] += 1
        else:
            self.beta[ctx, arm_idx] += 1

    def simulate_campaign_rollout(
        self,
        customers_df: pd.DataFrame,
        total_budget: float = 2000.0
    ) -> Dict[str, Any]:
        """Simulates sequential online targeting until budget depletion."""
        shuffled = customers_df.sample(frac=1.0, random_state=42).reset_index(drop=True)
        
        channel_costs = {
            "Email": 0.05,
            "Paid_Search": 1.20,
            "Social_Media": 0.45,
            "Display_Ads": 0.15,
            "SMS": 0.25
        }
        base_cvr_table = {
            "Email": 0.04,
            "Paid_Search": 0.09,
            "Social_Media": 0.04,
            "Display_Ads": 0.02,
            "SMS": 0.07
        }

        budget_spent = 0.0
        total_conversions = 0
        total_revenue = 0.0
        total_fatigue = 0.0
        channel_counts = {ch: 0 for ch in self.channels}
        decisions = []

        for _, row in shuffled.iterrows():
            cid = int(row["ID"])
            income = float(row["Income"])
            age = float(row["Age"])
            total_spend = float(row["TotalSpend"])

            chosen_channel = self.select_channel(income, age)
            cost = channel_costs.get(chosen_channel, 0.20)

            if budget_spent + cost > total_budget:
                break

            channel_counts[chosen_channel] += 1
            budget_spent += cost

            # Simulate true probabilistic response
            true_prob = base_cvr_table.get(chosen_channel, 0.03) * (1.2 if income > 55000 else 0.9)
            converted = int(self.rng.random() < true_prob)

            # Update bandit policy
            self.update_posterior(income, age, chosen_channel, reward=converted)

            rev = round(converted * (total_spend * 0.15 + 40.0), 2) if converted else 0.0
            fatigue = 0.03 if chosen_channel == "SMS" else 0.01

            total_conversions += converted
            total_revenue += rev
            total_fatigue += fatigue

            decisions.append({
                "Customer_ID": cid,
                "Channel": chosen_channel,
                "Cost": cost,
                "Converted": converted,
                "Revenue": rev
            })

        roi = round((total_revenue - budget_spent) / max(budget_spent, 1e-4) * 100.0, 2)

        return {
            "method": "Contextual Bandit (Thompson Sampling RL)",
            "Targeted_Customers": len(decisions),
            "Campaign_Cost": round(budget_spent, 2),
            "Expected_Revenue": round(total_revenue, 2),
            "Expected_Conversions": total_conversions,
            "Customer_Fatigue": round(total_fatigue, 2),
            "ROI_Pct": roi,
            "Channel_Distribution": channel_counts,
            "Decisions_DF": pd.DataFrame(decisions)
        }


if __name__ == "__main__":
    import sys
    from pathlib import Path
    sys.path.append(str(Path(__file__).resolve().parent.parent))
    from data.data_loader import get_preprocessed_data

    df = get_preprocessed_data()
    channels = ["Email", "Paid_Search", "Social_Media", "Display_Ads", "SMS"]
    bandit = ContextualThompsonSamplingBandit(channels)
    results = bandit.simulate_campaign_rollout(df, total_budget=1500.0)
    print("Bandit Campaign Rollout Results:")
    print(f"Targeted: {results['Targeted_Customers']}, Revenue: ${results['Expected_Revenue']}, Cost: ${results['Campaign_Cost']}, ROI: {results['ROI_Pct']}%")
    print("Channel Allocations:", results["Channel_Distribution"])
