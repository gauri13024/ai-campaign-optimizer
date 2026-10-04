"""Evolutionary Multi-Objective Optimization using NSGA-II for Campaign Strategy.

Implements Deb et al.'s Non-dominated Sorting Genetic Algorithm II (NSGA-II)
with custom genetic operators, feasibility repair heuristics, and Pareto strategy clustering.

Simultaneously optimizes:
- Maximize: Expected Revenue (CLV)
- Maximize: Expected Conversions
- Minimize: Total Marketing Budget Cost
- Minimize: Customer Fatigue & Churn Penalty

Identifies distinct strategy archetypes on the Pareto Frontier:
1. High-ROI Strategy (Cost-Efficient)
2. Balanced Growth Strategy
3. High-Reach / Scale Strategy
Grounded in MDPI Mathematics (Abildaeva et al., 2026).
"""

from __future__ import annotations

from typing import List, Tuple, Dict, Any
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from .problem_formulation import CampaignConstraints, evaluate_solution_objectives


class NSGA2CampaignOptimizer:
    """Vectorized, self-contained NSGA-II implementation tailored for
    large-scale combinatorial marketing intervention selection.
    """

    def __init__(
        self,
        candidate_df: pd.DataFrame,
        constraints: CampaignConstraints | None = None,
        population_size: int = 50,
        n_generations: int = 35,
        mutation_rate: float = 0.05,
        crossover_rate: float = 0.85,
        random_state: int = 42
    ):
        self.candidate_df = candidate_df.reset_index(drop=True)
        self.n_candidates = len(self.candidate_df)
        self.constraints = constraints or CampaignConstraints()
        self.pop_size = population_size
        self.n_gen = n_generations
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.rng = np.random.default_rng(random_state)
        
        # Precompute efficiency metrics for intelligent initialization & repair
        cost = np.maximum(self.candidate_df["Channel_Cost"].values, 0.01)
        rev = self.candidate_df["Effective_CVR"].values * (self.candidate_df["Customer_TotalSpend"].values * 0.15 + 40.0)
        self.unit_roi = rev / cost
        self.costs = cost
        self.customer_ids = self.candidate_df["Customer_ID"].values

    def _repair_solution(self, solution: np.ndarray) -> np.ndarray:
        """Enforces constraints:
        1. At most 1 channel per unique customer
        2. Total budget <= budget constraint
        Uses greedy priority ranking by unit ROI.
        """
        sol = solution.copy()
        selected_idx = np.where(sol > 0)[0]
        if len(selected_idx) == 0:
            return sol

        # 1. Enforce unique customer contact
        seen_customers = set()
        # Sort candidates in descending unit ROI
        order = selected_idx[np.argsort(-self.unit_roi[selected_idx])]
        for idx in order:
            cid = self.customer_ids[idx]
            if cid in seen_customers:
                sol[idx] = 0
            else:
                seen_customers.add(cid)

        # 2. Enforce total budget
        selected_idx = np.where(sol > 0)[0]
        total_cost = np.sum(self.costs[selected_idx])
        if total_cost > self.constraints.total_budget:
            # Drop lowest ROI elements until under budget
            sorted_asc_roi = selected_idx[np.argsort(self.unit_roi[selected_idx])]
            for idx in sorted_asc_roi:
                sol[idx] = 0
                total_cost -= self.costs[idx]
                if total_cost <= self.constraints.total_budget:
                    break

        return sol

    def _initialize_population(self) -> np.ndarray:
        """Creates an initial diverse population combining greedy ROI seeds,
        balanced seeds, and random perturbations.
        """
        population = np.zeros((self.pop_size, self.n_candidates), dtype=np.int8)

        # Individual 0: Pure Greedy by Unit ROI
        sorted_by_roi = np.argsort(-self.unit_roi)
        ind_greedy = np.zeros(self.n_candidates, dtype=np.int8)
        accum_cost = 0.0
        seen_c = set()
        for idx in sorted_by_roi:
            cid = self.customer_ids[idx]
            c = self.costs[idx]
            if cid not in seen_c and (accum_cost + c) <= self.constraints.total_budget:
                ind_greedy[idx] = 1
                seen_c.add(cid)
                accum_cost += c
        population[0] = ind_greedy

        # Individual 1: Pure Reach / Conversion Focus
        sorted_by_cvr = np.argsort(-self.candidate_df["Effective_CVR"].values)
        ind_reach = np.zeros(self.n_candidates, dtype=np.int8)
        accum_cost = 0.0
        seen_c = set()
        for idx in sorted_by_cvr:
            cid = self.customer_ids[idx]
            c = self.costs[idx]
            if cid not in seen_c and (accum_cost + c) <= self.constraints.total_budget:
                ind_reach[idx] = 1
                seen_c.add(cid)
                accum_cost += c
        population[1] = ind_reach

        # Remaining individuals: Stochastic biased selections
        for p in range(2, self.pop_size):
            # Sample random budget factor between 30% and 100%
            target_budget = self.constraints.total_budget * self.rng.uniform(0.3, 1.0)
            perm = self.rng.permutation(self.n_candidates)
            ind = np.zeros(self.n_candidates, dtype=np.int8)
            curr_c = 0.0
            seen_c = set()
            for idx in perm:
                cid = self.customer_ids[idx]
                c = self.costs[idx]
                if cid not in seen_c and (curr_c + c) <= target_budget:
                    # Probabilistic acceptance biased by ROI
                    if self.rng.random() < 0.6:
                        ind[idx] = 1
                        seen_c.add(cid)
                        curr_c += c
            population[p] = self._repair_solution(ind)

        return population

    def _evaluate_population(self, population: np.ndarray) -> np.ndarray:
        """Returns objective matrix of shape (N, 4):
        Objectives to maximize: [Revenue, Conversions]
        Objectives to minimize: [Cost, Fatigue] -> negated for maximization: [-Cost, -Fatigue]
        """
        N = len(population)
        objs = np.zeros((N, 4))
        for i in range(N):
            rev, conv, cost, fatigue = evaluate_solution_objectives(population[i], self.candidate_df)
            objs[i, 0] = rev
            objs[i, 1] = conv
            objs[i, 2] = -cost      # Minimizing cost
            objs[i, 3] = -fatigue   # Minimizing fatigue
        return objs

    @staticmethod
    def _fast_non_dominated_sort(objs: np.ndarray) -> List[List[int]]:
        """Deb's fast non-dominated sorting algorithm."""
        N = len(objs)
        domination_counts = np.zeros(N, dtype=int)
        dominated_solutions = [[] for _ in range(N)]
        fronts: List[List[int]] = [[]]

        for p in range(N):
            for q in range(p + 1, N):
                # Check dominance (Maximizing all 4 objectives in negated space)
                p_dominates_q = (objs[p] >= objs[q]).all() and (objs[p] > objs[q]).any()
                q_dominates_p = (objs[q] >= objs[p]).all() and (objs[q] > objs[p]).any()

                if p_dominates_q:
                    dominated_solutions[p].append(q)
                    domination_counts[q] += 1
                elif q_dominates_p:
                    dominated_solutions[q].append(p)
                    domination_counts[p] += 1

            if domination_counts[p] == 0:
                fronts[0].append(p)

        i = 0
        while len(fronts[i]) > 0:
            next_front = []
            for p in fronts[i]:
                for q in dominated_solutions[p]:
                    domination_counts[q] -= 1
                    if domination_counts[q] == 0:
                        next_front.append(q)
            i += 1
            fronts.append(next_front)

        return [f for f in fronts if len(f) > 0]

    @staticmethod
    def _compute_crowding_distance(objs: np.ndarray, front: List[int]) -> np.ndarray:
        """Computes diversity crowding distance for solutions in a Pareto front."""
        dist = np.zeros(len(front))
        if len(front) <= 2:
            return np.full(len(front), np.inf)

        sub_objs = objs[front]
        n_objs = sub_objs.shape[1]

        for m in range(n_objs):
            sorted_idx = np.argsort(sub_objs[:, m])
            dist[sorted_idx[0]] = np.inf
            dist[sorted_idx[-1]] = np.inf
            val_range = sub_objs[sorted_idx[-1], m] - sub_objs[sorted_idx[0], m]
            if val_range > 1e-6:
                for k in range(1, len(front) - 1):
                    dist[sorted_idx[k]] += (sub_objs[sorted_idx[k + 1], m] - sub_objs[sorted_idx[k - 1], m]) / val_range

        return dist

    def run_optimization(self) -> Dict[str, Any]:
        """Executes the NSGA-II evolutionary search loop."""
        population = self._initialize_population()
        objs = self._evaluate_population(population)

        for gen in range(self.n_gen):
            # Binary tournament selection & mating
            offspring = []
            for _ in range(self.pop_size // 2):
                # Tournament
                i1, i2 = self.rng.integers(0, self.pop_size, size=2)
                p1 = population[i1] if (objs[i1, 0] >= objs[i2, 0]) else population[i2]
                i3, i4 = self.rng.integers(0, self.pop_size, size=2)
                p2 = population[i3] if (objs[i3, 0] >= objs[i4, 0]) else population[i4]

                # Uniform crossover
                if self.rng.random() < self.crossover_rate:
                    mask = self.rng.random(self.n_candidates) < 0.5
                    c1 = np.where(mask, p1, p2)
                    c2 = np.where(mask, p2, p1)
                else:
                    c1, c2 = p1.copy(), p2.copy()

                # Bit-flip mutation
                mut_mask1 = self.rng.random(self.n_candidates) < self.mutation_rate
                mut_mask2 = self.rng.random(self.n_candidates) < self.mutation_rate
                c1 = np.bitwise_xor(c1, mut_mask1.astype(np.int8))
                c2 = np.bitwise_xor(c2, mut_mask2.astype(np.int8))

                offspring.append(self._repair_solution(c1))
                offspring.append(self._repair_solution(c2))

            offspring = np.array(offspring, dtype=np.int8)
            offspring_objs = self._evaluate_population(offspring)

            # Combine Parent + Offspring (2N)
            combined_pop = np.vstack([population, offspring])
            combined_objs = np.vstack([objs, offspring_objs])

            # Non-dominated sort
            fronts = self._fast_non_dominated_sort(combined_objs)

            new_pop_indices = []
            for front in fronts:
                if len(new_pop_indices) + len(front) <= self.pop_size:
                    new_pop_indices.extend(front)
                else:
                    needed = self.pop_size - len(new_pop_indices)
                    crowd_dist = self._compute_crowding_distance(combined_objs, front)
                    sorted_by_crowd = np.argsort(-crowd_dist)
                    new_pop_indices.extend([front[k] for k in sorted_by_crowd[:needed]])
                    break

            population = combined_pop[new_pop_indices]
            objs = combined_objs[new_pop_indices]

        # Extract Rank 1 (Pareto Frontier)
        fronts = self._fast_non_dominated_sort(objs)
        pareto_indices = fronts[0]
        pareto_pop = population[pareto_indices]
        pareto_raw_objs = objs[pareto_indices]

        # Convert back to natural objective metrics
        pareto_records = []
        for i, idx in enumerate(pareto_indices):
            rev = float(pareto_raw_objs[i, 0])
            conv = float(pareto_raw_objs[i, 1])
            cost = float(-pareto_raw_objs[i, 2])
            fatigue = float(-pareto_raw_objs[i, 3])
            roi = round((rev - cost) / max(cost, 1e-4) * 100.0, 2)
            n_targeted = int(np.sum(pareto_pop[i]))

            pareto_records.append({
                "Solution_ID": i,
                "Expected_Revenue": rev,
                "Expected_Conversions": conv,
                "Campaign_Cost": cost,
                "Customer_Fatigue": fatigue,
                "ROI_Pct": roi,
                "Targeted_Customers": n_targeted,
                "Allocation_Vector": pareto_pop[i]
            })

        pareto_df = pd.DataFrame(pareto_records).drop_duplicates(subset=["Campaign_Cost", "Expected_Revenue"])
        
        # Strategy Clustering on Pareto Frontier (Abildaeva et al. 2026)
        if len(pareto_df) >= 3:
            cluster_features = pareto_df[["Campaign_Cost", "Expected_Revenue", "ROI_Pct"]].values
            kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
            clusters = kmeans.fit_predict(cluster_features)

            # Map cluster labels by ROI rank
            cluster_order = np.argsort([-kmeans.cluster_centers_[c, 2] for c in range(3)])
            cluster_map = {
                cluster_order[0]: "High-ROI (Cost Efficient)",
                cluster_order[1]: "Balanced Growth",
                cluster_order[2]: "Aggressive Reach (Scale)"
            }
            pareto_df["Strategy_Archetype"] = [cluster_map[c] for c in clusters]
        else:
            pareto_df["Strategy_Archetype"] = "Balanced Growth"

        return {
            "pareto_dataframe": pareto_df,
            "pareto_solutions": pareto_pop,
            "population_size": self.pop_size,
            "generations": self.n_gen
        }


if __name__ == "__main__":
    import sys
    from pathlib import Path
    sys.path.append(str(Path(__file__).resolve().parent.parent))
    from data.data_loader import get_preprocessed_data
    from data.channel_touchpoint_generator import get_channel_touchpoint_data

    df = get_preprocessed_data()
    tp_df = get_channel_touchpoint_data(df)
    
    # Test optimizer on subset for fast verification
    optimizer = NSGA2CampaignOptimizer(
        candidate_df=tp_df.head(500),
        constraints=CampaignConstraints(total_budget=1500.0),
        population_size=30,
        n_generations=15
    )
    res = optimizer.run_optimization()
    print("Pareto Solutions Found:", len(res["pareto_dataframe"]))
    print(res["pareto_dataframe"][["Solution_ID", "Strategy_Archetype", "Campaign_Cost", "Expected_Revenue", "ROI_Pct", "Customer_Fatigue"]].head(6))
