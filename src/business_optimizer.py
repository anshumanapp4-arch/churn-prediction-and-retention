import os
import numpy as np
import pandas as pd
from scipy.optimize import milp, LinearConstraint, Bounds
import pulp

class RetentionOptimizer:
    """
    Business Layer & Optimization Engine.
    Translates model predicted churn probabilities into optimal retention decisions under budget constraints.
    Formulated as a 0/1 Knapsack Problem.
    """
    def __init__(self, offer_success_rate: float = 0.30, offer_cost_pct: float = 0.15, flat_offer_cost: float = None):
        self.s = offer_success_rate
        self.offer_cost_pct = offer_cost_pct
        self.flat_offer_cost = flat_offer_cost

    def compute_expected_net_benefit(self, df_test: pd.DataFrame, churn_probs: np.ndarray) -> pd.DataFrame:
        """
        Computes expected customer value Vi, offer cost ci, and expected net benefit Ei for each customer.
        Formula: Ei = pi * s * Vi - ci
        """
        df = df_test.copy().reset_index(drop=True)
        df['churn_prob'] = churn_probs

        # Customer Value Vi (Annual Value: MonthlyCharges * 12 or TotalCharges if MonthlyCharges missing)
        if 'MonthlyCharges' in df.columns:
            df['V_i'] = df['MonthlyCharges'] * 12.0
        elif 'total_charges' in df.columns or 'TotalCharges' in df.columns:
            df['V_i'] = df.get('TotalCharges', df.get('total_charges', 1000.0))
        else:
            df['V_i'] = 1000.0  # Fallback default annual value

        # Offer Cost ci
        if self.flat_offer_cost is not None:
            df['c_i'] = float(self.flat_offer_cost)
        else:
            df['c_i'] = df['V_i'] * self.offer_cost_pct

        # Expected Net Benefit Ei = pi * s * Vi - ci
        df['E_i'] = df['churn_prob'] * self.s * df['V_i'] - df['c_i']

        # Efficiency ratio for greedy heuristic
        df['ROI_ratio'] = np.where(df['c_i'] > 0, df['E_i'] / df['c_i'], 0)

        # Rule: Filter candidates where E_i > 0
        df['eligible'] = df['E_i'] > 0

        return df

    def solve_exact_milp(self, df_opt: pd.DataFrame, budget: float) -> np.ndarray:
        """
        Solves 0/1 Knapsack problem exactly using SciPy MILP solver.
        Maximize sum(x_i * E_i) subject to sum(x_i * c_i) <= Budget, x_i in {0, 1}.
        """
        n = len(df_opt)
        E_vec = df_opt['E_i'].values
        c_vec = df_opt['c_i'].values
        eligible_mask = df_opt['eligible'].values

        # Set E_i to negative infinity for non-eligible candidates so solver ignores them
        E_vec_effective = np.where(eligible_mask, E_vec, -1e9)

        # MILP minimizes c^T x, so we minimize -E_vec_effective
        c_obj = -E_vec_effective

        constraints = LinearConstraint(c_vec, 0, budget)
        bounds = Bounds(0, 1) # Binary bounds [0, 1]
        integrality = np.ones(n) # All variables are integers (binary)

        res = milp(c=c_obj, constraints=constraints, bounds=bounds, integrality=integrality)

        if res.success:
            x_selected = np.round(res.x).astype(int)
            # Ensure non-eligible are not selected
            x_selected = x_selected * eligible_mask.astype(int)
            return x_selected
        else:
            print("[RetentionOptimizer] MILP solver warning: fallback to PuLP solver.")
            return self.solve_pulp_knapsack(df_opt, budget)

    def solve_pulp_knapsack(self, df_opt: pd.DataFrame, budget: float) -> np.ndarray:
        """
        Solves 0/1 Knapsack using PuLP Integer Linear Programming.
        """
        prob = pulp.LpProblem("Retention_Optimization", pulp.LpMaximize)
        n = len(df_opt)
        
        x = [pulp.LpVariable(f"x_{i}", cat="Binary") for i in range(n)]

        E_vec = df_opt['E_i'].values
        c_vec = df_opt['c_i'].values
        eligible_mask = df_opt['eligible'].values

        # Objective Function
        prob += pulp.lpSum([x[i] * E_vec[i] for i in range(n) if eligible_mask[i]])

        # Budget Constraint
        prob += pulp.lpSum([x[i] * c_vec[i] for i in range(n) if eligible_mask[i]]) <= budget

        prob.solve(pulp.PULP_CBC_CMD(msg=0))

        x_selected = np.array([int(pulp.value(x[i])) if eligible_mask[i] else 0 for i in range(n)])
        return x_selected

    def solve_greedy_heuristic(self, df_opt: pd.DataFrame, budget: float) -> np.ndarray:
        """
        Solves Knapsack problem via Greedy Heuristic:
        Sort eligible candidates by ROI ratio (E_i / c_i) descending and select until budget runs out.
        """
        n = len(df_opt)
        x_selected = np.zeros(n, dtype=int)

        # Eligible candidates sorted by ROI ratio
        eligible_df = df_opt[df_opt['eligible']].copy()
        sorted_indices = eligible_df.sort_values(by='ROI_ratio', ascending=False).index

        current_cost = 0.0
        for idx in sorted_indices:
            cost = df_opt.loc[idx, 'c_i']
            if current_cost + cost <= budget:
                x_selected[idx] = 1
                current_cost += cost

        return x_selected

    def optimize_retention(self, df_test: pd.DataFrame, churn_probs: np.ndarray, budget: float = None):
        """
        Full retention optimization workflow.
        Returns evaluation results and target list DataFrame.
        """
        df_opt = self.compute_expected_net_benefit(df_test, churn_probs)

        # Default budget: spend enough to target ~15% of candidates if budget not provided
        if budget is None:
            avg_cost = df_opt['c_i'].mean()
            budget = avg_cost * (0.15 * len(df_opt))

        # Solve Exact MILP
        x_exact = self.solve_exact_milp(df_opt, budget)
        df_opt['selected_exact'] = x_exact

        # Solve Greedy Heuristic
        x_greedy = self.solve_greedy_heuristic(df_opt, budget)
        df_opt['selected_greedy'] = x_greedy

        # Compute summary metrics for exact solution
        selected_df = df_opt[df_opt['selected_exact'] == 1]
        total_targeted = len(selected_df)
        total_cost_spent = selected_df['c_i'].sum()
        total_expected_net_benefit = selected_df['E_i'].sum()
        total_retained_value = (selected_df['churn_prob'] * self.s * selected_df['V_i']).sum()

        print(f"=== Retention Optimization Results ===")
        print(f"Total Test Customers: {len(df_opt)}")
        print(f"Available Budget:     ${budget:,.2f}")
        print(f"Customers Targeted:   {total_targeted} ({total_targeted/len(df_opt)*100:.1f}%)")
        print(f"Total Budget Spent:   ${total_cost_spent:,.2f}")
        print(f"Expected Net Benefit: ${total_expected_net_benefit:,.2f}")
        print(f"Expected Value Saved: ${total_retained_value:,.2f}")

        return df_opt, {
            'budget': budget,
            'total_targeted': total_targeted,
            'total_cost_spent': total_cost_spent,
            'total_expected_net_benefit': total_expected_net_benefit,
            'total_retained_value': total_retained_value
        }

    def export_target_list(self, df_opt: pd.DataFrame, id_col: str = 'customerID', output_path: str = os.path.join('outputs', 'target_list.csv')):
        """
        Exports final target list of selected customer IDs to CSV.
        """
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        selected_customers = df_opt[df_opt['selected_exact'] == 1].copy()
        
        cols_to_export = [c for c in [id_col, 'churn_prob', 'V_i', 'c_i', 'E_i', 'ROI_ratio'] if c in selected_customers.columns]
        target_list = selected_customers[cols_to_export].sort_values(by='E_i', ascending=False)
        
        target_list.to_csv(output_path, index=False)
        print(f"[RetentionOptimizer] Target list of {len(target_list)} customers saved to {output_path}")
        return target_list
