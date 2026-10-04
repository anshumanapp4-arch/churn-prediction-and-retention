import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

class StrategyEvaluator:
    """
    Evaluates baseline strategies against the 0/1 Knapsack Optimized selection.
    Performs sensitivity analysis over budgets, offer success rates, and costs.
    """
    def __init__(self, output_dir: str = os.path.join('outputs', 'plots')):
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

    def evaluate_all_strategies(self, df_opt: pd.DataFrame, budget: float, y_test: np.ndarray = None, seed: int = 42) -> pd.DataFrame:
        """
        Compares 4 targeting strategies under identical budget B:
        1. Random Selection
        2. Top-k by Churn Probability
        3. Top-k by Customer Value
        4. Optimized Selection (0/1 Knapsack)
        """
        df = df_opt.copy().reset_index(drop=True)
        np.random.seed(seed)
        
        if y_test is not None:
            df['actual_churn'] = y_test

        results = []

        # Strategy 1: Random Selection
        shuffled_indices = np.random.permutation(df.index)
        rand_spent = 0.0
        rand_selected = []
        for idx in shuffled_indices:
            cost = df.loc[idx, 'c_i']
            if rand_spent + cost <= budget:
                rand_selected.append(idx)
                rand_spent += cost
        
        df_rand = df.loc[rand_selected]
        results.append({
            'Strategy': 'Random Selection',
            'Targeted_Count': len(df_rand),
            'Budget_Spent': df_rand['c_i'].sum(),
            'Expected_Net_Benefit': df_rand['E_i'].sum(),
            'Expected_Retained_Value': (df_rand['churn_prob'] * df_rand['E_i'].apply(lambda x: 0.3) * df_rand['V_i']).sum(),
            'Realized_Value': (df_rand['actual_churn'] * 0.3 * df_rand['V_i'] - df_rand['c_i']).sum() if y_test is not None else 0.0
        })

        # Strategy 2: Top-k by Churn Probability
        top_prob_df = df.sort_values(by='churn_prob', ascending=False)
        prob_spent = 0.0
        prob_selected = []
        for idx in top_prob_df.index:
            cost = df.loc[idx, 'c_i']
            if prob_spent + cost <= budget:
                prob_selected.append(idx)
                prob_spent += cost

        df_prob = df.loc[prob_selected]
        results.append({
            'Strategy': 'Top-k Churn Probability',
            'Targeted_Count': len(df_prob),
            'Budget_Spent': df_prob['c_i'].sum(),
            'Expected_Net_Benefit': df_prob['E_i'].sum(),
            'Expected_Retained_Value': (df_prob['churn_prob'] * 0.3 * df_prob['V_i']).sum(),
            'Realized_Value': (df_prob['actual_churn'] * 0.3 * df_prob['V_i'] - df_prob['c_i']).sum() if y_test is not None else 0.0
        })

        # Strategy 3: Top-k by Customer Value
        top_val_df = df.sort_values(by='V_i', ascending=False)
        val_spent = 0.0
        val_selected = []
        for idx in top_val_df.index:
            cost = df.loc[idx, 'c_i']
            if val_spent + cost <= budget:
                val_selected.append(idx)
                val_spent += cost

        df_val = df.loc[val_selected]
        results.append({
            'Strategy': 'Top-k Customer Value',
            'Targeted_Count': len(df_val),
            'Budget_Spent': df_val['c_i'].sum(),
            'Expected_Net_Benefit': df_val['E_i'].sum(),
            'Expected_Retained_Value': (df_val['churn_prob'] * 0.3 * df_val['V_i']).sum(),
            'Realized_Value': (df_val['actual_churn'] * 0.3 * df_val['V_i'] - df_val['c_i']).sum() if y_test is not None else 0.0
        })

        # Strategy 4: Optimized Selection (0/1 Knapsack)
        df_opt_sel = df[df['selected_exact'] == 1]
        results.append({
            'Strategy': 'Optimized (0/1 Knapsack)',
            'Targeted_Count': len(df_opt_sel),
            'Budget_Spent': df_opt_sel['c_i'].sum(),
            'Expected_Net_Benefit': df_opt_sel['E_i'].sum(),
            'Expected_Retained_Value': (df_opt_sel['churn_prob'] * 0.3 * df_opt_sel['V_i']).sum(),
            'Realized_Value': (df_opt_sel['actual_churn'] * 0.3 * df_opt_sel['V_i'] - df_opt_sel['c_i']).sum() if y_test is not None else 0.0
        })

        results_df = pd.DataFrame(results)
        print("=== Baseline Strategy Comparison ===")
        print(results_df.to_string(index=False))
        
        self._plot_baseline_comparison(results_df)
        return results_df

    def _plot_baseline_comparison(self, results_df: pd.DataFrame):
        plt.figure(figsize=(9, 5))
        sns.barplot(x='Strategy', y='Expected_Net_Benefit', data=results_df, palette='crest')
        plt.title('Expected Net Benefit by Targeting Strategy (Same Budget)')
        plt.ylabel('Expected Net Benefit ($)')
        plt.xticks(rotation=15)
        plt.grid(True, alpha=0.3)

        save_path = os.path.join(self.output_dir, 'baseline_comparison.png')
        plt.tight_layout()
        plt.savefig(save_path, dpi=300)
        plt.close()
        print(f"[StrategyEvaluator] Saved baseline comparison plot to {save_path}")

    def plot_sensitivity_analysis(self, df_opt: pd.DataFrame, base_budget: float, optimizer_cls):
        """
        Performs sensitivity analysis over varying budgets and success rates (s = 0.1 to 0.5).
        """
        budgets = np.linspace(base_budget * 0.2, base_budget * 2.0, 10)
        success_rates = [0.15, 0.30, 0.45]

        plt.figure(figsize=(10, 6))

        for s in success_rates:
            opt = optimizer_cls(offer_success_rate=s)
            benefits = []
            for b in budgets:
                df_temp = opt.compute_expected_net_benefit(df_opt, df_opt['churn_prob'].values)
                x_sel = opt.solve_greedy_heuristic(df_temp, b)
                net_benefit = df_temp.loc[x_sel == 1, 'E_i'].sum()
                benefits.append(net_benefit)

            plt.plot(budgets, benefits, marker='o', linewidth=2.5, label=f'Success Rate s = {int(s*100)}%')

        plt.xlabel('Campaign Budget ($)')
        plt.ylabel('Expected Net Benefit ($)')
        plt.title('Sensitivity Analysis: Net Benefit vs Budget & Offer Success Rate')
        plt.legend()
        plt.grid(True, alpha=0.3)

        save_path = os.path.join(self.output_dir, 'sensitivity_analysis.png')
        plt.tight_layout()
        plt.savefig(save_path, dpi=300)
        plt.close()
        print(f"[StrategyEvaluator] Saved sensitivity analysis plot to {save_path}")
