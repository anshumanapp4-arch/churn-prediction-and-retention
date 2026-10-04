import os
import argparse
import pandas as pd
import numpy as np
import joblib

from src.data_cleaner import DataCleaner
from src.business_optimizer import RetentionOptimizer

def predict_on_new_data(data_path: str, model_path: str = os.path.join("outputs", "trained_model.joblib"), id_col: str = "CustomerID", budget: float = 10000.0, offer_success_rate: float = 0.30):
    """
    Loads a trained model pipeline and makes predictions + 0/1 Knapsack targeting decisions
    on ANY new unseen customer dataset (CSV or Excel).
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Trained model not found at '{model_path}'. Please run 'python run_pipeline.py' first to train and save the model.")

    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Input file '{data_path}' not found.")

    print(f"\n=========================================================================")
    print(f"       INFERENCE ON UNSEEN DATASET: {os.path.basename(data_path)}")
    print(f"=========================================================================\n")

    # Load dataset (CSV or Excel)
    ext = os.path.splitext(data_path)[1].lower()
    if ext in ['.xlsx', '.xls']:
        raw_df = pd.read_excel(data_path)
    else:
        raw_df = pd.read_csv(data_path)

    print(f"Loaded dataset with {len(raw_df)} rows.")

    # Clean & Preprocess
    cleaner = DataCleaner(target_col="Churn", id_col=id_col)
    clean_df = cleaner.clean_data(raw_df)

    # Extract ID series
    customer_ids = clean_df[id_col] if id_col in clean_df.columns else pd.Series(clean_df.index, name=id_col)
    X_new = clean_df.drop(columns=[c for c in ["Churn", id_col] if c in clean_df.columns])

    # Load trained model pipeline
    print(f"Loading trained pipeline from {model_path}...")
    model = joblib.load(model_path)

    # Predict Churn Risk Probabilities
    print("Predicting churn probabilities...")
    churn_probs = model.predict_proba(X_new)[:, 1]

    # Run 0/1 Knapsack Optimization
    print(f"Running 0/1 Knapsack Retention Optimization (Budget: ${budget:,.2f})...")
    df_opt_input = X_new.copy()
    df_opt_input[id_col] = customer_ids.values

    optimizer = RetentionOptimizer(offer_success_rate=offer_success_rate, offer_cost_pct=0.15)
    df_results, summary = optimizer.optimize_retention(df_opt_input, churn_probs, budget=budget)

    # Save output predictions
    output_path = os.path.join("outputs", "new_data_targeting_results.csv")
    os.makedirs("outputs", exist_ok=True)
    
    export_cols = [c for c in [id_col, 'churn_prob', 'V_i', 'c_i', 'E_i', 'ROI_ratio', 'selected_exact'] if c in df_results.columns]
    results_export = df_results[export_cols].rename(columns={'selected_exact': 'Target_Offer_Decision'})
    results_export = results_export.sort_values(by='E_i', ascending=False)
    results_export.to_csv(output_path, index=False)

    print("\n=========================================================================")
    print(f" INFERENCE COMPLETED SUCCESSFULLY!")
    print(f" Results Saved to: {os.path.abspath(output_path)}")
    print(f" Selected Candidates: {summary['total_targeted']} / {len(clean_df)} customers")
    print(f" Total Expected Net Benefit: ${summary['total_expected_net_benefit']:,.2f}")
    print("=========================================================================\n")

    return results_export

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Churn Prediction & Retention Optimization on New Customer Data")
    parser.add_argument("--data_path", type=str, required=True, help="Path to new CSV or Excel customer file")
    parser.add_argument("--model_path", type=str, default=os.path.join("outputs", "trained_model.joblib"), help="Path to trained model file")
    parser.add_argument("--id_col", type=str, default="CustomerID", help="Customer ID column name")
    parser.add_argument("--budget", type=float, default=10000.0, help="Campaign budget ($)")
    args = parser.parse_args()

    predict_on_new_data(data_path=args.data_path, model_path=args.model_path, id_col=args.id_col, budget=args.budget)
