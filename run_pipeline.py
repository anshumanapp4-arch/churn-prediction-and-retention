import os
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from generate_sample_data import generate_telco_churn_dataset
from src.data_cleaner import DataCleaner
from src.pipeline import build_preprocessing_pipeline
from src.model_trainer import ModelTrainer
from src.evaluator import ModelEvaluator
from src.business_optimizer import RetentionOptimizer
from src.baselines import StrategyEvaluator

def run_eda(df: pd.DataFrame, output_dir: str, target_col: str = 'Churn'):
    """
    Performs Exploratory Data Analysis (Phase 4) and saves visualizations.
    """
    os.makedirs(output_dir, exist_ok=True)
    print("\n--- Phase 4: Exploratory Data Analysis (EDA) ---")
    print(f"Dataset Shape: {df.shape[0]} rows, {df.shape[1]} columns")
    churn_rate = df[target_col].mean()
    print(f"Overall Churn Rate: {churn_rate*100:.2f}%")

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # 1. Churn Rate by Contract Type
    if 'Contract' in df.columns:
        sns.barplot(x='Contract', y=target_col, data=df, ax=axes[0, 0], palette='Blues_d')
        axes[0, 0].set_title('Churn Rate by Contract Type')
        axes[0, 0].set_ylabel('Churn Rate')

    # 2. Churn vs Tenure Boxplot
    if 'tenure' in df.columns:
        sns.boxplot(x=target_col, y='tenure', data=df, ax=axes[0, 1], palette='Set2')
        axes[0, 1].set_title('Tenure Distribution by Churn')
        axes[0, 1].set_xticklabels(['Retained (0)', 'Churned (1)'])

    # 3. Churn vs Monthly Charges
    if 'MonthlyCharges' in df.columns:
        sns.kdeplot(data=df, x='MonthlyCharges', hue=target_col, ax=axes[1, 0], common_norm=False, palette='crest')
        axes[1, 0].set_title('Monthly Charges Distribution by Churn')

    # 4. Correlation Heatmap
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) > 1:
        corr = df[numeric_cols].corr()
        sns.heatmap(corr, annot=True, fmt='.2f', cmap='coolwarm', ax=axes[1, 1], cbar=True)
        axes[1, 1].set_title('Numeric Features Correlation Matrix')

    plt.tight_layout()
    eda_path = os.path.join(output_dir, 'eda_summary.png')
    plt.savefig(eda_path, dpi=300)
    plt.close()
    print(f"[EDA] Saved EDA plots to {eda_path}")

def main():
    parser = argparse.ArgumentParser(description="Telecom Churn Prediction and Retention Optimization Pipeline")
    parser.add_argument("--data_path", type=str, default=os.path.join("data", "raw", "telco_churn.csv"), help="Path to raw CSV dataset")
    parser.add_argument("--target_col", type=str, default="Churn", help="Target column name")
    parser.add_argument("--id_col", type=str, default="customerID", help="Customer ID column name")
    parser.add_argument("--budget", type=float, default=None, help="Campaign budget constraint ($)")
    parser.add_argument("--success_rate", type=float, default=0.30, help="Offer success rate s (default: 0.30)")
    parser.add_argument("--offer_cost_pct", type=float, default=0.15, help="Offer cost as fraction of annual bill (default: 0.15)")
    args = parser.parse_args()

    plots_dir = os.path.join("outputs", "plots")
    os.makedirs(plots_dir, exist_ok=True)

    print("=========================================================================")
    print("        TELECOM CHURN PREDICTION AND RETENTION OPTIMIZATION SYSTEM        ")
    print("=========================================================================\n")

    # Phase 0 & 2: Ensure Data Exists
    if not os.path.exists(args.data_path):
        print(f"[Data Load] Raw data file '{args.data_path}' not found. Generating sample IBM Telco dataset...")
        os.makedirs(os.path.dirname(args.data_path), exist_ok=True)
        df_gen = generate_telco_churn_dataset()
        df_gen.to_csv(args.data_path, index=False)
        print(f"[Data Load] Generated sample dataset at '{args.data_path}'")

    print(f"Loading raw dataset from {args.data_path}...")
    ext = os.path.splitext(args.data_path)[1].lower()
    if ext in ['.xlsx', '.xls']:
        raw_df = pd.read_excel(args.data_path)
    else:
        raw_df = pd.read_csv(args.data_path)

    # Phase 3: Data Cleaning
    print("\n--- Phase 3: Data Cleaning & Preprocessing ---")
    cleaner = DataCleaner(target_col=args.target_col, id_col=args.id_col)
    clean_df = cleaner.clean_data(raw_df)
    processed_path = os.path.join("data", "processed", "processed_churn.csv")
    cleaner.save_processed(clean_df, processed_path)

    # Phase 4: EDA
    run_eda(clean_df, plots_dir, target_col=args.target_col)

    # Phase 5: Train / Test Split
    print("\n--- Phase 5: Stratified 80/20 Train/Test Split ---")
    trainer = ModelTrainer(target_col=args.target_col, id_col=args.id_col)
    X_train, X_test, y_train, y_test, ids_train, ids_test = trainer.split_data(clean_df)
    print(f"Training set: {X_train.shape[0]} samples | Test set: {X_test.shape[0]} samples (Locked)")

    # Phase 6: Build Preprocessing Pipeline (Fit on train only)
    print("\n--- Phase 6: Building Preprocessing Pipeline ---")
    preprocessor, num_cols, cat_cols = build_preprocessing_pipeline(clean_df, target_col=args.target_col, id_col=args.id_col)
    print(f"Numeric features ({len(num_cols)}): {num_cols}")
    print(f"Categorical features ({len(cat_cols)}): {cat_cols}")

    # Phase 7 & 8: Train Model
    print("\n--- Phase 7 & 8: Model Training & Tuning ---")
    best_model = trainer.train_logistic_regression(X_train, y_train, preprocessor, tune_c=True)

    # Save trained model pipeline for future prediction on new datasets
    import joblib
    model_path = os.path.join("outputs", "trained_model.joblib")
    joblib.dump(best_model, model_path)
    print(f"[ModelTrainer] Saved trained model pipeline to {model_path}")

    # Phase 9: Evaluation on Locked Test Set
    print("\n--- Phase 9: Evaluation on Test Set ---")
    evaluator = ModelEvaluator(output_dir=plots_dir)
    churn_probs, roc_auc, pr_auc = evaluator.evaluate_model(best_model, X_test, y_test, model_name="Logistic Regression")
    evaluator.plot_roc_pr_curves(y_test, churn_probs, model_name="Logistic Regression")
    evaluator.plot_calibration_curve(y_test, churn_probs, model_name="Logistic Regression")
    evaluator.extract_and_plot_coefficients(best_model)

    # Benchmark against Random Forest & Gradient Boosting
    print("\nTraining Benchmark Models...")
    benchmarks = trainer.train_benchmark_models(X_train, y_train, preprocessor)
    for name, bm_model in benchmarks.items():
        evaluator.evaluate_model(bm_model, X_test, y_test, model_name=name)

    # Phase 10 & 11: Business Layer & Retention Optimization
    print("\n--- Phase 10 & 11: Business Layer & 0/1 Knapsack Optimization ---")
    df_test_opt = X_test.copy()
    df_test_opt[args.id_col] = ids_test.values

    optimizer = RetentionOptimizer(offer_success_rate=args.success_rate, offer_cost_pct=args.offer_cost_pct)
    df_opt_results, summary_opt = optimizer.optimize_retention(df_test_opt, churn_probs, budget=args.budget)
    target_list_path = os.path.join("outputs", "target_list.csv")
    optimizer.export_target_list(df_opt_results, id_col=args.id_col, output_path=target_list_path)

    # Phase 12: Baselines & Sensitivity Analysis
    print("\n--- Phase 12: Baseline Strategy Comparison & Sensitivity Analysis ---")
    strategy_eval = StrategyEvaluator(output_dir=plots_dir)
    strategy_eval.evaluate_all_strategies(df_opt_results, budget=summary_opt['budget'], y_test=y_test)
    strategy_eval.plot_sensitivity_analysis(df_opt_results, base_budget=summary_opt['budget'], optimizer_cls=RetentionOptimizer)

    print("\n=========================================================================")
    print("                 PIPELINE RUN COMPLETED SUCCESSFULLY!                    ")
    print(f" Generated Plots: {os.path.abspath(plots_dir)}")
    print(f" Target List CSV: {os.path.abspath(target_list_path)}")
    print("=========================================================================\n")

if __name__ == "__main__":
    main()
