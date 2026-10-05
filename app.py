import os
import tempfile
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
import joblib

from generate_sample_data import generate_telco_churn_dataset
from src.data_cleaner import DataCleaner
from src.pipeline import build_preprocessing_pipeline
from src.model_trainer import ModelTrainer
from src.evaluator import ModelEvaluator
from src.business_optimizer import RetentionOptimizer
from src.baselines import StrategyEvaluator
from src.supabase_client import upload_file_to_supabase, log_campaign_to_supabase

# Page Configuration
st.set_page_config(
    page_title="Telecom Churn & Retention System",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling
st.markdown("""
    <style>
    .main {
        background-color: #0e1117;
    }
    .stMetric {
        background-color: #1e222d;
        padding: 15px;
        border-radius: 10px;
        border: 1px solid #2e364f;
    }
    .stAlert {
        border-radius: 8px;
    }
    .supabase-badge {
        background-color: #3ecf8e22;
        color: #3ecf8e;
        padding: 4px 10px;
        border-radius: 12px;
        font-weight: 600;
        font-size: 0.85rem;
        border: 1px solid #3ecf8e44;
    }
    </style>
""", unsafe_allow_html=True)

# Header Section
col_title, col_badge = st.columns([3, 1])
with col_title:
    st.title("⚡ Telecom Churn Prediction & Retention System")
    st.markdown("**Production 2-Stage Machine Learning & 0/1 Knapsack Optimization System**")
with col_badge:
    st.markdown("<br><span class='supabase-badge'>🟢 Supabase Cloud Active</span>", unsafe_allow_html=True)

st.divider()

# Sidebar Configuration
st.sidebar.header("⚙️ Control Panel")
mode = st.sidebar.radio(
    "Choose Mode:",
    ["⚡ Instant Prediction (Existing Model)", "🏋️ Full Train & Evaluate (New Dataset)"]
)

st.sidebar.divider()
st.sidebar.subheader("💰 Campaign Financial Settings")
budget_input = st.sidebar.number_input("Campaign Budget ($)", min_value=1000.0, max_value=500000.0, value=15000.0, step=1000.0)
success_rate_input = st.sidebar.slider("Offer Success Rate (s)", min_value=0.05, max_value=0.70, value=0.30, step=0.05)
offer_cost_pct_input = st.sidebar.slider("Offer Cost (% of Bill)", min_value=0.05, max_value=0.40, value=0.15, step=0.05)

st.sidebar.divider()
st.sidebar.subheader("📋 Dataset Column Settings")
id_col_input = st.sidebar.text_input("Customer ID Column", value="CustomerID")
target_col_input = st.sidebar.text_input("Target Column (if present)", value="Churn Label")

# File Upload Section
st.subheader("📂 Step 1: Upload Customer Dataset")
uploaded_file = st.file_uploader("Upload CSV or Excel file (.csv, .xlsx, .xls)", type=["csv", "xlsx", "xls"])

def load_uploaded_df(file):
    if file.name.endswith('.csv'):
        return pd.read_csv(file)
    else:
        return pd.read_excel(file)

df_raw = None
if uploaded_file is not None:
    try:
        df_raw = load_uploaded_df(uploaded_file)
        
        # Save temp copy and upload to Supabase Storage
        temp_dir = tempfile.gettempdir()
        temp_path = os.path.join(temp_dir, uploaded_file.name)
        with open(temp_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        supabase_url = upload_file_to_supabase(temp_path, bucket_name="churn-datasets")
        if supabase_url:
            st.success(f"Loaded **{uploaded_file.name}** ({df_raw.shape[0]} rows, {df_raw.shape[1]} cols) | ☁️ Uploaded to Supabase Storage")
        else:
            st.success(f"Loaded **{uploaded_file.name}** ({df_raw.shape[0]} rows, {df_raw.shape[1]} cols)")
    except Exception as e:
        st.error(f"Error reading file: {e}")
else:
    st.info("💡 No file uploaded yet. Click below to load the built-in IBM Telco sample dataset for testing.")
    if st.button("Load Built-in Telco Sample Dataset"):
        sample_path = os.path.join("data", "raw", "telco_churn.csv")
        if not os.path.exists(sample_path):
            os.makedirs(os.path.dirname(sample_path), exist_ok=True)
            df_gen = generate_telco_churn_dataset()
            df_gen.to_csv(sample_path, index=False)
        df_raw = pd.read_csv(sample_path)
        st.success(f"Loaded IBM Telco Sample Dataset ({df_raw.shape[0]} rows, {df_raw.shape[1]} columns)")

if df_raw is not None:
    with st.expander("🔍 View Data Preview"):
        st.dataframe(df_raw.head(10), use_container_width=True)

    st.divider()

    # MODE 1: INSTANT PREDICTION
    if mode == "⚡ Instant Prediction (Existing Model)":
        st.subheader("⚡ Step 2: Instant Churn Risk Prediction & Retention Targeting")
        
        model_path = os.path.join("outputs", "trained_model.joblib")
        if not os.path.exists(model_path):
            st.warning("⚠️ No pre-trained model file found. Please run 'Full Train & Evaluate' first to train and save the model.")
        else:
            if st.button("🚀 Run Instant Prediction & Knapsack Optimization", type="primary"):
                with st.spinner("Cleaning dataset, making predictions, and solving 0/1 Knapsack optimization..."):
                    cleaner = DataCleaner(target_col=target_col_input, id_col=id_col_input)
                    clean_df = cleaner.clean_data(df_raw)

                    actual_id_col = id_col_input if id_col_input in clean_df.columns else ('customerID' if 'customerID' in clean_df.columns else clean_df.columns[0])
                    ids = clean_df[actual_id_col] if actual_id_col in clean_df.columns else pd.Series(clean_df.index)
                    X_features = clean_df.drop(columns=[c for c in [target_col_input, 'Churn', 'Churn Label', actual_id_col] if c in clean_df.columns])

                    # Feature alignment to ensure compatibility with trained ColumnTransformer
                    MODEL_NUM_COLS = ['SeniorCitizen', 'tenure', 'MonthlyCharges', 'TotalCharges']
                    MODEL_CAT_COLS = [
                        'gender', 'Partner', 'Dependents', 'PhoneService', 'MultipleLines',
                        'InternetService', 'OnlineSecurity', 'OnlineBackup', 'DeviceProtection',
                        'TechSupport', 'StreamingTV', 'StreamingMovies', 'Contract',
                        'PaperlessBilling', 'PaymentMethod'
                    ]

                    for col in MODEL_NUM_COLS:
                        if col not in X_features.columns:
                            X_features[col] = 0.0
                        else:
                            X_features[col] = pd.to_numeric(X_features[col], errors='coerce').fillna(0.0)

                    for col in MODEL_CAT_COLS:
                        if col not in X_features.columns:
                            X_features[col] = "Male" if col == "gender" else ("Month-to-month" if col == "Contract" else "No")
                        else:
                            X_features[col] = X_features[col].astype(str).fillna("No")

                    X_aligned = X_features[MODEL_NUM_COLS + MODEL_CAT_COLS]

                    # Load model & Predict
                    model = joblib.load(model_path)
                    churn_probs = model.predict_proba(X_aligned)[:, 1]

                    # Optimize
                    df_opt_input = X_aligned.copy()
                    df_opt_input[actual_id_col] = ids.values

                    optimizer = RetentionOptimizer(offer_success_rate=success_rate_input, offer_cost_pct=offer_cost_pct_input)
                    df_results, summary = optimizer.optimize_retention(df_opt_input, churn_probs, budget=budget_input)

                    # Log to Supabase
                    log_campaign_to_supabase({
                        "mode": "Instant Prediction",
                        "total_customers": len(clean_df),
                        "budget": budget_input,
                        "targeted_count": summary['total_targeted'],
                        "budget_spent": summary['total_cost_spent'],
                        "expected_net_benefit": summary['total_expected_net_benefit']
                    })

                    # Display KPIs
                    col1, col2, col3, col4 = st.columns(4)
                    col1.metric("Targeted Customers", f"{summary['total_targeted']} / {len(clean_df)}")
                    col2.metric("Budget Spent", f"${summary['total_cost_spent']:,.2f}")
                    col3.metric("Expected Net Profit", f"${summary['total_expected_net_benefit']:,.2f}")
                    col4.metric("Revenue Saved", f"${summary['total_retained_value']:,.2f}")

                    st.divider()

                    st.subheader("🎯 Targeted Candidates Decision Table")
                    export_cols = [c for c in [actual_id_col, 'churn_prob', 'V_i', 'c_i', 'E_i', 'ROI_ratio', 'selected_exact'] if c in df_results.columns]
                    targeted_df = df_results[df_results['selected_exact'] == 1][export_cols].sort_values(by='E_i', ascending=False)
                    st.dataframe(targeted_df, use_container_width=True)

                    csv_data = targeted_df.to_csv(index=False).encode('utf-8')
                    st.download_button(
                        label="📥 Download Targeted Candidates CSV",
                        data=csv_data,
                        file_name="targeted_retention_list.csv",
                        mime="text/csv"
                    )

    # MODE 2: FULL TRAIN & EVALUATE
    else:
        st.subheader("🏋️ Step 2: Full Train & Model Evaluation on New Dataset")
        
        if st.button("🚀 Train New Model & Optimize Retention", type="primary"):
            with st.spinner("Executing full 14-phase pipeline... Split, Preprocessing, GridSearchCV, Benchmarks & Knapsack Optimization"):
                cleaner = DataCleaner(target_col=target_col_input, id_col=id_col_input)
                clean_df = cleaner.clean_data(df_raw)

                actual_target = target_col_input if target_col_input in clean_df.columns else ('Churn' if 'Churn' in clean_df.columns else clean_df.columns[-1])
                actual_id = id_col_input if id_col_input in clean_df.columns else ('customerID' if 'customerID' in clean_df.columns else clean_df.columns[0])

                trainer = ModelTrainer(target_col=actual_target, id_col=actual_id)
                X_train, X_test, y_train, y_test, ids_train, ids_test = trainer.split_data(clean_df)

                preprocessor, num_cols, cat_cols = build_preprocessing_pipeline(clean_df, target_col=actual_target, id_col=actual_id)
                best_model = trainer.train_logistic_regression(X_train, y_train, preprocessor, tune_c=True)
                
                os.makedirs("outputs", exist_ok=True)
                joblib.dump(best_model, os.path.join("outputs", "trained_model.joblib"))

                churn_probs = best_model.predict_proba(X_test)[:, 1]
                evaluator = ModelEvaluator()
                _, roc_auc, pr_auc = evaluator.evaluate_model(best_model, X_test, y_test)

                m1, m2, m3 = st.columns(3)
                m1.metric("ROC-AUC Score", f"{roc_auc:.4f}")
                m2.metric("PR-AUC Score", f"{pr_auc:.4f}")
                m3.metric("Test Dataset Size", f"{len(X_test)} rows")

                st.divider()

                df_test_opt = X_test.copy()
                df_test_opt[actual_id] = ids_test.values

                optimizer = RetentionOptimizer(offer_success_rate=success_rate_input, offer_cost_pct=offer_cost_pct_input)
                df_opt_results, summary_opt = optimizer.optimize_retention(df_test_opt, churn_probs, budget=budget_input)

                log_campaign_to_supabase({
                    "mode": "Full Train & Evaluate",
                    "total_customers": len(clean_df),
                    "roc_auc": roc_auc,
                    "budget": budget_input,
                    "targeted_count": summary_opt['total_targeted'],
                    "budget_spent": summary_opt['total_cost_spent'],
                    "expected_net_benefit": summary_opt['total_expected_net_benefit']
                })

                st.subheader("📊 Targeting Strategy Baseline Comparison")
                strategy_eval = StrategyEvaluator()
                results_df = strategy_eval.evaluate_all_strategies(df_opt_results, budget=summary_opt['budget'], y_test=y_test)
                st.dataframe(results_df, use_container_width=True)

                st.subheader("🖼️ Evaluation & Sensitivity Plots")
                col_img1, col_img2 = st.columns(2)
                
                if os.path.exists(os.path.join("outputs", "plots", "roc_pr_curves.png")):
                    col_img1.image(os.path.join("outputs", "plots", "roc_pr_curves.png"), caption="ROC & Precision-Recall Curves")
                if os.path.exists(os.path.join("outputs", "plots", "baseline_comparison.png")):
                    col_img2.image(os.path.join("outputs", "plots", "baseline_comparison.png"), caption="Strategy Net Benefit Comparison")

                st.divider()
                selected_df = df_opt_results[df_opt_results['selected_exact'] == 1]
                csv_data = selected_df.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="📥 Download Full Evaluation Target List CSV",
                    data=csv_data,
                    file_name="trained_retention_target_list.csv",
                    mime="text/csv"
                )
