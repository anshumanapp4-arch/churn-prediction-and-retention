# Telecom Churn Prediction and Retention Optimization

A complete, production-ready 14-phase machine learning and decision optimization system. This framework predicts customer churn risk and solves a **0/1 Knapsack Problem** to optimize retention offer targeting under budget constraints.

---

## 🚀 Features

- **End-to-End Pipeline**: Ingests raw data, cleans data, builds scikit-learn pipelines, trains models, evaluates metrics, and optimizes business targeting.
- **Works on Any Dataset**: Auto-detects numeric vs categorical features; accepts custom schemas and column configurations.
- **Data Leakage Prevention**: Stratified train/test split with strict pipeline fitting on training data only.
- **Interpretable ML**: Logistic Regression baseline tuned with 5-fold cross-validation & `GridSearchCV`. Benchmarked against Random Forest & Gradient Boosting.
- **Business Layer & Knapsack Optimization**: Formulates targeting as a 0/1 Knapsack optimization solved via SciPy Integer Linear Programming (`MILP`) & PuLP, plus a Greedy Heuristic ($E_i / c_i$).
- **Baseline Comparison & Sensitivity Analysis**: Compares against Random, Top-k Churn Probability, and Top-k Customer Value strategies. Evaluates net benefit sensitivity across budgets and offer success rates.

---

## 📁 Repository Structure

```
d:/churn prediction and retention/
├── data/
│   ├── raw/                  # Raw input datasets (e.g., telco_churn.csv)
│   └── processed/            # Cleaned data ready for model ingestion
├── notebooks/
│   ├── 01_eda.ipynb          # Exploratory Data Analysis & cleaning
│   ├── 02_model.ipynb        # Model training, tuning, and evaluation
│   └── 03_optimization.ipynb # Business layer & Knapsack optimization
├── src/
│   ├── __init__.py
│   ├── data_cleaner.py       # Robust data cleaner & type converter
│   ├── pipeline.py           # ColumnTransformer with imputer, scaler, onehot
│   ├── model_trainer.py      # Stratified split, Logistic Regression, tuning
│   ├── evaluator.py          # ROC-AUC, PR-AUC, calibration & feature coefficients
│   ├── business_optimizer.py # Expected benefit calculation & 0/1 Knapsack solver
│   └── baselines.py          # Strategy comparisons & sensitivity analysis
├── outputs/
│   ├── plots/                # ROC/PR curves, feature drivers, baseline plots
│   └── target_list.csv       # Recommended customer list for retention offers
├── generate_sample_data.py   # Self-contained IBM Telco Churn dataset generator
├── run_pipeline.py           # Master execution script
├── requirements.txt          # Python dependencies
└── README.md                 # Project documentation
```

---

## 🛠️ Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Run Complete Pipeline
To run the full end-to-end system on the default/sample dataset:
```bash
python run_pipeline.py
```

### 3. Run on Custom CSV Dataset
To test on your own churn dataset:
```bash
python run_pipeline.py --data_path "path/to/your_dataset.csv" --target_col "Churn" --id_col "CustomerID" --budget 15000
```

---

## 📊 Business Logic & Optimization Formula

1. **Customer Value ($V_i$)**: Expected revenue if retained (e.g., `MonthlyCharges * 12`).
2. **Offer Cost ($c_i$)**: Campaign cost per customer (e.g., 15% of annual bill).
3. **Success Rate ($s$)**: Acceptance probability of offer (default 30%).
4. **Expected Net Benefit ($E_i$)**:
   $$E_i = p_i \cdot s \cdot V_i - c_i$$
   *Note: Customers with $E_i \le 0$ are automatically excluded from targeting.*

5. **0/1 Knapsack Problem**:
   $$\text{Maximize } \sum x_i \cdot E_i \quad \text{subject to } \sum x_i \cdot c_i \le B, \quad x_i \in \{0, 1\}$$

---

## 📈 Deliverables

After running `run_pipeline.py`, check the `outputs/` folder:
- `outputs/target_list.csv`: List of customer IDs selected for offers, ranked by ROI.
- `outputs/plots/eda_summary.png`: Visual insights on contract, tenure, and charges.
- `outputs/plots/roc_pr_curves.png`: Model discrimination evaluation.
- `outputs/plots/calibration_curve.png`: Probability calibration graph.
- `outputs/plots/feature_coefficients.png`: Top positive and negative churn drivers.
- `outputs/plots/baseline_comparison.png`: Net benefit comparison across targeting strategies.
- `outputs/plots/sensitivity_analysis.png`: Net benefit vs budget across success rates.
