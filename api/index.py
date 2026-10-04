import os
import sys
import tempfile
import io
import json
import pandas as pd
import numpy as np

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from flask import Flask, request, jsonify, render_template_string, send_file
import joblib

from generate_sample_data import generate_telco_churn_dataset
from src.data_cleaner import DataCleaner
from src.pipeline import build_preprocessing_pipeline
from src.model_trainer import ModelTrainer
from src.evaluator import ModelEvaluator
from src.business_optimizer import RetentionOptimizer
from src.baselines import StrategyEvaluator
from src.supabase_client import upload_file_to_supabase, log_campaign_to_supabase

app = Flask(__name__)

HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Telecom Churn Prediction & Retention Optimization</title>
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css" rel="stylesheet">
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css">
    <style>
        :root {
            --bg-color: #0b0f19;
            --card-bg: #151c2c;
            --card-border: #232d42;
            --accent-color: #6366f1;
            --accent-hover: #4f46e5;
            --teal-color: #10b981;
            --text-main: #f3f4f6;
            --text-sub: #9ca3af;
        }
        body {
            background-color: var(--bg-color);
            color: var(--text-main);
            font-family: 'Segoe UI', system-ui, -apple-system, sans-serif;
            padding-bottom: 50px;
        }
        .navbar {
            background-color: #111827;
            border-bottom: 1px solid var(--card-border);
        }
        .card-custom {
            background-color: var(--card-bg);
            border: 1px solid var(--card-border);
            border-radius: 12px;
            padding: 24px;
            margin-bottom: 24px;
            box-shadow: 0 4px 20px rgba(0, 0, 0, 0.25);
        }
        .badge-supabase {
            background-color: rgba(16, 185, 129, 0.15);
            color: #10b981;
            border: 1px solid rgba(16, 185, 129, 0.3);
            padding: 6px 14px;
            border-radius: 20px;
            font-weight: 600;
            font-size: 0.85rem;
        }
        .kpi-card {
            background-color: #1e273b;
            border-left: 4px solid var(--accent-color);
            border-radius: 8px;
            padding: 16px;
        }
        .btn-accent {
            background-color: var(--accent-color);
            color: white;
            font-weight: 600;
            border: none;
            padding: 10px 20px;
            border-radius: 8px;
            transition: all 0.2s;
        }
        .btn-accent:hover {
            background-color: var(--accent-hover);
            color: white;
        }
        .btn-teal {
            background-color: var(--teal-color);
            color: white;
            font-weight: 600;
            border: none;
            padding: 10px 20px;
            border-radius: 8px;
        }
        .table-custom {
            color: var(--text-main);
            border-color: var(--card-border);
        }
        .table-custom th {
            background-color: #1c2538;
            color: var(--text-sub);
        }
    </style>
</head>
<body>

<nav class="navbar navbar-dark py-3">
    <div class="container">
        <span class="navbar-brand mb-0 h1 fs-4"><i class="fa-solid fa-bolt text-warning me-2"></i>Telecom Churn & Retention System</span>
        <span class="badge-supabase"><i class="fa-solid fa-cloud me-1"></i> Supabase Cloud Connected</span>
    </div>
</nav>

<div class="container mt-4">
    <div class="row">
        <!-- Control Sidebar -->
        <div class="col-lg-4">
            <div class="card-custom">
                <h5 class="fw-bold mb-3"><i class="fa-solid fa-sliders me-2 text-primary"></i>Campaign Settings</h5>
                
                <div class="mb-3">
                    <label class="form-label text-sub small">Campaign Budget ($)</label>
                    <input type="number" id="budget" class="form-control bg-dark text-light border-secondary" value="15000">
                </div>
                <div class="mb-3">
                    <label class="form-label text-sub small">Offer Success Rate (s)</label>
                    <input type="range" id="success_rate" class="form-range" min="0.05" max="0.70" step="0.05" value="0.30" oninput="document.getElementById('sr_val').innerText = this.value">
                    <span class="small text-sub">Value: <b id="sr_val">0.30</b></span>
                </div>
                <div class="mb-3">
                    <label class="form-label text-sub small">Offer Cost (% of Bill)</label>
                    <input type="range" id="offer_cost" class="form-range" min="0.05" max="0.40" step="0.05" value="0.15" oninput="document.getElementById('oc_val').innerText = this.value">
                    <span class="small text-sub">Value: <b id="oc_val">0.15</b></span>
                </div>
                <div class="mb-3">
                    <label class="form-label text-sub small">Customer ID Column</label>
                    <input type="text" id="id_col" class="form-control bg-dark text-light border-secondary" value="CustomerID">
                </div>
                <div class="mb-3">
                    <label class="form-label text-sub small">Target Column</label>
                    <input type="text" id="target_col" class="form-control bg-dark text-light border-secondary" value="Churn Label">
                </div>
            </div>
        </div>

        <!-- Main Workspace -->
        <div class="col-lg-8">
            <div class="card-custom">
                <h5 class="fw-bold mb-3"><i class="fa-solid fa-file-csv me-2 text-success"></i>Dataset Upload</h5>
                <form id="upload-form" enctype="multipart/form-data">
                    <input class="form-control bg-dark text-light border-secondary mb-3" type="file" id="file-input" accept=".csv, .xlsx, .xls">
                    
                    <div class="d-flex gap-2">
                        <button type="button" class="btn btn-accent flex-fill" onclick="runAction('predict')">
                            <i class="fa-solid fa-bolt me-2"></i>1. Instant Prediction
                        </button>
                        <button type="button" class="btn btn-teal flex-fill" onclick="runAction('train')">
                            <i class="fa-solid fa-brain me-2"></i>2. Full Train & Evaluate
                        </button>
                    </div>
                </form>
            </div>

            <!-- Results Section -->
            <div id="results-area" style="display: none;">
                <div class="card-custom">
                    <h5 class="fw-bold mb-3" id="result-title">Results Summary</h5>
                    <div class="row g-3 mb-4">
                        <div class="col-6 col-md-3">
                            <div class="kpi-card">
                                <span class="text-sub small">Targeted</span>
                                <h4 class="fw-bold mt-1 mb-0" id="kpi-targeted">0</h4>
                            </div>
                        </div>
                        <div class="col-6 col-md-3">
                            <div class="kpi-card">
                                <span class="text-sub small">Budget Spent</span>
                                <h4 class="fw-bold mt-1 mb-0" id="kpi-spent">$0</h4>
                            </div>
                        </div>
                        <div class="col-6 col-md-3">
                            <div class="kpi-card">
                                <span class="text-sub small">Net Benefit</span>
                                <h4 class="fw-bold mt-1 mb-0 text-success" id="kpi-profit">$0</h4>
                            </div>
                        </div>
                        <div class="col-6 col-md-3">
                            <div class="kpi-card">
                                <span class="text-sub small">Value Saved</span>
                                <h4 class="fw-bold mt-1 mb-0 text-info" id="kpi-saved">$0</h4>
                            </div>
                        </div>
                    </div>

                    <div id="table-container" class="table-responsive">
                        <!-- Decision table rendered here -->
                    </div>
                </div>
            </div>
        </div>
    </div>
</div>

<script>
async function runAction(mode) {
    const fileInput = document.getElementById('file-input');
    const budget = document.getElementById('budget').value;
    const successRate = document.getElementById('success_rate').value;
    const offerCost = document.getElementById('offer_cost').value;
    const idCol = document.getElementById('id_col').value;
    const targetCol = document.getElementById('target_col').value;

    const formData = new FormData();
    if (fileInput.files.length > 0) {
        formData.append('file', fileInput.files[0]);
    }
    formData.append('budget', budget);
    formData.append('success_rate', successRate);
    formData.append('offer_cost_pct', offerCost);
    formData.append('id_col', idCol);
    formData.append('target_col', targetCol);

    const endpoint = mode === 'predict' ? '/api/predict' : '/api/train';
    
    document.getElementById('results-area').style.display = 'block';
    document.getElementById('result-title').innerText = mode === 'predict' ? '⚡ Instant Prediction & Targeting Results' : '🏋️ Full Train, Evaluation & Optimization Results';
    document.getElementById('table-container').innerHTML = '<p class="text-sub">Processing dataset and running 0/1 Knapsack optimization...</p>';

    try {
        const response = await fetch(endpoint, {
            method: 'POST',
            body: formData
        });
        const data = await response.json();

        if (data.error) {
            alert('Error: ' + data.error);
            return;
        }

        document.getElementById('kpi-targeted').innerText = `${data.summary.total_targeted} / ${data.summary.total_customers}`;
        document.getElementById('kpi-spent').innerText = `$${Number(data.summary.total_cost_spent).toLocaleString(undefined, {minimumFractionDigits: 2})}`;
        document.getElementById('kpi-profit').innerText = `$${Number(data.summary.total_expected_net_benefit).toLocaleString(undefined, {minimumFractionDigits: 2})}`;
        document.getElementById('kpi-saved').innerText = `$${Number(data.summary.total_retained_value).toLocaleString(undefined, {minimumFractionDigits: 2})}`;

        // Build Table
        let tableHtml = '<table class="table table-custom table-hover mt-3"><thead><tr><th>Customer ID</th><th>Churn Risk</th><th>Customer Value</th><th>Offer Cost</th><th>Net Benefit</th><th>Target Decision</th></tr></thead><tbody>';
        data.sample_rows.forEach(row => {
            tableHtml += `<tr>
                <td><b>${row.id}</b></td>
                <td>${(row.prob * 100).toFixed(1)}%</td>
                <td>$${Number(row.V_i).toFixed(2)}</td>
                <td>$${Number(row.c_i).toFixed(2)}</td>
                <td class="text-success">$${Number(row.E_i).toFixed(2)}</td>
                <td><span class="badge bg-success">TARGET OFFER</span></td>
            </tr>`;
        });
        tableHtml += '</tbody></table>';
        document.getElementById('table-container').innerHTML = tableHtml;

    } catch (err) {
        alert('API Request Failed: ' + err);
    }
}
</script>
</body>
</html>
"""

@app.route('/', methods=['GET'])
def home():
    return render_template_string(HTML_TEMPLATE)

@app.route('/api/predict', methods=['POST'])
def api_predict():
    try:
        budget = float(request.form.get('budget', 15000.0))
        success_rate = float(request.form.get('success_rate', 0.30))
        offer_cost_pct = float(request.form.get('offer_cost_pct', 0.15))
        id_col = request.form.get('id_col', 'CustomerID')
        target_col = request.form.get('target_col', 'Churn Label')

        if 'file' in request.files and request.files['file'].filename != '':
            file = request.files['file']
            if file.filename.endswith('.csv'):
                df_raw = pd.read_csv(file)
            else:
                df_raw = pd.read_excel(file)
        else:
            sample_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'raw', 'telco_churn.csv')
            if not os.path.exists(sample_path):
                df_raw = generate_telco_churn_dataset()
            else:
                df_raw = pd.read_csv(sample_path)

        cleaner = DataCleaner(target_col=target_col, id_col=id_col)
        clean_df = cleaner.clean_data(df_raw)

        actual_id = id_col if id_col in clean_df.columns else ('customerID' if 'customerID' in clean_df.columns else clean_df.columns[0])
        ids = clean_df[actual_id] if actual_id in clean_df.columns else pd.Series(clean_df.index)
        X_features = clean_df.drop(columns=[c for c in [target_col, 'Churn', actual_id] if c in clean_df.columns])

        model_path = os.path.join(os.path.dirname(__file__), '..', 'outputs', 'trained_model.joblib')
        if not os.path.exists(model_path):
            trainer = ModelTrainer(target_col=target_col, id_col=id_col)
            X_tr, _, y_tr, _, _, _ = trainer.split_data(clean_df)
            preprocessor, _, _ = build_preprocessing_pipeline(clean_df, target_col=target_col, id_col=actual_id)
            model = trainer.train_logistic_regression(X_tr, y_tr, preprocessor)
            os.makedirs(os.path.dirname(model_path), exist_ok=True)
            joblib.dump(model, model_path)
        else:
            model = joblib.load(model_path)

        churn_probs = model.predict_proba(X_features)[:, 1]

        df_opt_input = X_features.copy()
        df_opt_input[actual_id] = ids.values

        optimizer = RetentionOptimizer(offer_success_rate=success_rate, offer_cost_pct=offer_cost_pct)
        df_results, summary = optimizer.optimize_retention(df_opt_input, churn_probs, budget=budget)

        log_campaign_to_supabase({
            "mode": "Instant Prediction (API)",
            "total_customers": len(clean_df),
            "budget": budget,
            "targeted_count": summary['total_targeted'],
            "budget_spent": summary['total_cost_spent'],
            "expected_net_benefit": summary['total_expected_net_benefit']
        })

        selected_candidates = df_results[df_results['selected_exact'] == 1].head(15)
        sample_rows = []
        for idx, row in selected_candidates.iterrows():
            sample_rows.append({
                "id": str(row[actual_id]),
                "prob": float(row['churn_prob']),
                "V_i": float(row['V_i']),
                "c_i": float(row['c_i']),
                "E_i": float(row['E_i'])
            })

        return jsonify({
            "summary": {
                "total_customers": len(clean_df),
                "total_targeted": summary['total_targeted'],
                "total_cost_spent": summary['total_cost_spent'],
                "total_expected_net_benefit": summary['total_expected_net_benefit'],
                "total_retained_value": summary['total_retained_value']
            },
            "sample_rows": sample_rows
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/train', methods=['POST'])
def api_train():
    return api_predict()

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
