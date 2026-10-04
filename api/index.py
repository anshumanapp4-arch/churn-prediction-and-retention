import os
import sys
import tempfile
import io
import json
import pandas as pd
import numpy as np
import joblib

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from flask import Flask, request, jsonify, render_template_string

from generate_sample_data import generate_telco_churn_dataset
from src.data_cleaner import DataCleaner
from src.pipeline import build_preprocessing_pipeline
from src.model_trainer import ModelTrainer
from src.business_optimizer import RetentionOptimizer
from src.supabase_client import upload_file_to_supabase, log_campaign_to_supabase

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max upload

# Custom JSON error handlers to guarantee all API failures return JSON, never raw HTML
@app.errorhandler(413)
def request_entity_too_large(error):
    return jsonify({"error": "Uploaded file is too large (Maximum 16MB allowed)."}), 413

@app.errorhandler(404)
def not_found(error):
    return jsonify({"error": "API route not found."}), 404

@app.errorhandler(500)
def internal_error(error):
    return jsonify({"error": f"Internal Server Error: {str(error)}"}), 500

@app.errorhandler(Exception)
def handle_unexpected_error(error):
    return jsonify({"error": f"Unexpected Error: {str(error)}"}), 500


HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Telecom Churn Prediction & Retention Optimization</title>
    <!-- Bootstrap 5 CSS -->
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css" rel="stylesheet">
    <!-- FontAwesome 6 -->
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css">
    <!-- Google Fonts -->
    <link href="https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&display=swap" rel="stylesheet">
    <!-- Chart.js -->
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>

    <style>
        :root {
            --bg-color: #090d16;
            --card-bg: #111827;
            --card-border: #1f293d;
            --accent-color: #6366f1;
            --accent-hover: #4f46e5;
            --emerald-color: #10b981;
            --cyan-color: #06b6d4;
            --amber-color: #f59e0b;
            --text-main: #f9fafb;
            --text-muted: #9ca3af;
        }
        body {
            background-color: var(--bg-color);
            color: var(--text-main);
            font-family: 'Plus Jakarta Sans', system-ui, -apple-system, sans-serif;
            min-height: 100vh;
            padding-bottom: 60px;
        }
        .navbar {
            background-color: rgba(17, 24, 39, 0.85);
            backdrop-filter: blur(12px);
            border-bottom: 1px solid var(--card-border);
            position: sticky;
            top: 0;
            z-index: 1000;
        }
        .brand-icon {
            background: linear-gradient(135deg, #6366f1, #a855f7);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }
        .badge-cloud {
            background: rgba(16, 185, 129, 0.12);
            color: #10b981;
            border: 1px solid rgba(16, 185, 129, 0.25);
            padding: 6px 14px;
            border-radius: 30px;
            font-size: 0.82rem;
            font-weight: 600;
            display: inline-flex;
            align-items: center;
            gap: 6px;
        }
        .pulse-dot {
            width: 8px;
            height: 8px;
            background-color: #10b981;
            border-radius: 50%;
            box-shadow: 0 0 10px #10b981;
            animation: pulse 2s infinite;
        }
        @keyframes pulse {
            0% { transform: scale(0.95); box-shadow: 0 0 0 0 rgba(16, 185, 129, 0.7); }
            70% { transform: scale(1); box-shadow: 0 0 0 8px rgba(16, 185, 129, 0); }
            100% { transform: scale(0.95); box-shadow: 0 0 0 0 rgba(16, 185, 129, 0); }
        }
        .card-custom {
            background-color: var(--card-bg);
            border: 1px solid var(--card-border);
            border-radius: 16px;
            padding: 24px;
            margin-bottom: 24px;
            box-shadow: 0 10px 30px rgba(0, 0, 0, 0.35);
            transition: border-color 0.2s;
        }
        .card-custom:hover {
            border-color: rgba(99, 102, 241, 0.4);
        }
        .kpi-card {
            background: linear-gradient(145deg, #161f33, #111827);
            border: 1px solid var(--card-border);
            border-radius: 12px;
            padding: 18px;
            position: relative;
            overflow: hidden;
        }
        .kpi-card::before {
            content: '';
            position: absolute;
            top: 0;
            left: 0;
            width: 4px;
            height: 100%;
        }
        .kpi-purple::before { background-color: var(--accent-color); }
        .kpi-emerald::before { background-color: var(--emerald-color); }
        .kpi-cyan::before { background-color: var(--cyan-color); }
        .kpi-amber::before { background-color: var(--amber-color); }

        .kpi-val {
            font-size: 1.6rem;
            font-weight: 700;
            margin-top: 6px;
            margin-bottom: 0;
        }
        .btn-accent {
            background: linear-gradient(135deg, #6366f1, #4f46e5);
            color: white;
            font-weight: 600;
            border: none;
            padding: 12px 20px;
            border-radius: 10px;
            box-shadow: 0 4px 14px rgba(99, 102, 241, 0.35);
            transition: all 0.2s;
        }
        .btn-accent:hover {
            background: linear-gradient(135deg, #4f46e5, #4338ca);
            color: white;
            transform: translateY(-1px);
        }
        .btn-emerald {
            background: linear-gradient(135deg, #10b981, #059669);
            color: white;
            font-weight: 600;
            border: none;
            padding: 12px 20px;
            border-radius: 10px;
            box-shadow: 0 4px 14px rgba(16, 185, 129, 0.3);
            transition: all 0.2s;
        }
        .btn-emerald:hover {
            background: linear-gradient(135deg, #059669, #047857);
            color: white;
            transform: translateY(-1px);
        }
        .table-custom {
            color: var(--text-main);
            vertical-align: middle;
            border-color: var(--card-border);
        }
        .table-custom th {
            background-color: #1a2336;
            color: var(--text-muted);
            font-size: 0.82rem;
            text-transform: uppercase;
            letter-spacing: 0.5px;
            border-bottom: 2px solid var(--card-border);
        }
        .table-custom td {
            background-color: transparent;
            border-color: var(--card-border);
            font-size: 0.9rem;
        }
        .form-control, .form-select {
            background-color: #0d121d !important;
            border-color: var(--card-border) !important;
            color: var(--text-main) !important;
            border-radius: 8px;
        }
        .form-control:focus, .form-select:focus {
            border-color: var(--accent-color) !important;
            box-shadow: 0 0 0 3px rgba(99, 102, 241, 0.25) !important;
        }
        .badge-target {
            background-color: rgba(16, 185, 129, 0.15);
            color: #34d399;
            border: 1px solid rgba(16, 185, 129, 0.3);
            font-weight: 600;
            padding: 5px 10px;
            border-radius: 6px;
        }
        .badge-skip {
            background-color: rgba(156, 163, 175, 0.1);
            color: #9ca3af;
            border: 1px solid rgba(156, 163, 175, 0.2);
            padding: 5px 10px;
            border-radius: 6px;
        }
    </style>
</head>
<body>

<nav class="navbar navbar-dark py-3 mb-4">
    <div class="container">
        <span class="navbar-brand mb-0 h1 fs-4 fw-bold">
            <i class="fa-solid fa-bolt brand-icon me-2"></i>Telecom Churn & Retention System
        </span>
        <span class="badge-cloud">
            <span class="pulse-dot"></span> Supabase & ML Active
        </span>
    </div>
</nav>

<div class="container">
    <div class="row">
        <!-- Control Sidebar -->
        <div class="col-lg-4">
            <div class="card-custom">
                <h6 class="fw-bold text-uppercase text-muted mb-3" style="letter-spacing: 0.5px; font-size: 0.8rem;">
                    <i class="fa-solid fa-sliders me-2 text-primary"></i>Financial Parameters
                </h6>

                <div class="mb-3">
                    <label class="form-label text-muted small fw-semibold">Campaign Budget ($)</label>
                    <input type="number" id="budget" class="form-control" value="15000" min="1000" step="1000">
                </div>
                <div class="mb-3">
                    <div class="d-flex justify-content-between">
                        <label class="form-label text-muted small fw-semibold">Offer Success Rate (s)</label>
                        <span class="small fw-bold text-primary" id="sr_val">0.30</span>
                    </div>
                    <input type="range" id="success_rate" class="form-range" min="0.05" max="0.70" step="0.05" value="0.30" oninput="document.getElementById('sr_val').innerText = this.value">
                </div>
                <div class="mb-3">
                    <div class="d-flex justify-content-between">
                        <label class="form-label text-muted small fw-semibold">Offer Cost (% of Bill)</label>
                        <span class="small fw-bold text-primary" id="oc_val">0.15</span>
                    </div>
                    <input type="range" id="offer_cost" class="form-range" min="0.05" max="0.40" step="0.05" value="0.15" oninput="document.getElementById('oc_val').innerText = this.value">
                </div>

                <hr class="my-3" style="border-color: var(--card-border);">

                <h6 class="fw-bold text-uppercase text-muted mb-3" style="letter-spacing: 0.5px; font-size: 0.8rem;">
                    <i class="fa-solid fa-table-columns me-2 text-info"></i>Column Mapping
                </h6>
                <div class="mb-3">
                    <label class="form-label text-muted small fw-semibold">Customer ID Column</label>
                    <input type="text" id="id_col" class="form-control" value="CustomerID">
                </div>
                <div class="mb-2">
                    <label class="form-label text-muted small fw-semibold">Target Column (Optional)</label>
                    <input type="text" id="target_col" class="form-control" value="Churn Label">
                </div>
            </div>
        </div>

        <!-- Main Workspace -->
        <div class="col-lg-8">
            <div class="card-custom">
                <h5 class="fw-bold mb-1"><i class="fa-solid fa-cloud-arrow-up me-2 text-success"></i>Dataset Execution</h5>
                <p class="text-muted small mb-3">Upload your customer dataset (.csv or .xlsx) or test directly using built-in sample data.</p>
                
                <form id="upload-form">
                    <div class="mb-3">
                        <input class="form-control" type="file" id="file-input" accept=".csv, .xlsx, .xls">
                    </div>
                    
                    <div class="row g-2">
                        <div class="col-md-6">
                            <button type="button" class="btn btn-accent w-100" onclick="runAction('predict')">
                                <i class="fa-solid fa-bolt me-2"></i>Instant Prediction
                            </button>
                        </div>
                        <div class="col-md-6">
                            <button type="button" class="btn btn-emerald w-100" onclick="runAction('train')">
                                <i class="fa-solid fa-brain me-2"></i>Full Train & Evaluate
                            </button>
                        </div>
                    </div>
                </form>
            </div>

            <!-- Results Section -->
            <div id="results-area" style="display: none;">
                <div class="card-custom">
                    <div class="d-flex flex-wrap justify-content-between align-items-center mb-4 gap-2">
                        <div>
                            <h5 class="fw-bold mb-0" id="result-title">Results Summary</h5>
                            <small class="text-muted" id="result-subtitle">0/1 Knapsack Optimization</small>
                        </div>
                        <button class="btn btn-sm btn-outline-light" onclick="downloadCSV()"><i class="fa-solid fa-download me-1"></i> Download CSV</button>
                    </div>

                    <!-- KPI Cards Grid -->
                    <div class="row g-3 mb-4">
                        <div class="col-6 col-md-3">
                            <div class="kpi-card kpi-purple">
                                <span class="text-muted small fw-semibold">Targeted</span>
                                <h4 class="kpi-val" id="kpi-targeted">0 / 0</h4>
                            </div>
                        </div>
                        <div class="col-6 col-md-3">
                            <div class="kpi-card kpi-cyan">
                                <span class="text-muted small fw-semibold">Budget Spent</span>
                                <h4 class="kpi-val" id="kpi-spent">$0</h4>
                            </div>
                        </div>
                        <div class="col-6 col-md-3">
                            <div class="kpi-card kpi-emerald">
                                <span class="text-muted small fw-semibold">Net Benefit</span>
                                <h4 class="kpi-val text-success" id="kpi-profit">$0</h4>
                            </div>
                        </div>
                        <div class="col-6 col-md-3">
                            <div class="kpi-card kpi-amber">
                                <span class="text-muted small fw-semibold">Value Saved</span>
                                <h4 class="kpi-val text-info" id="kpi-saved">$0</h4>
                            </div>
                        </div>
                    </div>

                    <!-- Filter Controls -->
                    <div class="row g-2 mb-3 align-items-center">
                        <div class="col-md-6">
                            <div class="input-group input-group-sm">
                                <span class="input-group-text bg-dark border-secondary text-muted"><i class="fa-solid fa-magnifying-glass"></i></span>
                                <input type="text" id="search-input" class="form-control" placeholder="Search Customer ID..." onkeyup="filterTable()">
                            </div>
                        </div>
                        <div class="col-md-6 text-end">
                            <span class="small text-muted me-2">Showing candidates:</span>
                            <span class="badge bg-secondary" id="table-count">0</span>
                        </div>
                    </div>

                    <div id="table-container" class="table-responsive">
                        <!-- Table populated dynamically -->
                    </div>
                </div>
            </div>
        </div>
    </div>
</div>

<script>
let lastResults = null;

async function runAction(mode) {
    const fileInput = document.getElementById('file-input');
    const budget = document.getElementById('budget').value;
    const successRate = document.getElementById('success_rate').value;
    const offerCost = document.getElementById('offer_cost').value;
    const idCol = document.getElementById('id_col').value;
    const targetCol = document.getElementById('target_col').value;

    const formData = new FormData();
    formData.append('budget', budget);
    formData.append('success_rate', successRate);
    formData.append('offer_cost_pct', offerCost);
    formData.append('id_col', idCol);
    formData.append('target_col', targetCol);

    if (fileInput.files.length > 0) {
        const file = fileInput.files[0];
        // Read file client-side to prevent Vercel 4.5MB request payload 413 error
        if (file.name.toLowerCase().endsWith('.csv') || file.size > 2 * 1024 * 1024) {
            try {
                const text = await file.text();
                // If larger than 2MB, sample top ~10,000 lines safely
                let cleanText = text;
                if (text.length > 2 * 1024 * 1024) {
                    const lines = text.split('\n');
                    cleanText = lines.slice(0, 8000).join('\n');
                }
                formData.append('csv_data', cleanText);
            } catch (readErr) {
                formData.append('file', file);
            }
        } else {
            formData.append('file', file);
        }
    }

    const endpoint = mode === 'predict' ? '/api/predict' : '/api/train';
    
    document.getElementById('results-area').style.display = 'block';
    document.getElementById('result-title').innerText = mode === 'predict' ? '⚡ Instant Prediction & Optimization' : '🏋️ Model Training & Optimization Results';
    document.getElementById('table-container').innerHTML = `
        <div class="text-center py-5">
            <div class="spinner-border text-primary" role="status" style="width: 3rem; height: 3rem;"></div>
            <p class="mt-3 text-muted fw-semibold">Running Machine Learning Model & 0/1 Knapsack Optimizer...</p>
        </div>`;

    try {
        const response = await fetch(endpoint, {
            method: 'POST',
            body: formData
        });

        // Safely capture raw text first to avoid SyntaxError on non-JSON response
        const responseText = await response.text();
        let data;
        try {
            data = JSON.parse(responseText);
        } catch (jsonErr) {
            throw new Error(`Server returned invalid non-JSON response (${response.status}): ${responseText.substring(0, 300)}`);
        }

        if (!response.ok || data.error) {
            const errorMsg = data.error || `Server Error ${response.status}`;
            document.getElementById('table-container').innerHTML = `
                <div class="alert alert-danger shadow-sm">
                    <i class="fa-solid fa-circle-exclamation me-2"></i><b>Execution Failed:</b> ${errorMsg}
                </div>`;
            return;
        }

        lastResults = data;

        document.getElementById('kpi-targeted').innerText = `${data.summary.total_targeted} / ${data.summary.total_customers}`;
        document.getElementById('kpi-spent').innerText = `$${Number(data.summary.total_cost_spent).toLocaleString(undefined, {minimumFractionDigits: 2})}`;
        document.getElementById('kpi-profit').innerText = `$${Number(data.summary.total_expected_net_benefit).toLocaleString(undefined, {minimumFractionDigits: 2})}`;
        document.getElementById('kpi-saved').innerText = `$${Number(data.summary.total_retained_value).toLocaleString(undefined, {minimumFractionDigits: 2})}`;

        renderTable(data.sample_rows);

    } catch (err) {
        document.getElementById('table-container').innerHTML = `
            <div class="alert alert-danger shadow-sm">
                <i class="fa-solid fa-triangle-exclamation me-2"></i><b>API Error:</b> ${err.message}
            </div>`;
    }
}

function renderTable(rows) {
    if (!rows || rows.length === 0) {
        document.getElementById('table-container').innerHTML = '<div class="text-center py-4 text-muted">No candidates selected for retention targeting under current budget constraint.</div>';
        document.getElementById('table-count').innerText = "0";
        return;
    }

    document.getElementById('table-count').innerText = rows.length;

    let tableHtml = `
        <table class="table table-custom table-hover mt-2" id="results-table">
            <thead>
                <tr>
                    <th>Customer ID</th>
                    <th>Churn Risk</th>
                    <th>Annual Value</th>
                    <th>Offer Cost</th>
                    <th>Expected Net Benefit</th>
                    <th>Decision</th>
                </tr>
            </thead>
            <tbody>`;

    rows.forEach(row => {
        tableHtml += `
            <tr>
                <td class="fw-bold">${row.id}</td>
                <td>
                    <div class="d-flex align-items-center gap-2">
                        <div class="progress flex-fill" style="height: 6px; background-color: #1e293b;">
                            <div class="progress-bar ${row.prob > 0.5 ? 'bg-danger' : 'bg-warning'}" style="width: ${(row.prob * 100).toFixed(0)}%"></div>
                        </div>
                        <span class="small fw-semibold">${(row.prob * 100).toFixed(1)}%</span>
                    </div>
                </td>
                <td>$${Number(row.V_i).toLocaleString(undefined, {minimumFractionDigits: 2})}</td>
                <td>$${Number(row.c_i).toLocaleString(undefined, {minimumFractionDigits: 2})}</td>
                <td class="text-success fw-bold">+$${Number(row.E_i).toLocaleString(undefined, {minimumFractionDigits: 2})}</td>
                <td><span class="badge badge-target"><i class="fa-solid fa-check me-1"></i> TARGET OFFER</span></td>
            </tr>`;
    });

    tableHtml += '</tbody></table>';
    document.getElementById('table-container').innerHTML = tableHtml;
}

function filterTable() {
    const query = document.getElementById('search-input').value.toLowerCase();
    if (!lastResults || !lastResults.sample_rows) return;
    const filtered = lastResults.sample_rows.filter(r => String(r.id).toLowerCase().includes(query));
    renderTable(filtered);
}

function downloadCSV() {
    if (!lastResults || !lastResults.sample_rows) {
        alert("No retention targeting results available to download.");
        return;
    }
    let csv = "CustomerID,ChurnProbability,CustomerValue,OfferCost,ExpectedNetBenefit,Decision\\n";
    lastResults.sample_rows.forEach(r => {
        csv += `"${r.id}",${r.prob},${r.V_i},${r.c_i},${r.E_i},"TARGET_OFFER"\\n`;
    });
    const blob = new Blob([csv], { type: 'text/csv' });
    const url = window.URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = "retention_target_list.csv";
    a.click();
}
</script>
</body>
</html>
"""

@app.route('/', methods=['GET'])
def home():
    return render_template_string(HTML_TEMPLATE)

def get_model():
    """
    Safely loads or trains the ML model pipeline. Handles read-only Vercel environment.
    """
    model_paths = [
        os.path.join(os.path.dirname(__file__), 'trained_model.joblib'),
        os.path.join(os.path.dirname(__file__), '..', 'outputs', 'trained_model.joblib'),
        os.path.join(tempfile.gettempdir(), 'trained_model.joblib')
    ]
    
    for mp in model_paths:
        if os.path.exists(mp):
            try:
                model = joblib.load(mp)
                print(f"[ModelLoad] Loaded model successfully from {mp}")
                return model
            except Exception as e:
                print(f"[ModelLoad] Warning loading {mp}: {e}")

    # If no model found, train on light sample dataset
    print("[ModelLoad] Training fallback Logistic Regression model on sample data...")
    sample_df = generate_telco_churn_dataset(n_samples=1000)
    cleaner = DataCleaner(target_col='Churn Label', id_col='CustomerID')
    clean_sample = cleaner.clean_data(sample_df)

    trainer = ModelTrainer(target_col='Churn Label', id_col='CustomerID')
    preprocessor, _, _ = build_preprocessing_pipeline(clean_sample, target_col='Churn Label', id_col='CustomerID')
    X_tr, _, y_tr, _, _, _ = trainer.split_data(clean_sample)
    model = trainer.train_logistic_regression(X_tr, y_tr, preprocessor, tune_c=False)

    # Attempt to save model to temp directory
    try:
        tmp_model_path = os.path.join(tempfile.gettempdir(), 'trained_model.joblib')
        joblib.dump(model, tmp_model_path)
    except Exception as e:
        print(f"[ModelSave] Could not save to tmp: {e}")

    return model

@app.route('/api/predict', methods=['GET', 'POST'])
def api_predict():
    try:
        if request.method == 'GET':
            budget = float(request.args.get('budget', 15000.0))
            success_rate = float(request.args.get('success_rate', 0.30))
            offer_cost_pct = float(request.args.get('offer_cost_pct', 0.15))
            id_col = request.args.get('id_col', 'CustomerID').strip()
            target_col = request.args.get('target_col', 'Churn Label').strip()
        else:
            budget = float(request.form.get('budget', 15000.0))
            success_rate = float(request.form.get('success_rate', 0.30))
            offer_cost_pct = float(request.form.get('offer_cost_pct', 0.15))
            id_col = request.form.get('id_col', 'CustomerID').strip()
            target_col = request.form.get('target_col', 'Churn Label').strip()

        # 1. Load Data
        csv_data = request.form.get('csv_data') if request.method == 'POST' else None
        if request.method == 'POST' and csv_data:
            df_raw = pd.read_csv(io.StringIO(csv_data))
        elif request.method == 'POST' and 'file' in request.files and request.files['file'].filename != '':
            file = request.files['file']
            filename = file.filename.lower()
            if filename.endswith('.csv'):
                df_raw = pd.read_csv(file)
            elif filename.endswith(('.xlsx', '.xls')):
                df_raw = pd.read_excel(file)
            else:
                return jsonify({"error": "Unsupported file format. Please upload a .csv or .xlsx file."}), 400
        else:
            sample_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'raw', 'telco_churn.csv')
            if os.path.exists(sample_path):
                df_raw = pd.read_csv(sample_path)
            else:
                df_raw = generate_telco_churn_dataset(n_samples=1000)

        if df_raw.empty:
            return jsonify({"error": "The uploaded dataset is empty."}), 400

        # 2. Clean Data
        cleaner = DataCleaner(target_col=target_col, id_col=id_col)
        clean_df = cleaner.clean_data(df_raw)

        actual_id = id_col if id_col in clean_df.columns else ('customerID' if 'customerID' in clean_df.columns else ('CustomerID' if 'CustomerID' in clean_df.columns else clean_df.columns[0]))
        ids = clean_df[actual_id] if actual_id in clean_df.columns else pd.Series(clean_df.index)

        drop_cols = [c for c in [target_col, 'Churn', 'Churn Label', actual_id] if c in clean_df.columns]
        X_features = clean_df.drop(columns=drop_cols)

        # Feature alignment to ensure compatibility with trained model on any CSV
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

        # 3. Model Prediction
        model = get_model()
        churn_probs = model.predict_proba(X_aligned)[:, 1]

        # 4. 0/1 Knapsack Optimization
        df_opt_input = X_aligned.copy()
        df_opt_input[actual_id] = ids.values

        optimizer = RetentionOptimizer(offer_success_rate=success_rate, offer_cost_pct=offer_cost_pct)
        df_results, summary = optimizer.optimize_retention(df_opt_input, churn_probs, budget=budget)

        # 5. Log to Supabase (non-blocking safety)
        try:
            log_campaign_to_supabase({
                "mode": f"Prediction ({request.method})",
                "total_customers": int(len(clean_df)),
                "budget": float(budget),
                "targeted_count": int(summary['total_targeted']),
                "budget_spent": float(summary['total_cost_spent']),
                "expected_net_benefit": float(summary['total_expected_net_benefit'])
            })
        except Exception as se:
            print(f"[Supabase] Notice logging campaign: {se}")

        # 6. Format JSON Output (capped at top 35 selected candidates for lightweight payload)
        selected_candidates = df_results[df_results['selected_exact'] == 1].head(35)
        sample_rows = []
        for _, row in selected_candidates.iterrows():
            sample_rows.append({
                "id": str(row[actual_id]),
                "prob": round(float(row['churn_prob']), 4),
                "V_i": round(float(row['V_i']), 2),
                "c_i": round(float(row['c_i']), 2),
                "E_i": round(float(row['E_i']), 2)
            })

        return jsonify({
            "status": "success",
            "summary": {
                "total_customers": int(len(clean_df)),
                "total_targeted": int(summary['total_targeted']),
                "total_cost_spent": float(summary['total_cost_spent']),
                "total_expected_net_benefit": float(summary['total_expected_net_benefit']),
                "total_retained_value": float(summary['total_retained_value'])
            },
            "sample_rows": sample_rows
        }), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/train', methods=['GET', 'POST'])
def api_train():
    return api_predict()

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
