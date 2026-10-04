import os
import numpy as np
import pandas as pd

def generate_telco_churn_dataset(n_samples=7043, seed=42):
    """
    Generates a realistic Telco Customer Churn dataset matching the IBM Telco sample dataset.
    This enables offline execution and self-contained testing for the churn prediction pipeline.
    """
    np.random.seed(seed)

    # Customer ID
    customer_ids = [f"{np.random.randint(1000, 9999)}-{np.random.choice(list('ABCDEFGHIJKLMNOPQRSTUVWXYZ'), 5)}" for _ in range(n_samples)]
    
    # Demographic features
    genders = np.random.choice(['Female', 'Male'], size=n_samples)
    senior_citizens = np.random.choice([0, 1], size=n_samples, p=[0.84, 0.16])
    partners = np.random.choice(['Yes', 'No'], size=n_samples, p=[0.48, 0.52])
    dependents = np.random.choice(['Yes', 'No'], size=n_samples, p=[0.30, 0.70])

    # Account info
    tenures = np.random.randint(0, 73, size=n_samples) # 0 to 72 months
    phone_services = np.random.choice(['Yes', 'No'], size=n_samples, p=[0.90, 0.10])

    multiple_lines = []
    for ps in phone_services:
        if ps == 'No':
            multiple_lines.append('No phone service')
        else:
            multiple_lines.append(np.random.choice(['Yes', 'No'], p=[0.45, 0.55]))

    internet_services = np.random.choice(['DSL', 'Fiber optic', 'No'], size=n_samples, p=[0.34, 0.44, 0.22])

    online_security = []
    online_backup = []
    device_protection = []
    tech_support = []
    streaming_tv = []
    streaming_movies = []

    for is_val in internet_services:
        if is_val == 'No':
            online_security.append('No internet service')
            online_backup.append('No internet service')
            device_protection.append('No internet service')
            tech_support.append('No internet service')
            streaming_tv.append('No internet service')
            streaming_movies.append('No internet service')
        else:
            online_security.append(np.random.choice(['Yes', 'No'], p=[0.35, 0.65]))
            online_backup.append(np.random.choice(['Yes', 'No'], p=[0.40, 0.60]))
            device_protection.append(np.random.choice(['Yes', 'No'], p=[0.40, 0.60]))
            tech_support.append(np.random.choice(['Yes', 'No'], p=[0.35, 0.65]))
            streaming_tv.append(np.random.choice(['Yes', 'No'], p=[0.50, 0.50]))
            streaming_movies.append(np.random.choice(['Yes', 'No'], p=[0.50, 0.50]))

    contracts = np.random.choice(['Month-to-month', 'One year', 'Two year'], size=n_samples, p=[0.55, 0.24, 0.21])
    paperless_billings = np.random.choice(['Yes', 'No'], size=n_samples, p=[0.59, 0.41])
    payment_methods = np.random.choice(
        ['Electronic check', 'Mailed check', 'Bank transfer (automatic)', 'Credit card (automatic)'],
        size=n_samples,
        p=[0.34, 0.23, 0.22, 0.21]
    )

    # Calculate monthly & total charges realistically
    monthly_charges = []
    total_charges = []

    for i in range(n_samples):
        base = 20.0
        if phone_services[i] == 'Yes':
            base += 15.0
        if multiple_lines[i] == 'Yes':
            base += 10.0
        if internet_services[i] == 'Fiber optic':
            base += 40.0
        elif internet_services[i] == 'DSL':
            base += 25.0
        if online_security[i] == 'Yes': base += 5.0
        if online_backup[i] == 'Yes': base += 5.0
        if device_protection[i] == 'Yes': base += 5.0
        if tech_support[i] == 'Yes': base += 5.0
        if streaming_tv[i] == 'Yes': base += 10.0
        if streaming_movies[i] == 'Yes': base += 10.0

        # Add small random noise
        monthly = round(base + np.random.normal(0, 2.0), 2)
        monthly = max(18.25, min(118.75, monthly))
        monthly_charges.append(monthly)

        if tenures[i] == 0:
            # Simulate blank strings for 0 tenure rows like in raw IBM dataset
            total_charges.append(" ")
        else:
            tot = round(monthly * tenures[i] + np.random.normal(0, 10.0), 2)
            total_charges.append(str(max(0.0, tot)))

    # Calculate churn probability based on features to mimic real signal
    churn_score = np.zeros(n_samples)
    for i in range(n_samples):
        score = 0.0
        if contracts[i] == 'Month-to-month': score += 1.2
        elif contracts[i] == 'Two year': score -= 1.5
        if internet_services[i] == 'Fiber optic': score += 0.8
        if tech_support[i] == 'No': score += 0.6
        if online_security[i] == 'No': score += 0.5
        if payment_methods[i] == 'Electronic check': score += 0.5
        if tenures[i] < 12: score += 0.9
        elif tenures[i] > 48: score -= 1.1
        if senior_citizens[i] == 1: score += 0.3
        
        churn_score[i] = score

    # Sigmoidal churn probability roughly ~26% positive rate
    probs = 1 / (1 + np.exp(-(churn_score - 1.2)))
    churn_labels = ['Yes' if p > np.random.rand() else 'No' for p in probs]

    df = pd.DataFrame({
        'customerID': customer_ids,
        'gender': genders,
        'SeniorCitizen': senior_citizens,
        'Partner': partners,
        'Dependents': dependents,
        'tenure': tenures,
        'PhoneService': phone_services,
        'MultipleLines': multiple_lines,
        'InternetService': internet_services,
        'OnlineSecurity': online_security,
        'OnlineBackup': online_backup,
        'DeviceProtection': device_protection,
        'TechSupport': tech_support,
        'StreamingTV': streaming_tv,
        'StreamingMovies': streaming_movies,
        'Contract': contracts,
        'PaperlessBilling': paperless_billings,
        'PaymentMethod': payment_methods,
        'MonthlyCharges': monthly_charges,
        'TotalCharges': total_charges,
        'Churn': churn_labels
    })

    return df

if __name__ == '__main__':
    raw_dir = os.path.join('data', 'raw')
    os.makedirs(raw_dir, exist_ok=True)
    out_path = os.path.join(raw_dir, 'telco_churn.csv')
    df = generate_telco_churn_dataset()
    df.to_csv(out_path, index=False)
    print(f"Dataset successfully created at {out_path} with {len(df)} rows and {len(df.columns)} columns.")
