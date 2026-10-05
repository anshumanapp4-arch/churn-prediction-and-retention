import os
import pandas as pd
import numpy as np

class DataCleaner:
    """
    Robust data cleaner for Churn Datasets.
    Designed to work automatically on the Telco Customer Churn dataset or any generic CSV dataset.
    """
    def __init__(self, target_col='Churn', id_col='customerID'):
        self.target_col = target_col
        self.id_col = id_col

    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Cleans raw DataFrame by fixing data types, handling missing values,
        standardizing labels, dropping leakage columns, and encoding target.
        """
        df = df.copy()

        # Rename standard columns for consistency if present
        column_mapping = {
            'Gender': 'gender',
            'Tenure Months': 'tenure',
            'Monthly Charges': 'MonthlyCharges',
            'Total Charges': 'TotalCharges',
            'Senior Citizen': 'SeniorCitizen',
            'Phone Service': 'PhoneService',
            'Multiple Lines': 'MultipleLines',
            'Internet Service': 'InternetService',
            'Online Security': 'OnlineSecurity',
            'Online Backup': 'OnlineBackup',
            'Device Protection': 'DeviceProtection',
            'Tech Support': 'TechSupport',
            'Streaming TV': 'StreamingTV',
            'Streaming Movies': 'StreamingMovies',
            'Paperless Billing': 'PaperlessBilling',
            'Payment Method': 'PaymentMethod'
        }
        df = df.rename(columns={k: v for k, v in column_mapping.items() if k in df.columns})

        # Drop potential target leakage & location columns if present
        leakage_cols = ['Churn Reason', 'Churn Score', 'Count', 'Country', 'State', 'City', 'Zip Code', 'Lat Long', 'Latitude', 'Longitude']
        if self.target_col in ['Churn Label', 'Churn Value', 'Churn']:
            for lc in ['Churn Label', 'Churn Value', 'Churn']:
                if lc != self.target_col and lc in df.columns:
                    leakage_cols.append(lc)
        
        cols_to_drop = [c for c in leakage_cols if c in df.columns]
        if cols_to_drop:
            df = df.drop(columns=cols_to_drop)

        # 1. Remove duplicate rows if any
        n_dups = df.duplicated().sum()
        if n_dups > 0:
            df = df.drop_duplicates().reset_index(drop=True)

        # 2. Fix known numeric columns that might load as text
        for col in df.columns:
            if col in ['TotalCharges', 'total_charges', 'totalcharges', 'MonthlyCharges', 'tenure']:
                df[col] = pd.to_numeric(df[col], errors='coerce')

        # 3. Handle missing values for numeric columns (e.g., tenure 0 rows)
        if 'tenure' in df.columns:
            if 'TotalCharges' in df.columns:
                df.loc[df['tenure'] == 0, 'TotalCharges'] = 0.0
                df['TotalCharges'] = df['TotalCharges'].fillna(df['TotalCharges'].median())

        # Fill any other numeric NaNs with median
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if df[col].isna().sum() > 0:
                df[col] = df[col].fillna(df[col].median())

        # 4. Standardize text labels
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns
        for col in categorical_cols:
            if col not in [self.target_col, self.id_col]:
                df[col] = df[col].astype(str).str.strip()
                df[col] = df[col].replace({'No internet service': 'No', 'No phone service': 'No'})

        # 5. Encode Target Column into Binary 1/0
        if self.target_col in df.columns:
            s = df[self.target_col]
            if pd.api.types.is_numeric_dtype(s):
                df[self.target_col] = (s > 0).astype(int)
            else:
                str_s = s.astype(str).str.strip().str.lower()
                mapping = {
                    '1': 1, '1.0': 1, 'yes': 1, 'true': 1, 'churned': 1, 'churn': 1, 'leave': 1, 'left': 1, 'exited': 1, 'positive': 1,
                    '0': 0, '0.0': 0, 'no': 0, 'false': 0, 'retained': 0, 'stayed': 0, 'keep': 0, 'stay': 0, 'negative': 0
                }
                mapped = str_s.map(mapping)
                if mapped.isna().sum() > 0:
                    mapped = mapped.fillna(pd.to_numeric(s, errors='coerce')).fillna(0)
                df[self.target_col] = mapped.astype(int)

        return df

    def save_processed(self, df: pd.DataFrame, output_path: str):
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        df.to_csv(output_path, index=False)
        print(f"[DataCleaner] Processed data saved successfully to {output_path}")

if __name__ == '__main__':
    raw_file = os.path.join('data', 'raw', 'telco_churn.csv')
    if os.path.exists(raw_file):
        df_raw = pd.read_csv(raw_file)
        cleaner = DataCleaner()
        df_clean = cleaner.clean_data(df_raw)
        cleaner.save_processed(df_clean, os.path.join('data', 'processed', 'processed_churn.csv'))
