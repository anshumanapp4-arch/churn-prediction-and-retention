import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

def build_preprocessing_pipeline(df: pd.DataFrame, target_col: str = 'Churn', id_col: str = 'customerID'):
    """
    Dynamically creates an sklearn ColumnTransformer preprocessing pipeline based on DataFrame column types.
    Works for any tabular churn dataset.
    
    Numeric features: Imputed with median + Scaled with StandardScaler.
    Categorical features: Imputed with most frequent + One-Hot Encoded.
    """
    feature_cols = [c for c in df.columns if c not in [target_col, id_col]]
    
    numeric_cols = df[feature_cols].select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = df[feature_cols].select_dtypes(include=['object', 'category', 'bool']).columns.tolist()

    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', drop='if_binary', sparse_output=False))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_cols),
            ('cat', categorical_transformer, categorical_cols)
        ],
        remainder='drop'
    )

    return preprocessor, numeric_cols, categorical_cols
