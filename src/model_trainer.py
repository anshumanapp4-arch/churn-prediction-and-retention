import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

from src.pipeline import build_preprocessing_pipeline

class ModelTrainer:
    """
    Handles dataset splitting, preprocessing pipeline integration, model training,
    cross-validation, and benchmarking.
    """
    def __init__(self, target_col: str = 'Churn', id_col: str = 'customerID', test_size: float = 0.20, random_state: int = 42):
        self.target_col = target_col
        self.id_col = id_col
        self.test_size = test_size
        self.random_state = random_state

    def split_data(self, df: pd.DataFrame):
        """
        Stratified train/test split.
        Returns X_train, X_test, y_train, y_test, ids_train, ids_test.
        """
        y = np.asarray(df[self.target_col].values)
        if self.id_col in df.columns:
            ids = np.asarray(df[self.id_col].values)
        else:
            ids = np.asarray(df.index.to_numpy())

        X = df.drop(columns=[c for c in [self.target_col, self.id_col] if c in df.columns]).copy().reset_index(drop=True)

        # Split using pure integer indices to guarantee compatibility with all Pandas/PyArrow/sklearn versions
        indices = np.arange(len(df))
        train_idx, test_idx = train_test_split(
            indices,
            test_size=self.test_size,
            stratify=y,
            random_state=self.random_state
        )

        X_train = X.iloc[train_idx].reset_index(drop=True)
        X_test = X.iloc[test_idx].reset_index(drop=True)
        y_train = y[train_idx]
        y_test = y[test_idx]
        ids_train = pd.Series(ids[train_idx])
        ids_test = pd.Series(ids[test_idx])

        return X_train, X_test, y_train, y_test, ids_train, ids_test

    def train_logistic_regression(self, X_train, y_train, preprocessor, tune_c=False):
        """
        Trains Logistic Regression model inside a scikit-learn pipeline.
        Optionally tunes hyperparameter C via GridSearchCV.
        """
        base_lr = LogisticRegression(max_iter=1000, class_weight='balanced', random_state=self.random_state)
        
        full_pipeline = Pipeline([
            ('prep', preprocessor),
            ('clf', base_lr)
        ])

        if tune_c:
            param_grid = {'clf__C': [0.01, 0.1, 1.0, 10.0]}
            grid_search = GridSearchCV(full_pipeline, param_grid, cv=5, scoring='roc_auc', n_jobs=-1)
            grid_search.fit(X_train, y_train)
            print(f"[ModelTrainer] Best Logistic Regression C parameter: {grid_search.best_params_['clf__C']}")
            return grid_search.best_estimator_
        else:
            full_pipeline.fit(X_train, y_train)
            cv_scores = cross_val_score(full_pipeline, X_train, y_train, cv=5, scoring='roc_auc')
            print(f"[ModelTrainer] 5-Fold Stratified Cross-Validation ROC-AUC: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")
            return full_pipeline

    def train_benchmark_models(self, X_train, y_train, preprocessor):
        """
        Trains Random Forest and Gradient Boosting benchmarks for comparison.
        """
        rf_pipeline = Pipeline([
            ('prep', preprocessor),
            ('clf', RandomForestClassifier(n_estimators=100, random_state=self.random_state, class_weight='balanced'))
        ])
        rf_pipeline.fit(X_train, y_train)

        gb_pipeline = Pipeline([
            ('prep', preprocessor),
            ('clf', GradientBoostingClassifier(n_estimators=100, random_state=self.random_state))
        ])
        gb_pipeline.fit(X_train, y_train)

        return {'RandomForest': rf_pipeline, 'GradientBoosting': gb_pipeline}
