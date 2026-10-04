import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    roc_auc_score, roc_curve, precision_recall_curve, average_precision_score,
    confusion_matrix, classification_report
)
from sklearn.calibration import calibration_curve

class ModelEvaluator:
    """
    Evaluates trained models on the locked test dataset.
    Generates ROC curves, PR curves, Confusion Matrix, Calibration plots, and Feature Coefficient plots.
    """
    def __init__(self, output_dir: str = os.path.join('outputs', 'plots')):
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

    def evaluate_model(self, model, X_test, y_test, model_name="Logistic Regression"):
        """
        Computes probability predictions and prints key metrics.
        """
        churn_probs = model.predict_proba(X_test)[:, 1]
        roc_auc = roc_auc_score(y_test, churn_probs)
        pr_auc = average_precision_score(y_test, churn_probs)

        print(f"=== Evaluation for {model_name} ===")
        print(f"ROC-AUC Score: {roc_auc:.4f}")
        print(f"PR-AUC Score:  {pr_auc:.4f}")
        
        preds_50 = (churn_probs >= 0.5).astype(int)
        cm = confusion_matrix(y_test, preds_50)
        print("Confusion Matrix (0.5 threshold):\n", cm)

        return churn_probs, roc_auc, pr_auc

    def plot_roc_pr_curves(self, y_test, churn_probs, model_name="Logistic Regression"):
        """
        Plots ROC Curve and Precision-Recall Curve side-by-side and saves to disk.
        """
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # ROC Curve
        fpr, tpr, _ = roc_curve(y_test, churn_probs)
        roc_auc = roc_auc_score(y_test, churn_probs)
        axes[0].plot(fpr, tpr, color='#2b5c8f', lw=2.5, label=f'ROC curve (AUC = {roc_auc:.3f})')
        axes[0].plot([0, 1], [0, 1], color='gray', linestyle='--')
        axes[0].set_xlabel('False Positive Rate')
        axes[0].set_ylabel('True Positive Rate')
        axes[0].set_title(f'ROC Curve - {model_name}')
        axes[0].legend(loc='lower right')
        axes[0].grid(True, alpha=0.3)

        # PR Curve
        precision, recall, _ = precision_recall_curve(y_test, churn_probs)
        pr_auc = average_precision_score(y_test, churn_probs)
        axes[1].plot(recall, precision, color='#d95f02', lw=2.5, label=f'PR curve (AUC = {pr_auc:.3f})')
        axes[1].set_xlabel('Recall')
        axes[1].set_ylabel('Precision')
        axes[1].set_title(f'Precision-Recall Curve - {model_name}')
        axes[1].legend(loc='lower left')
        axes[1].grid(True, alpha=0.3)

        plt.tight_layout()
        save_path = os.path.join(self.output_dir, 'roc_pr_curves.png')
        plt.savefig(save_path, dpi=300)
        plt.close()
        print(f"[ModelEvaluator] Saved ROC/PR curves to {save_path}")

    def plot_calibration_curve(self, y_test, churn_probs, model_name="Logistic Regression"):
        """
        Plots reliability calibration curve to ensure predicted probabilities match real probabilities.
        """
        prob_true, prob_pred = calibration_curve(y_test, churn_probs, n_bins=10)

        plt.figure(figsize=(7, 6))
        plt.plot(prob_pred, prob_true, marker='o', linewidth=2, color='#7570b3', label=model_name)
        plt.plot([0, 1], [0, 1], linestyle='--', color='gray', label='Perfectly Calibrated')
        plt.xlabel('Mean Predicted Probability')
        plt.ylabel('Fraction of Positives (True Churn Rate)')
        plt.title('Probability Calibration Curve')
        plt.legend(loc='upper left')
        plt.grid(True, alpha=0.3)

        save_path = os.path.join(self.output_dir, 'calibration_curve.png')
        plt.tight_layout()
        plt.savefig(save_path, dpi=300)
        plt.close()
        print(f"[ModelEvaluator] Saved calibration curve to {save_path}")

    def extract_and_plot_coefficients(self, pipeline_model):
        """
        Extracts feature names after ColumnTransformer preprocessing and plots top churn drivers.
        """
        preprocessor = pipeline_model.named_steps['prep']
        clf = pipeline_model.named_steps['clf']

        if not hasattr(clf, 'coef_'):
            print("[ModelEvaluator] Model does not have linear coefficients (e.g. non-linear model). Skipping coefficient plot.")
            return None

        # Extract feature names from preprocessor
        feature_names = []
        for name, trans, cols in preprocessor.transformers_:
            if name == 'num':
                feature_names.extend(cols)
            elif name == 'cat':
                # Extract one-hot encoded categories
                onehot = trans.named_steps['onehot']
                cat_names = onehot.get_feature_names_out(cols).tolist()
                feature_names.extend(cat_names)

        coefs = clf.coef_[0]
        coef_df = pd.DataFrame({'Feature': feature_names, 'Coefficient': coefs})
        coef_df['OddsRatio'] = np.exp(coef_df['Coefficient'])
        coef_df = coef_df.sort_values(by='Coefficient', ascending=False)

        # Plot Top 10 Positive and Top 10 Negative Drivers
        top_positive = coef_df.head(8)
        top_negative = coef_df.tail(8)
        plot_df = pd.concat([top_positive, top_negative])

        plt.figure(figsize=(10, 7))
        colors = ['#d95f02' if c > 0 else '#2b5c8f' for c in plot_df['Coefficient']]
        sns.barplot(x='Coefficient', y='Feature', data=plot_df, palette=colors)
        plt.title('Top Drivers of Customer Churn (Logistic Regression Coefficients)')
        plt.xlabel('Coefficient Value (Positive = Higher Churn Risk)')
        plt.grid(True, alpha=0.3)

        save_path = os.path.join(self.output_dir, 'feature_coefficients.png')
        plt.tight_layout()
        plt.savefig(save_path, dpi=300)
        plt.close()
        print(f"[ModelEvaluator] Saved feature coefficients plot to {save_path}")

        return coef_df
