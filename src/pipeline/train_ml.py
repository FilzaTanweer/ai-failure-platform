import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, root_mean_squared_error, r2_score
from sklearn.model_selection import StratifiedKFold, KFold
from src.core.config import settings
from src.core.logging import app_logger
from src.pipeline.preprocessor import ProjectDataPreprocessor

class ProjectMLEngine:
    """
    Enterprise Machine Learning Lifecycle Engine.
    Orchestrates cross-validation, model comparison, evaluation metrics profiling,
    and automatic serialization of champion artifacts.
    """
    def __init__(self):
        self.preprocessor = ProjectDataPreprocessor()
        
        # Initialize competitive candidate algorithms for comparison
        self.classifiers = {
            "LogisticRegression": LogisticRegression(max_iter=1000, random_state=42),
            "RandomForest": RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42),
            "GradientBoosting": GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, random_state=42)
        }
        
        self.regressors = {
            "RidgeRegression": Ridge(alpha=1.0),
            "RandomForestRegressor": RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42)
        }

    def evaluate_classifiers(self, X, y) -> str:
        """Runs Stratified 5-Fold Cross-Validation across candidate classifiers and saves the champion."""
        app_logger.info("Starting classification model evaluation loop...")
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        best_f1 = -1.0
        champion_name = None
        champion_model = None

        for name, model in self.classifiers.items():
            app_logger.info(f"Evaluating Classifier: {name} via 5-Fold CV...")
            f1_scores, acc_scores, prec_scores, rec_scores, auc_scores = [], [], [], [], []

            for fold, (train_idx, val_idx) in enumerate(skf.split(X, y), 1):
                X_train, X_val = X[train_idx], X[val_idx]
                y_train, y_val = y[train_idx], y[val_idx]

                # Train model copy clone
                model.fit(X_train, y_train)
                preds = model.predict(X_val)
                probs = model.predict_proba(X_val)[:, 1]

                # Append metrics
                f1_scores.append(f1_score(y_val, preds))
                acc_scores.append(accuracy_score(y_val, preds))
                prec_scores.append(precision_score(y_val, preds, zero_division=0))
                rec_scores.append(recall_score(y_val, preds))
                auc_scores.append(roc_auc_score(y_val, probs))

            mean_f1 = np.mean(f1_scores)
            app_logger.info(f"[{name}] Metrics -> Mean F1: {mean_f1:.4f} | Accuracy: {np.mean(acc_scores):.4f} | Precision: {np.mean(prec_scores):.4f} | Recall: {np.mean(rec_scores):.4f} | ROC-AUC: {np.mean(auc_scores):.4f}")

            if mean_f1 > best_f1:
                best_f1 = mean_f1
                champion_name = name
                champion_model = model

        # Persist Champion Classifier
        classifier_path = settings.MODEL_DIR / "champion_classifier.joblib"
        # Refit model onto complete dataset pool for optimal runtime utilization
        champion_model.fit(X, y)
        joblib.dump(champion_model, classifier_path)
        app_logger.info(f"🏆 Classifier Champion Selected: {champion_name} (F1: {best_f1:.4f}). Serialized to: {classifier_path}")
        return champion_name

    def evaluate_regressors(self, X, y) -> str:
        """Runs 5-Fold Cross-Validation across candidate regression models and saves the champion."""
        app_logger.info("Starting regression model evaluation loop...")
        kf = KFold(n_splits=5, shuffle=True, random_state=42)
        best_r2 = -float("inf")
        champion_name = None
        champion_model = None

        for name, model in self.regressors.items():
            app_logger.info(f"Evaluating Regressor: {name} via 5-Fold CV...")
            rmse_scores, r2_scores = [], []

            for fold, (train_idx, val_idx) in enumerate(kf.split(X, y), 1):
                X_train, X_val = X[train_idx], X[val_idx]
                y_train, y_val = y[train_idx], y[val_idx]

                model.fit(X_train, y_train)
                preds = model.predict(X_val)

                rmse_scores.append(root_mean_squared_error(y_val, preds))
                r2_scores.append(r2_score(y_val, preds))

            mean_r2 = np.mean(r2_scores)
            app_logger.info(f"[{name}] Metrics -> Mean R2: {mean_r2:.4f} | Mean RMSE: {np.mean(rmse_scores):.4f}")

            if mean_r2 > best_r2:
                best_r2 = mean_r2
                champion_name = name
                champion_model = model

        # Persist Champion Regressor
        regressor_path = settings.MODEL_DIR / "champion_regressor.joblib"
        champion_model.fit(X, y)
        joblib.dump(champion_model, regressor_path)
        app_logger.info(f"🏆 Regression Champion Selected: {champion_name} (R2: {best_r2:.4f}). Serialized to: {regressor_path}")
        return champion_name

    def execute_training_pipeline(self):
        """Orchestrates extraction, evaluation loops, and system persistence."""
        app_logger.info("Initializing comprehensive Machine Learning Training Lifecycle...")
        
        # Pull preprocessed partitions from Phase 5 pipeline directly
        X_train, X_test, y_train_cls, y_test_cls, y_train_reg, y_test_reg, feature_cols = self.preprocessor.transform_and_split()
        
        # Save feature column blueprint records alongside model weights
        joblib.dump(feature_cols, settings.MODEL_DIR / "feature_columns.joblib")
        joblib.dump(self.preprocessor.scaler, settings.MODEL_DIR / "fitted_scaler.joblib")
        
        # Run comparative loops
        self.evaluate_classifiers(X_train, y_train_cls)
        self.evaluate_regressors(X_train, y_train_reg)
        
        app_logger.info("Machine Learning Lifecycle pipeline executed successfully.")

if __name__ == "__main__":
    engine = ProjectMLEngine()
    engine.execute_training_pipeline()