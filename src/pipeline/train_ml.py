import joblib
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import accuracy_score, f1_score, precision_score, r2_score, recall_score, roc_auc_score, root_mean_squared_error
from sklearn.model_selection import KFold, StratifiedKFold

from src.core.config import settings
from src.core.logging import app_logger
from src.pipeline.preprocessor import ProjectDataPreprocessor


class ProjectMLEngine:
    """Machine learning lifecycle engine for training and persisting champion models."""

    def __init__(self):
        self.preprocessor = ProjectDataPreprocessor()
        self.classifiers = {
            "LogisticRegression": LogisticRegression(max_iter=1000, random_state=42),
            "RandomForest": RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42),
            "GradientBoosting": GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, random_state=42),
        }
        self.regressors = {
            "RidgeRegression": Ridge(alpha=1.0),
            "RandomForestRegressor": RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42),
        }

    def evaluate_classifiers(self, X, y) -> str:
        """Evaluate candidate classifiers and persist the best one."""
        app_logger.info("Starting classification model evaluation")
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        best_f1 = -1.0
        champion_name = None
        champion_model = None

        for name, model in self.classifiers.items():
            app_logger.info("Evaluating classifier %s", name)
            f1_scores, acc_scores, prec_scores, rec_scores, auc_scores = [], [], [], [], []
            for train_idx, val_idx in skf.split(X, y):
                X_train, X_val = X[train_idx], X[val_idx]
                y_train, y_val = y[train_idx], y[val_idx]
                model.fit(X_train, y_train)
                preds = model.predict(X_val)
                probs = model.predict_proba(X_val)[:, 1]
                f1_scores.append(f1_score(y_val, preds))
                acc_scores.append(accuracy_score(y_val, preds))
                prec_scores.append(precision_score(y_val, preds, zero_division=0))
                rec_scores.append(recall_score(y_val, preds))
                auc_scores.append(roc_auc_score(y_val, probs))

            mean_f1 = np.mean(f1_scores)
            app_logger.info(
                "[%s] Mean F1: %.4f | Accuracy: %.4f | Precision: %.4f | Recall: %.4f | ROC-AUC: %.4f",
                name,
                mean_f1,
                np.mean(acc_scores),
                np.mean(prec_scores),
                np.mean(rec_scores),
                np.mean(auc_scores),
            )

            if mean_f1 > best_f1:
                best_f1 = mean_f1
                champion_name = name
                champion_model = model

        classifier_path = settings.MODEL_DIR / "champion_classifier.joblib"
        champion_model.fit(X, y)
        joblib.dump(champion_model, classifier_path)
        app_logger.info("Champion classifier selected: %s (F1: %.4f)", champion_name, best_f1)
        return champion_name

    def evaluate_regressors(self, X, y) -> str:
        """Evaluate candidate regressors and persist the best one."""
        app_logger.info("Starting regression model evaluation")
        kf = KFold(n_splits=5, shuffle=True, random_state=42)
        best_r2 = -float("inf")
        champion_name = None
        champion_model = None

        for name, model in self.regressors.items():
            app_logger.info("Evaluating regressor %s", name)
            rmse_scores, r2_scores = [], []
            for train_idx, val_idx in kf.split(X, y):
                X_train, X_val = X[train_idx], X[val_idx]
                y_train, y_val = y[train_idx], y[val_idx]
                model.fit(X_train, y_train)
                preds = model.predict(X_val)
                rmse_scores.append(root_mean_squared_error(y_val, preds))
                r2_scores.append(r2_score(y_val, preds))

            mean_r2 = np.mean(r2_scores)
            app_logger.info("[%s] Mean R2: %.4f | RMSE: %.4f", name, mean_r2, np.mean(rmse_scores))
            if mean_r2 > best_r2:
                best_r2 = mean_r2
                champion_name = name
                champion_model = model

        regressor_path = settings.MODEL_DIR / "champion_regressor.joblib"
        champion_model.fit(X, y)
        joblib.dump(champion_model, regressor_path)
        app_logger.info("Champion regressor selected: %s (R2: %.4f)", champion_name, best_r2)
        return champion_name

    def execute_training_pipeline(self):
        """Run the complete training workflow and serialize model artifacts."""
        settings.MODEL_DIR.mkdir(parents=True, exist_ok=True)
        app_logger.info("Initializing machine learning training lifecycle")
        X_train, _, y_train_cls, _, y_train_reg, _, feature_cols = self.preprocessor.transform_and_split()
        joblib.dump(feature_cols, settings.MODEL_DIR / "feature_columns.joblib")
        joblib.dump(self.preprocessor.scaler, settings.MODEL_DIR / "fitted_scaler.joblib")
        self.evaluate_classifiers(X_train, y_train_cls)
        self.evaluate_regressors(X_train, y_train_reg)
        app_logger.info("Machine learning training lifecycle completed successfully")


if __name__ == "__main__":
    engine = ProjectMLEngine()
    engine.execute_training_pipeline()