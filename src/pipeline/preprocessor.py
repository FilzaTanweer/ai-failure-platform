import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from src.core.config import settings
from src.core.logging import app_logger

class ProjectDataPreprocessor:
    """
    Enterprise Preprocessing & Feature Engineering Pipeline.
    Transforms raw project snapshot series into engineered feature matrices
    ready for predictive model training.
    """
    def __init__(self, raw_data_path: str = None):
        self.raw_data_path = raw_data_path or str(settings.DATA_DIR / "synthetic_project_data.csv")
        self.scaler = StandardScaler()
        self.burnout_encoder = LabelEncoder()
        
    def load_data(self) -> pd.DataFrame:
        """Loads the raw synthetic dataset from disk securely."""
        app_logger.info(f"Loading raw dataset from {self.raw_data_path}...")
        return pd.read_csv(self.raw_data_path)

    def engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Executes rolling temporal mutations and cross-feature engineering
        to generate deep signal matrices while avoiding historical lookahead bias.
        """
        app_logger.info("Executing enterprise feature engineering pipeline...")
        
        # Ensure correct sorting order across timespans before running shifts
        df = df.sort_values(by=["project_id", "week_index"]).reset_index(drop=True)
        
        # 1. Rolling Window Aggregations (Historical Trends)
        app_logger.debug("Computing rolling workload and velocity windows...")
        df["workload_3wk_avg"] = df.groupby("project_id")["developer_workload"].transform(lambda x: x.rolling(window=3, min_periods=1).mean())
        df["velocity_3wk_avg"] = df.groupby("project_id")["sprint_velocity"].transform(lambda x: x.rolling(window=3, min_periods=1).mean())
        
        # 2. Lagged Momentum Metrics (Rate of Change)
        app_logger.debug("Calculating metric velocity changes and bug momentum values...")
        df["delayed_tasks_lag1"] = df.groupby("project_id")["delayed_tasks"].shift(1).fillna(0)
        df["delayed_tasks_acceleration"] = df["delayed_tasks"] - df["delayed_tasks_lag1"]
        
        df["bugs_lag1"] = df.groupby("project_id")["open_bugs"].shift(1).fillna(0)
        df["bug_growth_rate"] = df["open_bugs"] - df["bugs_lag1"]
        
        # 3. Domain Cross-Ratios
        # Avoid division-by-zero errors safely by using a baseline floor operator
        df["bug_severity_ratio"] = df["high_priority_bugs"] / (df["open_bugs"].replace(0, 1))
        df["review_friction_index"] = df["code_review_duration"] * df["pull_request_activity"]
        
        return df

    def transform_and_split(self) -> tuple:
        """
        Applies mathematical scaling, transforms multi-class targets, 
        and partitions features into pristine Train/Test segments with class insulation.
        """
        # Load and augment data
        raw_df = self.load_data()
        engineered_df = self.engineer_features(raw_df)
        
        # Define operational features
        feature_cols = [
            "delayed_tasks", "git_commits", "open_bugs", "high_priority_bugs",
            "sprint_velocity", "developer_workload", "code_review_duration",
            "requirement_changes", "meeting_attendance", "ci_cd_failures",
            "testing_coverage", "pull_request_activity", "workload_3wk_avg",
            "velocity_3wk_avg", "delayed_tasks_acceleration", "bug_growth_rate",
            "bug_severity_ratio", "review_friction_index"
        ]
        
        X = engineered_df[feature_cols].copy()
        
        # Separate classification and regression validation targets
       # Separate classification and regression validation targets (Add .copy() here!)
        y_classification = engineered_df["is_failed"].values.copy()
        y_regression = engineered_df["project_failure_risk_pct"].values.copy()
        
        # --- 🛡️ CLASS BALANCE SAFEGUARD START ---
        # If the generated data is highly imbalanced or lacks a class, seed the edge cases
        if len(np.unique(y_classification)) < 2:
            app_logger.warning("Single class detected in targets. Activating edge-case signal injection...")
            # Set high-risk indexes (where risk pct is in the top 5%) to 1 to simulate failures
            high_risk_threshold = np.percentile(y_regression, 95)
            y_classification[y_regression >= high_risk_threshold] = 1
        # --- 🛡️ CLASS BALANCE SAFEGUARD END ---
        
        # Encode downstream categorical features gracefully
        app_logger.info("Normalizing continuous numerical feature distributions...")
        X_scaled = self.scaler.fit_transform(X)
        
        # Generate strict evaluation splits
        app_logger.info("Partitioning datasets into production train and validation matrices (80/20)...")
        X_train, X_test, y_train_cls, y_test_cls, y_train_reg, y_test_reg = train_test_split(
            X_scaled, y_classification, y_regression, test_size=0.2, random_state=42, stratify=y_classification
        )
        
        app_logger.info(f"Preprocessing completed. Train shape: {X_train.shape} | Test shape: {X_test.shape}")
        return X_train, X_test, y_train_cls, y_test_cls, y_train_reg, y_test_reg, feature_cols

if __name__ == "__main__":
    preprocessor = ProjectDataPreprocessor()
    preprocessor.transform_and_split()