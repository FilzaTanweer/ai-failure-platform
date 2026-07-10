from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from src.core.config import settings
from src.core.logging import app_logger


class ProjectDataPreprocessor:
    """Enterprise preprocessing and feature engineering for project risk modeling."""

    DEFAULT_FEATURE_COLUMNS = [
        "delayed_tasks",
        "git_commits",
        "open_bugs",
        "high_priority_bugs",
        "sprint_velocity",
        "developer_workload",
        "code_review_duration",
        "requirement_changes",
        "meeting_attendance",
        "ci_cd_failures",
        "testing_coverage",
        "pull_request_activity",
        "workload_3wk_avg",
        "velocity_3wk_avg",
        "delayed_tasks_acceleration",
        "bug_growth_rate",
        "bug_severity_ratio",
        "review_friction_index",
    ]

    COLUMN_ALIASES = {
        "delayed_tasks": ["delayed_tasks", "delayed tasks", "delayedtask", "late_tasks"],
        "git_commits": ["git_commits", "git commits", "commits", "weekly_git_commits"],
        "open_bugs": ["open_bugs", "open bugs", "active_bugs", "bug_count"],
        "high_priority_bugs": ["high_priority_bugs", "high priority bugs", "critical_bugs"],
        "sprint_velocity": ["sprint_velocity", "velocity", "sprint_velocity_index"],
        "developer_workload": ["developer_workload", "workload", "team_workload"],
        "code_review_duration": ["code_review_duration", "review_duration", "code_review_hours"],
        "requirement_changes": ["requirement_changes", "scope_changes", "change_requests"],
        "meeting_attendance": ["meeting_attendance", "attendance", "team_attendance"],
        "ci_cd_failures": ["ci_cd_failures", "cicd_failures", "pipeline_failures"],
        "testing_coverage": ["testing_coverage", "coverage", "unit_test_coverage"],
        "pull_request_activity": ["pull_request_activity", "pr_activity", "pull_requests"],
        "developer_burnout_status": ["developer_burnout_status", "burnout_status", "burnout"],
    }

    def __init__(self, raw_data_path: str | None = None):
        self.raw_data_path = raw_data_path or str(settings.BASE_DIR / "data" / "synthetic_project_data.csv")
        self.scaler = StandardScaler()

    def load_data(self) -> pd.DataFrame:
        """Load the synthetic training dataset from disk."""
        app_logger.info("Loading training dataset from %s", self.raw_data_path)
        return pd.read_csv(self.raw_data_path)

    def _normalize_column_names(self, df: pd.DataFrame) -> pd.DataFrame:
        normalized_df = df.copy()
        normalized_columns: list[str] = []
        for column in normalized_df.columns:
            normalized_name = str(column).strip().lower().replace(" ", "_")
            normalized_name = normalized_name.replace("-", "_")
            normalized_name = normalized_name.replace("(", "").replace(")", "")
            normalized_columns.append(normalized_name)
        normalized_df.columns = normalized_columns
        return normalized_df

    def _resolve_column(self, df: pd.DataFrame, canonical_name: str) -> str | None:
        normalized_df = self._normalize_column_names(df)
        for candidate in self.COLUMN_ALIASES.get(canonical_name, [canonical_name]):
            normalized_candidate = candidate.strip().lower().replace(" ", "_")
            for column in normalized_df.columns:
                if column == normalized_candidate:
                    return column
        return None

    def engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create derived features that are useful for failure prediction."""
        app_logger.info("Executing feature engineering pipeline")
        feature_frame = self._normalize_column_names(df).copy()
        if "project_id" not in feature_frame.columns:
            feature_frame["project_id"] = "PROJECT_1"
        if "week_index" not in feature_frame.columns:
            feature_frame["week_index"] = 0

        feature_frame = feature_frame.sort_values(by=["project_id", "week_index"]).reset_index(drop=True)
        if "developer_workload" in feature_frame.columns:
            feature_frame["workload_3wk_avg"] = feature_frame.groupby("project_id")["developer_workload"].transform(
                lambda values: values.rolling(window=3, min_periods=1).mean()
            )
        else:
            feature_frame["workload_3wk_avg"] = 0.0

        if "sprint_velocity" in feature_frame.columns:
            feature_frame["velocity_3wk_avg"] = feature_frame.groupby("project_id")["sprint_velocity"].transform(
                lambda values: values.rolling(window=3, min_periods=1).mean()
            )
        else:
            feature_frame["velocity_3wk_avg"] = 0.0

        if "delayed_tasks" in feature_frame.columns:
            feature_frame["delayed_tasks_lag1"] = feature_frame.groupby("project_id")["delayed_tasks"].shift(1).fillna(0)
            feature_frame["delayed_tasks_acceleration"] = feature_frame["delayed_tasks"] - feature_frame["delayed_tasks_lag1"]
        else:
            feature_frame["delayed_tasks_lag1"] = 0
            feature_frame["delayed_tasks_acceleration"] = 0.0

        if "open_bugs" in feature_frame.columns:
            feature_frame["bugs_lag1"] = feature_frame.groupby("project_id")["open_bugs"].shift(1).fillna(0)
            feature_frame["bug_growth_rate"] = feature_frame["open_bugs"] - feature_frame["bugs_lag1"]
        else:
            feature_frame["bugs_lag1"] = 0
            feature_frame["bug_growth_rate"] = 0.0

        if "high_priority_bugs" in feature_frame.columns and "open_bugs" in feature_frame.columns:
            feature_frame["bug_severity_ratio"] = feature_frame["high_priority_bugs"] / feature_frame["open_bugs"].replace(0, 1)
        else:
            feature_frame["bug_severity_ratio"] = 0.0

        if "code_review_duration" in feature_frame.columns and "pull_request_activity" in feature_frame.columns:
            feature_frame["review_friction_index"] = feature_frame["code_review_duration"] * feature_frame["pull_request_activity"]
        else:
            feature_frame["review_friction_index"] = 0.0

        return feature_frame

    def prepare_for_prediction(self, df: pd.DataFrame, feature_columns: Iterable[str] | None = None) -> pd.DataFrame:
        """Prepare a raw dataset for model inference by mapping columns and creating derived features."""
        preparation_frame = self.engineer_features(df)
        feature_columns = list(feature_columns or self.DEFAULT_FEATURE_COLUMNS)

        prepared = pd.DataFrame(index=preparation_frame.index)
        for feature_name in feature_columns:
            source_column = self._resolve_column(preparation_frame, feature_name)
            if source_column is None:
                prepared[feature_name] = 0.0
                continue
            prepared[feature_name] = pd.to_numeric(preparation_frame[source_column], errors="coerce")

        for feature_name in feature_columns:
            if feature_name in prepared.columns:
                prepared[feature_name] = pd.to_numeric(prepared[feature_name], errors="coerce")
                prepared[feature_name] = prepared[feature_name].fillna(prepared[feature_name].median())

        if "developer_burnout_status" in preparation_frame.columns:
            burnout_levels = {"low": 0.0, "medium": 0.5, "high": 1.0}
            burnout_series = preparation_frame["developer_burnout_status"].astype(str).str.lower()
            prepared["developer_burnout_status"] = burnout_series.map(burnout_levels).fillna(0.5)

        prepared = prepared.fillna(0.0)
        if "developer_burnout_status" not in prepared.columns:
            prepared["developer_burnout_status"] = 0.5
        return prepared[feature_columns]

    def transform_and_split(self) -> tuple:
        """Apply preprocessing and produce train/test splits for model training."""
        raw_df = self.load_data()
        engineered_df = self.engineer_features(raw_df)
        feature_cols = self.DEFAULT_FEATURE_COLUMNS

        X = engineered_df[feature_cols].copy()
        y_classification = engineered_df["is_failed"].values.copy()
        y_regression = engineered_df["project_failure_risk_pct"].values.copy()

        if len(np.unique(y_classification)) < 2:
            app_logger.warning("Single class detected in targets; injecting synthetic high-risk examples")
            high_risk_threshold = np.percentile(y_regression, 95)
            y_classification[y_regression >= high_risk_threshold] = 1

        X = X.fillna(0.0)
        X_scaled = self.scaler.fit_transform(X)
        X_train, X_test, y_train_cls, y_test_cls, y_train_reg, y_test_reg = train_test_split(
            X_scaled,
            y_classification,
            y_regression,
            test_size=0.2,
            random_state=42,
            stratify=y_classification,
        )
        app_logger.info("Preprocessing completed. Train shape: %s | Test shape: %s", X_train.shape, X_test.shape)
        return X_train, X_test, y_train_cls, y_test_cls, y_train_reg, y_test_reg, feature_cols


if __name__ == "__main__":
    preprocessor = ProjectDataPreprocessor()
    preprocessor.transform_and_split()