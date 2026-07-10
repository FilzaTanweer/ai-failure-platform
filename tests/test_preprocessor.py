import pandas as pd

from src.pipeline.preprocessor import ProjectDataPreprocessor


def test_prepare_for_prediction_adds_required_features() -> None:
    preprocessor = ProjectDataPreprocessor()
    raw_df = pd.DataFrame(
        [
            {
                "project_id": "PROJ_1",
                "delayed_tasks": 12,
                "git_commits": 120,
                "open_bugs": 8,
                "high_priority_bugs": 2,
                "sprint_velocity": 78.0,
                "developer_workload": 4.2,
                "code_review_duration": 10.0,
                "requirement_changes": 1,
                "meeting_attendance": 90.0,
                "ci_cd_failures": 0,
                "testing_coverage": 72.5,
                "pull_request_activity": 10,
            }
        ]
    )

    prepared = preprocessor.prepare_for_prediction(raw_df, feature_columns=[
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
    ])

    assert prepared.shape[0] == 1
    assert prepared.shape[1] == 18
    assert {"workload_3wk_avg", "bug_severity_ratio"}.issubset(prepared.columns)
