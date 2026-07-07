import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from src.core.config import settings
from src.core.logging import app_logger

# Define the incoming data structure for rigorous validation checks
class ProjectMetricsInput(BaseModel):
    delayed_tasks: int = Field(..., description="Number of delayed tasks currently active", ge=0)
    git_commits: int = Field(..., description="Total Git commits pushed this week", ge=0)
    open_bugs: int = Field(..., description="Total open bugs active", ge=0)
    high_priority_bugs: int = Field(..., description="Number of critical priority bugs", ge=0)
    sprint_velocity: float = Field(..., description="Current team sprint velocity sprint rate")
    developer_workload: float = Field(..., description="Calculated workload stress level index index scores")
    code_review_duration: float = Field(..., description="Average code review resolution windows in hours")
    requirement_changes: int = Field(..., description="Total scoping target updates introduced", ge=0)
    meeting_attendance: float = Field(..., description="Percentage of project meetings attended", ge=0.0, le=100.0)
    ci_cd_failures: int = Field(..., description="Count of pipeline validation breaks this week", ge=0)
    testing_coverage: float = Field(..., description="Unit code coverage percentage boundaries", ge=0.0, le=100.0)
    pull_request_activity: int = Field(..., description="Total PR adjustments reviewed", ge=0)

# Initialize production FastAPI server application instances
app = FastAPI(
    title="AI Project Failure Intelligence Platform API",
    description="Enterprise Machine Learning Engine providing automated project operational risk and failure predictions.",
    version="1.0.0"
)

# Global memory containers for persistent production binary states
MODELS = {}

@app.on_event("startup")
def load_production_artifacts():
    """Asynchronously loads all serialized machine learning components into memory on server boot."""
    try:
        app_logger.info("Initializing artifact hydration. Loading model layers into application memory...")
        
        MODELS["classifier"] = joblib.load(settings.MODEL_DIR / "champion_classifier.joblib")
        MODELS["regressor"] = joblib.load(settings.MODEL_DIR / "champion_regressor.joblib")
        MODELS["scaler"] = joblib.load(settings.MODEL_DIR / "fitted_scaler.joblib")
        MODELS["feature_columns"] = joblib.load(settings.MODEL_DIR / "feature_columns.joblib")
        
        app_logger.info("All core machine learning layers and scalers hydrated successfully. System online.")
    except Exception as e:
        app_logger.error(f"Critical System Failure: Failed to load binary components: {str(e)}")
        raise RuntimeError(f"Model hydration block failed: {str(e)}")

@app.get("/health")
def health_check():
    """System heartbeat endpoint used for container orchestration and uptime probes."""
    return {"status": "healthy", "hydrated_models": list(MODELS.keys())}

@app.post("/predict")
def predict_project_risk(metrics: ProjectMetricsInput):
    """
    Accepts raw operational parameters, triggers online feature engineering wrappers,
    and returns localized dual predictions back across classification and regression.
    """
    try:
        # Convert incoming telemetry dict straight into a structure dataframe frame
        input_data = pd.DataFrame([metrics.dict()])
        
        # --- Real-Time On-The-Fly Online Feature Engineering ---
        # Recreate the rolling baseline signals calculated in Phase 5
        input_data["workload_3wk_avg"] = input_data["developer_workload"]
        input_data["velocity_3wk_avg"] = input_data["sprint_velocity"]
        input_data["delayed_tasks_lag1"] = input_data["delayed_tasks"]
        input_data["delayed_tasks_acceleration"] = 0.0  # Zero baseline initialization for snapshot inference 
        input_data["bugs_lag1"] = input_data["open_bugs"]
        input_data["bug_growth_rate"] = 0.0
        
        # Compute cross ratios safely
        input_data["bug_severity_ratio"] = input_data["high_priority_bugs"] / (input_data["open_bugs"].replace(0, 1))
        input_data["review_friction_index"] = input_data["code_review_duration"] * input_data["pull_request_activity"]
        
        # Enforce tracking compliance with training column layout shapes
        ordered_features = input_data[MODELS["feature_columns"]]
        
        # Apply standard distribution normalization scaling transformations
        scaled_features = MODELS["scaler"].transform(ordered_features)
        
        # Run inferences simultaneously
        failure_prediction = int(MODELS["classifier"].predict(scaled_features)[0])
        failure_probabilities = MODELS["classifier"].predict_proba(scaled_features)[0]
        failure_confidence = float(failure_probabilities[1] if failure_prediction == 1 else failure_probabilities[0])
        
        calculated_risk_pct = float(MODELS["regressor"].predict(scaled_features)[0])
        # Floor boundaries strictly to sensible percentage bounds
        calculated_risk_pct = max(0.0, min(100.0, calculated_risk_pct))
        
        return {
            "prediction": {
                "is_failed": bool(failure_prediction),
                "prediction_label": "High Risk of Failure" if failure_prediction == 1 else "Stable / Low Risk",
                "confidence_score": round(failure_confidence, 4)
            },
            "risk_analysis": {
                "project_failure_risk_pct": round(calculated_risk_pct, 2)
            }
        }
        
    except Exception as e:
        app_logger.error(f"Prediction Pipeline Interrupted Exception: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Inference Engine Processing Error: {str(e)}")