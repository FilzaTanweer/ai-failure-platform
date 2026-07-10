from __future__ import annotations

import io
import json
from typing import Any

import joblib
import numpy as np
import pandas as pd
import shap
from fastapi import Depends, FastAPI, File, HTTPException, UploadFile, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from src.core.config import settings
from src.core.logging import app_logger
from src.database.models import PredictionRecord, Project
from src.database.session import get_db_context, init_db_schema, run_db_operation_with_retry
from src.pipeline.preprocessor import ProjectDataPreprocessor
from src.pipeline.train_ml import ProjectMLEngine

app = FastAPI(
    title="AI Project Failure Intelligence Platform API",
    version="2.3.0",
    description="Enterprise ingestion, inference, analytics, and reporting engine.",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class ProjectSnapshotMetrics(BaseModel):
    delayed_tasks: int = Field(..., ge=0)
    git_commits: int = Field(..., ge=0)
    open_bugs: int = Field(..., ge=0)
    high_priority_bugs: int = Field(..., ge=0)
    sprint_velocity: float = Field(..., ge=0.0)
    developer_workload: float = Field(..., ge=0.0)
    code_review_duration: float = Field(..., ge=0.0)
    requirement_changes: int = Field(..., ge=0)
    meeting_attendance: float = Field(..., ge=0.0, le=100.0)
    ci_cd_failures: int = Field(..., ge=0)
    testing_coverage: float = Field(..., ge=0.0, le=100.0)
    pull_request_activity: int = Field(..., ge=0)


class PredictionItem(BaseModel):
    row_index: int
    is_failed: bool
    prediction_label: str
    confidence_score: float
    project_failure_risk_pct: float
    explanation: str
    driver_summary: list[dict[str, Any]]


class UploadPredictionResponse(BaseModel):
    message: str
    total_rows: int
    high_risk_rows: int
    average_risk_pct: float
    average_confidence: float
    validation_summary: dict[str, Any]
    preview_rows: list[dict[str, Any]]
    results: list[PredictionItem]


MODELS: dict[str, Any] = {}
PREPROCESSOR = ProjectDataPreprocessor()


@app.on_event("startup")
def bootstrap_server_components() -> None:
    try:
        init_db_schema()
        ensure_model_artifacts()
        app_logger.info("Application boot completed")
    except Exception as exc:  # pragma: no cover - startup guard
        app_logger.exception("Startup initialization failed")
        raise RuntimeError(f"System boot sequence halted: {exc}") from exc


def ensure_model_artifacts() -> None:
    artifact_paths = [
        settings.MODEL_DIR / "champion_classifier.joblib",
        settings.MODEL_DIR / "champion_regressor.joblib",
        settings.MODEL_DIR / "feature_columns.joblib",
        settings.MODEL_DIR / "fitted_scaler.joblib",
    ]
    if not all(path.exists() for path in artifact_paths):
        app_logger.warning("Model artifacts missing; generating them now")
        ProjectMLEngine().execute_training_pipeline()

    MODELS["classifier"] = joblib.load(settings.MODEL_DIR / "champion_classifier.joblib")
    MODELS["regressor"] = joblib.load(settings.MODEL_DIR / "champion_regressor.joblib")
    MODELS["scaler"] = joblib.load(settings.MODEL_DIR / "fitted_scaler.joblib")
    MODELS["feature_columns"] = joblib.load(settings.MODEL_DIR / "feature_columns.joblib")


def build_explanation(model: Any, prepared_frame: pd.DataFrame, feature_columns: list[str], predicted_label: str) -> tuple[str, list[dict[str, Any]]]:
    try:
        if hasattr(model, "feature_importances_"):
            importances = np.abs(model.feature_importances_)
        elif hasattr(model, "coef_"):
            importances = np.abs(np.asarray(model.coef_).reshape(-1))
        else:
            importances = np.ones(len(feature_columns), dtype=float)

        ranked_features = sorted(zip(feature_columns, importances), key=lambda item: item[1], reverse=True)[:5]
        top_features = [
            {"feature": feature, "importance": round(float(importance), 4)}
            for feature, importance in ranked_features
        ]
        feature_names = ", ".join(item["feature"] for item in top_features)
        if predicted_label == "High Risk":
            explanation = f"The strongest signals were {feature_names}. The model is prioritizing delivery pressure, defect growth, and review friction."
        else:
            explanation = f"The strongest signals were {feature_names}. The portfolio appears stable with no major escalation signals."
        return explanation, top_features
    except Exception as exc:  # pragma: no cover - fallback guard
        app_logger.warning("Unable to compute feature importance: %s", exc)
        return "The model produced a stable risk assessment using the available operational indicators.", []


def score_frame(input_frame: pd.DataFrame) -> tuple[list[PredictionItem], dict[str, Any]]:
    if not MODELS:
        ensure_model_artifacts()

    feature_columns = list(MODELS["feature_columns"])
    prepared_frame = PREPROCESSOR.prepare_for_prediction(input_frame, feature_columns=feature_columns)
    scaled_matrix = MODELS["scaler"].transform(prepared_frame)
    classifier = MODELS["classifier"]
    regressor = MODELS["regressor"]

    labels = classifier.predict(scaled_matrix)
    probabilities = classifier.predict_proba(scaled_matrix)
    risk_scores = regressor.predict(scaled_matrix)

    results: list[PredictionItem] = []
    for index, (label, probability, risk_score) in enumerate(zip(labels, probabilities, risk_scores)):
        is_failed = bool(label)
        prediction_label = "High Risk" if is_failed else "Stable"
        confidence = float(probability[1] if is_failed else probability[0])
        risk_pct = float(np.clip(risk_score, 0.0, 100.0))
        explanation, driver_summary = build_explanation(classifier, prepared_frame, feature_columns, prediction_label)
        results.append(
            PredictionItem(
                row_index=index,
                is_failed=is_failed,
                prediction_label=prediction_label,
                confidence_score=round(confidence, 4),
                project_failure_risk_pct=round(risk_pct, 2),
                explanation=explanation,
                driver_summary=driver_summary,
            )
        )
    summary = {
        "total_rows": len(results),
        "high_risk_rows": sum(1 for item in results if item.is_failed),
        "average_risk_pct": round(float(np.mean([item.project_failure_risk_pct for item in results])), 2),
        "average_confidence": round(float(np.mean([item.confidence_score for item in results])), 4),
    }
    return results, summary


def get_or_create_project(db: Session) -> Project:
    def _lookup_project() -> Project:
        project_record = db.query(Project).filter(Project.name == "Operations Workspace").first()
        if project_record is None:
            project_record = Project(name="Operations Workspace", description="Primary working workspace for predictions")
            db.add(project_record)
            db.commit()
            db.refresh(project_record)
        return project_record

    return run_db_operation_with_retry(_lookup_project, max_retries=3, delay_seconds=0.2)


def persist_prediction_record(
    db: Session,
    metrics: dict[str, Any],
    result: PredictionItem,
    source_type: str = "snapshot",
    source_name: str | None = None,
    project_record: Project | None = None,
) -> None:
    project_record = project_record or get_or_create_project(db)

    record = PredictionRecord(
        project_id=project_record.id,
        delayed_tasks=int(metrics.get("delayed_tasks", 0)),
        git_commits=int(metrics.get("git_commits", 0)),
        open_bugs=int(metrics.get("open_bugs", 0)),
        high_priority_bugs=int(metrics.get("high_priority_bugs", 0)),
        sprint_velocity=float(metrics.get("sprint_velocity", 0.0)),
        developer_workload=float(metrics.get("developer_workload", 0.0)),
        code_review_duration=float(metrics.get("code_review_duration", 0.0)),
        requirement_changes=int(metrics.get("requirement_changes", 0)),
        meeting_attendance=float(metrics.get("meeting_attendance", 0.0)),
        ci_cd_failures=int(metrics.get("ci_cd_failures", 0)),
        testing_coverage=float(metrics.get("testing_coverage", 0.0)),
        pull_request_activity=int(metrics.get("pull_request_activity", 0)),
        is_failed=result.is_failed,
        prediction_label=result.prediction_label,
        confidence_score=result.confidence_score,
        project_failure_risk_pct=result.project_failure_risk_pct,
        explanation=result.explanation,
        feature_summary={item["feature"]: item["importance"] for item in result.driver_summary},
        source_type=source_type,
        source_name=source_name,
    )
    db.add(record)


@app.get("/api/v1/health")
def health_check() -> dict[str, Any]:
    return {"status": "ok", "app": settings.APP_NAME, "version": "2.3.0"}


@app.post("/api/v1/predict/snapshot")
def predict_single_snapshot(metrics: ProjectSnapshotMetrics, db: Session = Depends(get_db_context)):
    try:
        input_frame = pd.DataFrame([metrics.model_dump()])
        results, summary = score_frame(input_frame)
        for result in results:
            persist_prediction_record(db=db, metrics=metrics.model_dump(), result=result, source_type="snapshot")
        return {
            "message": "Prediction completed",
            "summary": summary,
            "prediction": results[0].model_dump(),
        }
    except Exception as exc:  # pragma: no cover - request guard
        db.rollback()
        app_logger.exception("Snapshot prediction failed")
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.post("/api/v1/predict/upload", response_model=UploadPredictionResponse)
async def predict_from_upload(file: UploadFile = File(...), db: Session = Depends(get_db_context)):
    if not file.filename:
        raise HTTPException(status_code=400, detail="Please attach a CSV or Excel file")

    file_bytes = await file.read()
    try:
        if file.filename.lower().endswith(".csv"):
            frame = pd.read_csv(io.BytesIO(file_bytes))
        elif file.filename.lower().endswith((".xlsx", ".xls")):
            frame = pd.read_excel(io.BytesIO(file_bytes))
        else:
            raise ValueError("Only CSV and Excel uploads are supported")
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Unable to parse uploaded file: {exc}") from exc

    validation_summary = {
        "row_count": int(frame.shape[0]),
        "column_count": int(frame.shape[1]),
        "missing_values": int(frame.isna().sum().sum()),
        "columns": list(frame.columns),
    }

    project_record = get_or_create_project(db)
    batch_size = 2500 if frame.shape[0] > 2500 else frame.shape[0]
    all_results: list[PredictionItem] = []
    for start_idx in range(0, frame.shape[0], batch_size):
        batch_frame = frame.iloc[start_idx : start_idx + batch_size].copy()
        batch_results, _ = score_frame(batch_frame)
        all_results.extend(batch_results)

        for index, result in enumerate(batch_results):
            metrics = {
                "delayed_tasks": int(batch_frame.iloc[index]["delayed_tasks"]) if "delayed_tasks" in batch_frame.columns else 0,
                "git_commits": int(batch_frame.iloc[index]["git_commits"]) if "git_commits" in batch_frame.columns else 0,
                "open_bugs": int(batch_frame.iloc[index]["open_bugs"]) if "open_bugs" in batch_frame.columns else 0,
                "high_priority_bugs": int(batch_frame.iloc[index]["high_priority_bugs"]) if "high_priority_bugs" in batch_frame.columns else 0,
                "sprint_velocity": float(batch_frame.iloc[index]["sprint_velocity"]) if "sprint_velocity" in batch_frame.columns else 0.0,
                "developer_workload": float(batch_frame.iloc[index]["developer_workload"]) if "developer_workload" in batch_frame.columns else 0.0,
                "code_review_duration": float(batch_frame.iloc[index]["code_review_duration"]) if "code_review_duration" in batch_frame.columns else 0.0,
                "requirement_changes": int(batch_frame.iloc[index]["requirement_changes"]) if "requirement_changes" in batch_frame.columns else 0,
                "meeting_attendance": float(batch_frame.iloc[index]["meeting_attendance"]) if "meeting_attendance" in batch_frame.columns else 0.0,
                "ci_cd_failures": int(batch_frame.iloc[index]["ci_cd_failures"]) if "ci_cd_failures" in batch_frame.columns else 0,
                "testing_coverage": float(batch_frame.iloc[index]["testing_coverage"]) if "testing_coverage" in frame.columns else 0.0,
                "pull_request_activity": int(batch_frame.iloc[index]["pull_request_activity"]) if "pull_request_activity" in batch_frame.columns else 0,
            }
            persist_prediction_record(
                db=db,
                metrics=metrics,
                result=result,
                source_type="upload",
                source_name=file.filename,
                project_record=project_record,
            )
        db.commit()

    summary = {
        "total_rows": len(all_results),
        "high_risk_rows": sum(1 for item in all_results if item.is_failed),
        "average_risk_pct": round(float(np.mean([item.project_failure_risk_pct for item in all_results])), 2),
        "average_confidence": round(float(np.mean([item.confidence_score for item in all_results])), 4),
    }
    return UploadPredictionResponse(
        message="Prediction completed for uploaded dataset",
        total_rows=summary["total_rows"],
        high_risk_rows=summary["high_risk_rows"],
        average_risk_pct=summary["average_risk_pct"],
        average_confidence=summary["average_confidence"],
        validation_summary=validation_summary,
        preview_rows=frame.head(10).to_dict(orient="records"),
        results=[item.model_dump() for item in all_results[:100]],
    )


@app.get("/api/v1/history")
def list_history(limit: int = 50, db: Session = Depends(get_db_context)):
    records = db.query(PredictionRecord).order_by(PredictionRecord.evaluated_at.desc()).limit(limit).all()
    return [
        {
            "id": record.id,
            "prediction_label": record.prediction_label,
            "risk_pct": record.project_failure_risk_pct,
            "confidence": record.confidence_score,
            "evaluated_at": record.evaluated_at.isoformat(),
            "source_type": record.source_type,
            "source_name": record.source_name,
        }
        for record in records
    ]


@app.get("/api/v1/analytics")
def analytics_summary(db: Session = Depends(get_db_context)):
    records = db.query(PredictionRecord).order_by(PredictionRecord.evaluated_at.asc()).all()
    if not records:
        return {"message": "No prediction history yet", "trend": []}

    trend = [
        {
            "date": record.evaluated_at.strftime("%Y-%m-%d"),
            "risk": round(record.project_failure_risk_pct, 2),
            "confidence": round(record.confidence_score, 4),
        }
        for record in records[-20:]
    ]
    return {
        "total_predictions": len(records),
        "high_risk_count": sum(1 for record in records if record.is_failed),
        "average_risk": round(float(sum(record.project_failure_risk_pct for record in records) / len(records)), 2),
        "trend": trend,
    }