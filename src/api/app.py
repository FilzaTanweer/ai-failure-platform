import joblib
import pandas as pd
from typing import List
from fastapi import FastAPI, HTTPException, Depends, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from pydantic import BaseModel, EmailStr, Field
from sqlalchemy.orm import Session
from jose import jwt, JWTError

from src.core.config import settings
from src.core.logging import app_logger
from src.database.session import get_db_context, init_db_schema
from src.database.models import Project, PredictionRecord, User
from src.auth.security import hash_password, verify_password, create_access_token, ALGORITHM

# 2.2.0 Version with explicit Auth Documentation
app = FastAPI(
    title="AI Project Failure Intelligence Platform API", 
    version="2.2.0",
    description="Enterprise Ingestion, Inference, and Secure Token Gating Engine."
)

app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/api/v1/auth/login")

class UserRegistrationSchema(BaseModel):
    email: EmailStr
    password: str = Field(..., min_length=8, example="P@ssword123!")
    role: str = Field(default="manager", example="manager")

class TokenResponseSchema(BaseModel):
    access_token: str
    token_type: str

class ProjectSnapshotMetrics(BaseModel):
    delayed_tasks: int = Field(..., ge=0)
    git_commits: int = Field(..., ge=0)
    open_bugs: int = Field(..., ge=0)
    high_priority_bugs: int = Field(..., ge=0)
    sprint_velocity: float
    developer_workload: float
    code_review_duration: float
    requirement_changes: int = Field(..., ge=0)
    meeting_attendance: float
    ci_cd_failures: int = Field(..., ge=0)
    testing_coverage: float
    pull_request_activity: int = Field(..., ge=0)

MODELS = {}

@app.on_event("startup")
def bootstrap_server_components():
    try:
        init_db_schema()
        MODELS["classifier"] = joblib.load(settings.MODEL_DIR / "champion_classifier.joblib")
        MODELS["regressor"] = joblib.load(settings.MODEL_DIR / "champion_regressor.joblib")
        MODELS["scaler"] = joblib.load(settings.MODEL_DIR / "fitted_scaler.joblib")
        MODELS["feature_columns"] = joblib.load(settings.MODEL_DIR / "feature_columns.joblib")
        app_logger.info("System configuration stack with Auth operational.")
    except Exception as e:
        raise RuntimeError(f"System boot sequence halted: {str(e)}")

def get_current_active_user(token: str = Depends(oauth2_scheme), db: Session = Depends(get_db_context)) -> User:
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate active secure authentication credentials.",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = jwt.decode(token, settings.SECRET_KEY, algorithms=[ALGORITHM])
        user_email: str = payload.get("sub")
        if user_email is None:
            raise credentials_exception
    except JWTError:
        raise credentials_exception
        
    user = db.query(User).filter(User.email == user_email).first()
    if user is None or not user.is_active:
        raise credentials_exception
    return user

def execute_pipeline_inference(df_input: pd.DataFrame) -> List[dict]:
    df_processed = df_input.copy()
    df_processed["workload_3wk_avg"] = df_processed["developer_workload"]
    df_processed["velocity_3wk_avg"] = df_processed["sprint_velocity"]
    df_processed["delayed_tasks_acceleration"] = 0.0
    df_processed["bug_growth_rate"] = 0.0
    df_processed["bug_severity_ratio"] = df_processed["high_priority_bugs"] / (df_processed["open_bugs"].replace(0, 1))
    df_processed["review_friction_index"] = df_processed["code_review_duration"] * df_processed["pull_request_activity"]
    ordered_matrix = df_processed[MODELS["feature_columns"]]
    scaled_matrix = MODELS["scaler"].transform(ordered_matrix)
    cls_preds = MODELS["classifier"].predict(scaled_matrix)
    cls_probs = MODELS["classifier"].predict_proba(scaled_matrix)
    reg_preds = MODELS["regressor"].predict(scaled_matrix)
    results = []
    for idx in range(len(df_input)):
        p_fail = int(cls_preds[idx])
        prob = float(cls_probs[idx][1] if p_fail == 1 else cls_probs[idx][0])
        risk_val = max(0.0, min(100.0, float(reg_preds[idx])))
        results.append({"is_failed": bool(p_fail), "prediction_label": "High Risk" if p_fail == 1 else "Stable", "confidence_score": round(prob, 4), "project_failure_risk_pct": round(risk_val, 2)})
    return results

@app.post("/api/v1/auth/register", status_code=201)
def register_tenant_identity(user_data: UserRegistrationSchema, db: Session = Depends(get_db_context)):
    existing_user = db.query(User).filter(User.email == user_data.email).first()
    if existing_user:
        raise HTTPException(status_code=400, detail="Account with this email already exists.")
        
    secured_user = User(
        email=user_data.email,
        hashed_password=hash_password(user_data.password),
        role=user_data.role
    )
    db.add(secured_user)
    db.commit()
    return {"status": "success", "message": f"Identity record {user_data.email} provisioned successfully."}

@app.post("/api/v1/auth/login", response_model=TokenResponseSchema)
def authenticate_user_session(form_data: OAuth2PasswordRequestForm = Depends(), db: Session = Depends(get_db_context)):
    user = db.query(User).filter(User.email == form_data.username).first()
    if not user or not verify_password(form_data.password, user.hashed_password):
        raise HTTPException(status_code=400, detail="Incorrect email username or password string parameters.")
        
    access_token = create_access_token(data={"sub": user.email, "role": user.role})
    return {"access_token": access_token, "token_type": "bearer"}

# 🛡️ Added explicit dependency for Swagger locks visualization mapping
@app.post("/api/v1/predict/snapshot", dependencies=[Depends(oauth2_scheme)])
def predict_single_snapshot(
    metrics: ProjectSnapshotMetrics, 
    db: Session = Depends(get_db_context),
    current_user: User = Depends(get_current_active_user)
):
    try:
        project_record = db.query(Project).filter(Project.name == f"Workspace of {current_user.email}").first()
        if not project_record:
            project_record = Project(name=f"Workspace of {current_user.email}", description="Dynamic account analytics stream context.")
            db.add(project_record)
            db.commit()
            db.refresh(project_record)

        input_df = pd.DataFrame([metrics.dict()])
        inference = execute_pipeline_inference(input_df)[0]
        
        persistent_ledger = PredictionRecord(
            project_id=project_record.id, delayed_tasks=metrics.delayed_tasks, git_commits=metrics.git_commits,
            open_bugs=metrics.open_bugs, high_priority_bugs=metrics.high_priority_bugs, sprint_velocity=metrics.sprint_velocity,
            developer_workload=metrics.developer_workload, code_review_duration=metrics.code_review_duration,
            requirement_changes=metrics.requirement_changes, meeting_attendance=metrics.meeting_attendance,
            ci_cd_failures=metrics.ci_cd_failures, testing_coverage=metrics.testing_coverage, pull_request_activity=metrics.pull_request_activity,
            is_failed=inference["is_failed"], prediction_label=inference["prediction_label"], confidence_score=inference["confidence_score"], project_failure_risk_pct=inference["project_failure_risk_pct"]
        )
        db.add(persistent_ledger)
        db.commit()
        return inference
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=str(e))