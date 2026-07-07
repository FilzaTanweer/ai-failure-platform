import datetime
from typing import List, Optional

from sqlalchemy import JSON, Boolean, DateTime, Float, ForeignKey, Integer, String, Text
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship

class Base(DeclarativeBase):
    """Abstract Base Class providing core metadata registries for migrations."""
    pass

class Project(Base):
    """
    Represents an enterprise operational unit workspace tracking target telemetry.
    Maps 1-to-Many against historical prediction scoring arrays.
    """
    __tablename__ = "projects"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    name: Mapped[str] = mapped_column(String(100), nullable=False, index=True)
    description: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime.datetime] = mapped_column(
        DateTime, default=datetime.datetime.utcnow, nullable=False
    )

    # Relationship Linkage: Cascades deletions cleanly down to predictions
    predictions: Mapped[List["PredictionRecord"]] = relationship(
        "PredictionRecord", back_populates="project", cascade="all, delete-orphan"
    )


class PredictionRecord(Base):
    """
    The core ledger recording immutable structural project health states.
    Stores exact snapshot parameters alongside the corresponding model outputs.
    """
    __tablename__ = "prediction_records"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    project_id: Mapped[int] = mapped_column(Integer, ForeignKey("projects.id", ondelete="CASCADE"), nullable=False, index=True)
    
    # 📊 Ingested Operational Telemetry Snapshot Parameters
    delayed_tasks: Mapped[int] = mapped_column(Integer, nullable=False)
    git_commits: Mapped[int] = mapped_column(Integer, nullable=False)
    open_bugs: Mapped[int] = mapped_column(Integer, nullable=False)
    high_priority_bugs: Mapped[int] = mapped_column(Integer, nullable=False)
    sprint_velocity: Mapped[float] = mapped_column(Float, nullable=False)
    developer_workload: Mapped[float] = mapped_column(Float, nullable=False)
    code_review_duration: Mapped[float] = mapped_column(Float, nullable=False)
    requirement_changes: Mapped[int] = mapped_column(Integer, nullable=False)
    meeting_attendance: Mapped[float] = mapped_column(Float, nullable=False)
    ci_cd_failures: Mapped[int] = mapped_column(Integer, nullable=False)
    testing_coverage: Mapped[float] = mapped_column(Float, nullable=False)
    pull_request_activity: Mapped[int] = mapped_column(Integer, nullable=False)

    # 🔮 Machine Learning Pipeline Model Inference Layer Scores
    is_failed: Mapped[bool] = mapped_column(Boolean, nullable=False)
    prediction_label: Mapped[str] = mapped_column(String(50), nullable=False)
    confidence_score: Mapped[float] = mapped_column(Float, nullable=False)
    project_failure_risk_pct: Mapped[float] = mapped_column(Float, nullable=False)

    # Operational Logs & Auditing Metadata
    evaluated_at: Mapped[datetime.datetime] = mapped_column(
        DateTime, default=datetime.datetime.utcnow, nullable=False, index=True
    )

    # Bidirectional Relationship Link back to the primary project workspace boundary
    project: Mapped["Project"] = relationship("Project", back_populates="predictions")
    
class User(Base):
    """
    Represents an authenticated platform entity authorized to interact with tenancy matrices.
    Stores cryptographically salted pass-phrases alongside operational permission scopes.
    """
    __tablename__ = "users"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    email: Mapped[str] = mapped_column(String(150), unique=True, nullable=False, index=True)
    hashed_password: Mapped[str] = mapped_column(String(255), nullable=False)
    is_active: Mapped[bool] = mapped_column(Boolean, default=True, nullable=False)
    role: Mapped[str] = mapped_column(String(50), default="manager", nullable=False) # e.g., admin, manager, viewer
    created_at: Mapped[datetime.datetime] = mapped_column(
        DateTime, default=datetime.datetime.utcnow, nullable=False
    )