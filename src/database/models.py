import datetime
from typing import Any, List, Optional

from sqlalchemy import JSON, Boolean, DateTime, Float, ForeignKey, Integer, String, Text
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


class Base(DeclarativeBase):
    """Base class for SQLAlchemy models."""


class Project(Base):
    """Workspace container that groups predictions for a team or initiative."""

    __tablename__ = "projects"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    name: Mapped[str] = mapped_column(String(100), nullable=False, index=True)
    description: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime.datetime] = mapped_column(
        DateTime, default=datetime.datetime.utcnow, nullable=False
    )

    predictions: Mapped[List["PredictionRecord"]] = relationship(
        "PredictionRecord", back_populates="project", cascade="all, delete-orphan"
    )


class PredictionRecord(Base):
    """Historical record of a prediction result for audit and analytics."""

    __tablename__ = "prediction_records"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    project_id: Mapped[int] = mapped_column(Integer, ForeignKey("projects.id", ondelete="CASCADE"), nullable=False, index=True)

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

    is_failed: Mapped[bool] = mapped_column(Boolean, nullable=False)
    prediction_label: Mapped[str] = mapped_column(String(50), nullable=False)
    confidence_score: Mapped[float] = mapped_column(Float, nullable=False)
    project_failure_risk_pct: Mapped[float] = mapped_column(Float, nullable=False)
    explanation: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    feature_summary: Mapped[Optional[Any]] = mapped_column(JSON, nullable=True)
    source_type: Mapped[str] = mapped_column(String(20), default="snapshot", nullable=False)
    source_name: Mapped[Optional[str]] = mapped_column(String(150), nullable=True)

    evaluated_at: Mapped[datetime.datetime] = mapped_column(
        DateTime, default=datetime.datetime.utcnow, nullable=False, index=True
    )

    project: Mapped["Project"] = relationship("Project", back_populates="predictions")


class User(Base):
    """Authenticated platform user for basic access control."""

    __tablename__ = "users"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    email: Mapped[str] = mapped_column(String(150), unique=True, nullable=False, index=True)
    hashed_password: Mapped[str] = mapped_column(String(255), nullable=False)
    is_active: Mapped[bool] = mapped_column(Boolean, default=True, nullable=False)
    role: Mapped[str] = mapped_column(String(50), default="manager", nullable=False)
    created_at: Mapped[datetime.datetime] = mapped_column(
        DateTime, default=datetime.datetime.utcnow, nullable=False
    )