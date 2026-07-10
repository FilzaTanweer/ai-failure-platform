import time
from pathlib import Path
from typing import Any, Callable, TypeVar

from sqlalchemy import create_engine, inspect, text
from sqlalchemy.exc import OperationalError
from sqlalchemy.orm import Session, sessionmaker

from src.core.config import settings
from src.database.models import Base

Path(settings.DATABASE_URL.replace("sqlite:///./", "")).parent.mkdir(parents=True, exist_ok=True)

engine = create_engine(
    settings.DATABASE_URL,
    connect_args={"check_same_thread": False, "timeout": 30} if "sqlite" in settings.DATABASE_URL else {},
    pool_pre_ping=True,
)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

T = TypeVar("T")


def run_db_operation_with_retry(operation: Callable[[], T], *, max_retries: int = 3, delay_seconds: float = 0.2) -> T:
    """Retry transient SQLite lock failures a few times to avoid request flakiness."""
    last_error: Exception | None = None
    for attempt in range(max_retries):
        try:
            return operation()
        except OperationalError as exc:
            last_error = exc
            if "locked" not in str(exc).lower() or attempt == max_retries - 1:
                raise
            time.sleep(delay_seconds)
    if last_error is not None:
        raise last_error
    raise RuntimeError("Database operation failed without a captured error")


def init_db_schema() -> None:
    """Create the schema and add any missing columns for existing databases."""
    Base.metadata.create_all(bind=engine)
    inspector = inspect(engine)
    if "prediction_records" not in inspector.get_table_names():
        return

    existing_columns = {column["name"] for column in inspector.get_columns("prediction_records")}
    migration_specs = {
        "explanation": "TEXT",
        "feature_summary": "TEXT",
        "source_type": "VARCHAR(20)",
        "source_name": "VARCHAR(150)",
    }
    for column_name, column_type in migration_specs.items():
        if column_name not in existing_columns:
            with engine.begin() as connection:
                connection.execute(text(f'ALTER TABLE prediction_records ADD COLUMN {column_name} {column_type}'))


def get_db_context():
    """Provide a transactional database session for each request."""
    db_session: Session = SessionLocal()
    try:
        yield db_session
    finally:
        db_session.close()