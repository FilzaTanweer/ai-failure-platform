from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, Session
from src.core.config import settings
from src.database.models import Base

# Setup standard connection engine mapping back to verified database configuration URLs
# pool_pre_ping=True automatically validates dead connections safely before executing statements
engine = create_engine(
    settings.DATABASE_URL,
    connect_args={"check_same_thread": False} if "sqlite" in settings.DATABASE_URL else {},
    pool_pre_ping=True
)

# Instantiate a thread-isolated transactional session factory boundaries
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

def init_db_schema() -> None:
    """Executes automated table structural binding initialization on bootstrap."""
    Base.metadata.create_all(bind=engine)

def get_db_context():
    """
    FastAPI Context Dependency Provider yielding isolated transactional sessions.
    Guarantees reliable, auto-closing transaction boundaries per web request.
    """
    db_session: Session = SessionLocal()
    try:
        yield db_session
    finally:
        db_session.close()