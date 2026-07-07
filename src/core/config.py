import os
from pathlib import Path
from typing import Literal
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field

class EnterpriseSettings(BaseSettings):
    """
    Implements compile-time verified environment tracking.
    Validates configuration layers before the core system boots.
    """
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore"
    )

    # Core Deployment Boundaries
    APP_ENV: Literal["development", "staging", "production"] = "development"
    APP_NAME: str = "AI Project Failure Intelligence Platform SaaS"
    DEBUG: bool = True
    PORT: int = 8000

    # Cryptographic Safeguard Boundaries
    SECRET_KEY: str = Field(..., min_length=32)

    # Persistence Configurations
    DATABASE_URL: str = "sqlite:///./data/project_failure.db"

    # Core Directories for Model Resolution
    BASE_DIR: Path = Path(__file__).resolve().parent.parent.parent
    MODEL_DIR: Path = BASE_DIR / "artifacts" / "models"

    @property
    def is_production(self) -> bool:
        return self.APP_ENV == "production"

# Instantiate our modern EnterpriseSettings class globally
settings = EnterpriseSettings()