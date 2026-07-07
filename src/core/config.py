import os
from pathlib import Path
from typing import Optional
from pydantic_settings import BaseSettings, SettingsConfigDict

# Resolve absolute root directory pathing cleanly
BASE_DIR = Path(__file__).resolve().parent.parent.parent

class Settings(BaseSettings):
    """
    Enterprise Application Settings Manager.
    Uses Pydantic to strictly validate environment configurations at startup.
    """
    # Environment Setup
    APP_ENV: str = "development"
    PROJECT_NAME: str = "AI Project Intelligence Platform"
    DEBUG: bool = True
    
    # API Configurations
    API_HOST: str = "0.0.0.0"
    API_PORT: int = 8000
    API_V1_STR: str = "/api/v1"
    
    # Database Configurations
    DATABASE_URL: str = f"sqlite:///{BASE_DIR}/project_intelligence.db"
    
    # Directory Configurations for Artifacts
    DATA_DIR: Path = BASE_DIR / "data"
    MODEL_DIR: Path = BASE_DIR / "artifacts" / "models"
    
    # Security Configurations
    SECRET_KEY: str = "PRODUCTION_SECURE_CHANGEME_77391!@#!"
    
    # Model Configuration for .env file resolution
    model_config = SettingsConfigDict(
        env_file=f"{BASE_DIR}/.env",
        env_file_encoding="utf-8",
        extra="ignore"
    )

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Ensure critical lifecycle directories exist immediately on startup
        self.DATA_DIR.mkdir(parents=True, exist_ok=True)
        self.MODEL_DIR.mkdir(parents=True, exist_ok=True)

# Instantiate a global settings singleton
settings = Settings()