import os
from typing import List, Union
from pydantic import AnyHttpUrl, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    ENVIRONMENT: str = "development"
    PROJECT_NAME: str = "WELLWATCH PREVENT — Life-Course Preventive Care Intelligence"
    VERSION: str = "1.0.0"
    API_V1_STR: str = "/api/v1"
    
    # Security & Cryptography
    SECRET_KEY: str = "wellwatch-prevent-secure-development-secret-key-32chars-min"
    ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 30
    REFRESH_TOKEN_EXPIRE_MINUTES: int = 60 * 24 * 7  # 7 days
    
    # Account Protection
    MAX_LOGIN_ATTEMPTS: int = 5
    LOCKOUT_DURATION_MINUTES: int = 15
    
    # Database
    DATABASE_URL: str = "sqlite:///./data/wellwatch_prevent.db"
    DATABASE_POOL_SIZE: int = 5
    DATABASE_MAX_OVERFLOW: int = 10
    
    # CORS & Hosts
    BACKEND_CORS_ORIGINS: List[str] = [
        "http://localhost:3000",
        "http://127.0.0.1:3000",
        "http://localhost:8000",
        "http://127.0.0.1:8000",
    ]
    ALLOWED_HOSTS: List[str] = ["localhost", "127.0.0.1", "0.0.0.0"]
    
    # Rate Limiting
    RATE_LIMIT_PER_MINUTE: int = 60
    
    # UAE Healthcare Context Defaults
    JURISDICTION: str = "Abu Dhabi, United Arab Emirates"
    DEFAULT_TIMEZONE: str = "Asia/Dubai"

    # Clinical Data Quality & Operational Recency Thresholds
    DATA_QUALITY_VITALS_MAX_RECENCY_DAYS: int = 365
    DATA_QUALITY_SCREENING_MAX_RECENCY_DAYS: int = 1095
    DATA_QUALITY_CONFLICT_THRESHOLD_PERCENT: float = 30.0
    DATA_QUALITY_CONFLICT_WINDOW_HOURS: int = 48
    
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=True,
        extra="ignore"
    )

    @field_validator("SECRET_KEY")
    def validate_secret_key(cls, v: str, info) -> str:
        if len(v) < 32:
            raise ValueError("SECRET_KEY must be at least 32 characters long.")
        return v

    @classmethod
    def validate_production_settings(cls, settings_instance: "Settings") -> None:
        """Fail-safe check ensuring production environments do not use development defaults."""
        if settings_instance.ENVIRONMENT.lower() == "production":
            if "development" in settings_instance.SECRET_KEY.lower() or settings_instance.SECRET_KEY == "wellwatch-prevent-secure-development-secret-key-32chars-min":
                raise ValueError("CRITICAL SECURITY ERROR: In production, SECRET_KEY must be explicitly set and cannot use the development default.")
            if settings_instance.DATABASE_URL.startswith("sqlite"):
                raise ValueError("CRITICAL SECURITY ERROR: In production, DATABASE_URL must point to a production-grade database (e.g. PostgreSQL), not SQLite.")


settings = Settings()
settings.validate_production_settings(settings)
