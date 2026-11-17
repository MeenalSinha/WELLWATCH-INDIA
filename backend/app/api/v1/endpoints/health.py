from datetime import datetime, timezone
from fastapi import APIRouter, Depends, status
from backend.app.core.config import settings
from backend.app.core.database import check_database_readiness
from backend.app.schemas.common import HealthResponse, ReadinessResponse

router = APIRouter()


@router.get("/health", response_model=HealthResponse, status_code=status.HTTP_200_OK)
def get_health():
    """Liveness probe reporting platform metadata and runtime context."""
    db_status = "connected" if check_database_readiness() else "unreachable"
    return HealthResponse(
        status="healthy",
        version=settings.VERSION,
        environment=settings.ENVIRONMENT,
        jurisdiction=settings.JURISDICTION,
        database=db_status,
        timestamp=datetime.now(timezone.utc).isoformat()
    )


@router.get("/ready", response_model=ReadinessResponse, status_code=status.HTTP_200_OK)
def get_readiness():
    """Readiness probe checking database connectivity for traffic routing."""
    db_ok = check_database_readiness()
    is_ready = db_ok
    return ReadinessResponse(
        status="ready" if is_ready else "not_ready",
        database_ready=db_ok,
        checks={
            "database": db_ok,
            "configuration": True
        }
    )
