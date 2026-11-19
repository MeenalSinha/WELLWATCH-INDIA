import uuid
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, RedirectResponse
from backend.app.api.v1.api import api_router
from backend.app.core.config import settings
from backend.app.core.database import Base, engine
from backend.app.core.logging import logger
from backend.app.middleware.correlation import CorrelationIdMiddleware
from backend.app.middleware.rate_limit import SimpleRateLimitMiddleware
from backend.app.middleware.security_headers import SecurityHeadersMiddleware


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifecycle manager."""
    if settings.ENVIRONMENT.lower() == "production":
        logger.info("Production mode: Verifying schema via migrations. Automatic Base.metadata.create_all is disabled in production.")
    else:
        logger.info("Development mode: Ensuring local database tables exist...")
        Base.metadata.create_all(bind=engine)
        logger.info("Database schema initialized successfully.")

        try:
            from backend.app.core.database import SessionLocal
            from backend.app.models.user import User
            from backend.app.services.seed_service import seed_initial_development_data

            with SessionLocal() as db:
                if db.query(User).count() == 0:
                    logger.info("Empty user catalog in development mode. Auto-seeding default test personas...")
                    seed_initial_development_data(db)
                    logger.info("Auto-seeding completed successfully.")
        except Exception as seed_err:
            logger.warning(f"Development auto-seeding encountered an issue: {seed_err}")
    yield
    logger.info("Shutting down WellWatch Prevent application.")


app = FastAPI(
    title=settings.PROJECT_NAME,
    version=settings.VERSION,
    description="Production-grade AI-powered population health intelligence and preventive-care navigation platform for Abu Dhabi.",
    openapi_url=f"{settings.API_V1_STR}/openapi.json",
    docs_url=f"{settings.API_V1_STR}/docs",
    redoc_url=f"{settings.API_V1_STR}/redoc",
    lifespan=lifespan
)

# Middleware Pipeline (executed in reverse order of addition)
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.BACKEND_CORS_ORIGINS,
    allow_origin_regex=r"https://.*\.onrender\.com",
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["*"],
)
app.add_middleware(SimpleRateLimitMiddleware, max_requests=settings.RATE_LIMIT_PER_MINUTE)
app.add_middleware(SecurityHeadersMiddleware)
app.add_middleware(CorrelationIdMiddleware)


# Global Centralized Exception Handlers
@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    correlation_id = getattr(request.state, "correlation_id", str(uuid.uuid4()))
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "success": False,
            "correlation_id": correlation_id,
            "error": {
                "code": f"HTTP_{exc.status_code}",
                "message": exc.detail
            }
        },
        headers=getattr(exc, "headers", None)
    )


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    correlation_id = getattr(request.state, "correlation_id", str(uuid.uuid4()))
    errors = []
    for err in exc.errors():
        field = ".".join(str(loc) for loc in err.get("loc", []))
        errors.append(f"{field}: {err.get('msg')}")
        
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content={
            "success": False,
            "correlation_id": correlation_id,
            "error": {
                "code": "VALIDATION_ERROR",
                "message": "; ".join(errors)
            }
        }
    )


@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception):
    correlation_id = getattr(request.state, "correlation_id", str(uuid.uuid4()))
    logger.error(f"Unhandled system exception: {str(exc)}", extra={"correlation_id": correlation_id})
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "success": False,
            "correlation_id": correlation_id,
            "error": {
                "code": "INTERNAL_SERVER_ERROR",
                "message": "An internal server error occurred. Please contact system support with the correlation ID."
            }
        }
    )


# Mount versioned API routes
app.include_router(api_router, prefix=settings.API_V1_STR)


@app.get("/")
def root():
    return {
        "service": settings.PROJECT_NAME,
        "version": settings.VERSION,
        "jurisdiction": settings.JURISDICTION,
        "docs": f"{settings.API_V1_STR}/docs"
    }


@app.get("/docs", include_in_schema=False)
def redirect_docs():
    return RedirectResponse(url=f"{settings.API_V1_STR}/docs")
