from fastapi import APIRouter
from backend.app.api.v1.endpoints import (
    audit,
    auth,
    consent,
    guardians,
    health,
    organizations,
    screenings,
    users,
)

api_router = APIRouter()

api_router.include_router(health.router, tags=["Health & Readiness"])
api_router.include_router(auth.router, prefix="/auth", tags=["Authentication & Session"])
api_router.include_router(users.router, prefix="/users", tags=["Users & Profiles"])
api_router.include_router(organizations.router, prefix="/organizations", tags=["Healthcare Organizations"])
api_router.include_router(guardians.router, prefix="/guardians", tags=["Pediatric Guardianship"])
api_router.include_router(consent.router, prefix="/consent", tags=["Consent & Privacy"])
api_router.include_router(screenings.router, prefix="/screenings", tags=["Clinical Screenings"])
api_router.include_router(audit.router, prefix="/audit", tags=["Audit & Governance"])
