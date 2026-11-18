from typing import Optional
from fastapi import APIRouter, Depends, Query, HTTPException, status
from sqlalchemy.orm import Session

from backend.app.core.database import get_db
from backend.app.security.rbac import get_current_user
from backend.app.models.user import User, UserRole
from backend.app.schemas.population import (
    PopulationOverviewResponse,
    PediatricPopulationResponse,
    AdultPopulationResponse,
    GeographicAnalyticsResponse,
    PopulationEquityResponse,
    PopulationTrendsResponse,
    ReportGenerateRequest,
    PopulationReportResponse,
)
from backend.app.services.population_analytics_service import PopulationAnalyticsService

router = APIRouter()


@router.get("/overview", response_model=PopulationOverviewResponse)
def get_population_overview(
    organization_id: Optional[str] = Query(None),
    time_window_days: int = Query(90, ge=7, le=365),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """Retrieve population-level overview metrics with mathematically validated denominators."""
    org_filter = organization_id or (
        current_user.organization_id if current_user.role != UserRole.SYSTEM_ADMIN else None
    )
    return PopulationAnalyticsService.get_overview(
        db=db,
        organization_id=org_filter,
        time_window_days=time_window_days,
        user_id=current_user.id,
    )


@router.get("/pediatric", response_model=PediatricPopulationResponse)
def get_pediatric_intelligence(
    organization_id: Optional[str] = Query(None),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """Retrieve dedicated pediatric preventive intelligence (Well-Child visits and developmental stages)."""
    org_filter = organization_id or (
        current_user.organization_id if current_user.role != UserRole.SYSTEM_ADMIN else None
    )
    return PopulationAnalyticsService.get_pediatric_intelligence(
        db=db, organization_id=org_filter, user_id=current_user.id
    )


@router.get("/adult", response_model=AdultPopulationResponse)
def get_adult_intelligence(
    organization_id: Optional[str] = Query(None),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """Retrieve dedicated adult preventive intelligence (ADPHC IFHAS and cardiometabolic distribution)."""
    org_filter = organization_id or (
        current_user.organization_id if current_user.role != UserRole.SYSTEM_ADMIN else None
    )
    return PopulationAnalyticsService.get_adult_intelligence(
        db=db, organization_id=org_filter, user_id=current_user.id
    )


@router.get("/geographic", response_model=GeographicAnalyticsResponse)
def get_geographic_analytics(
    organization_id: Optional[str] = Query(None),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """Retrieve geographic analytics across Abu Dhabi healthcare regions with small-cell privacy suppression."""
    org_filter = organization_id or (
        current_user.organization_id if current_user.role != UserRole.SYSTEM_ADMIN else None
    )
    return PopulationAnalyticsService.get_geographic_analytics(
        db=db, organization_id=org_filter, user_id=current_user.id
    )


@router.get("/equity", response_model=PopulationEquityResponse)
def get_population_equity(
    organization_id: Optional[str] = Query(None),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """Retrieve population equity, disparity ratios, and responsible AI guardrails."""
    org_filter = organization_id or (
        current_user.organization_id if current_user.role != UserRole.SYSTEM_ADMIN else None
    )
    return PopulationAnalyticsService.get_equity_analytics(
        db=db, organization_id=org_filter, user_id=current_user.id
    )


@router.get("/trends", response_model=PopulationTrendsResponse)
def get_population_trends(
    organization_id: Optional[str] = Query(None),
    timeframe: str = Query("MONTHLY"),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """Retrieve longitudinal trends for screening coverage and care-gap closure."""
    org_filter = organization_id or (
        current_user.organization_id if current_user.role != UserRole.SYSTEM_ADMIN else None
    )
    return PopulationAnalyticsService.get_trends(
        db=db, organization_id=org_filter, timeframe=timeframe, user_id=current_user.id
    )


@router.post("/reports", response_model=PopulationReportResponse)
def generate_population_report(
    payload: ReportGenerateRequest,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """Generate an authorized, privacy-preserving aggregate population report."""
    user_role_str = current_user.role.value if hasattr(current_user.role, "value") else str(current_user.role)
    allowed_roles = [
        UserRole.SYSTEM_ADMIN.value,
        UserRole.DATA_GOVERNANCE_OFFICER.value,
        UserRole.HEALTHCARE_PROVIDER.value,
    ]
    if user_role_str not in allowed_roles:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Role not authorized to generate population health reports",
        )

    org_filter = payload.organization_id or (
        current_user.organization_id if user_role_str != UserRole.SYSTEM_ADMIN.value else None
    )
    return PopulationAnalyticsService.generate_report(
        db=db,
        report_type=payload.report_type,
        time_window_days=payload.time_window_days,
        organization_id=org_filter,
        user_id=current_user.id,
    )
