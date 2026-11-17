import uuid
from typing import List, Optional
from fastapi import APIRouter, Depends, Header, HTTPException, Query, status
from sqlalchemy.orm import Session
from backend.app.core.database import get_db
from backend.app.models.screening import ScreeningEncounter
from backend.app.models.user import User
from backend.app.schemas.screening import ScreeningCreate, ScreeningRead
from backend.app.security.permissions import Permission
from backend.app.security.rbac import enforce_organization_boundary, require_permission
from backend.app.services.screening_service import ScreeningService

router = APIRouter()


@router.post("", response_model=ScreeningRead, status_code=status.HTTP_201_CREATED)
def record_screening(
    screening_in: ScreeningCreate,
    db: Session = Depends(get_db),
    current_user: User = Depends(require_permission(Permission.SCREENING_CREATE)),
    x_correlation_id: str = Header(default_factory=lambda: str(uuid.uuid4()))
):
    """
    Records a clinical screening encounter with biometric vitals and epistemic classification.
    Requires SCREENING_CREATE permission (CHW or Healthcare Provider).
    """
    # Enforce facility tenancy
    enforce_organization_boundary(current_user, screening_in.facility_id)

    encounter = ScreeningService.create_screening_encounter(
        db=db,
        provider=current_user,
        data=screening_in,
        correlation_id=x_correlation_id
    )
    return encounter


@router.get("", response_model=List[ScreeningRead], status_code=status.HTTP_200_OK)
def list_screenings(
    facility_id: Optional[str] = None,
    db: Session = Depends(get_db),
    current_user: User = Depends(require_permission(Permission.SCREENING_READ_SCOPED))
):
    """
    Retrieves screening encounters scoped to the user's assigned healthcare facility.
    Requires SCREENING_READ_SCOPED permission.
    """
    if facility_id:
        enforce_organization_boundary(current_user, facility_id)
        
    return ScreeningService.get_scoped_screenings(
        db=db,
        user=current_user,
        facility_id=facility_id
    )
