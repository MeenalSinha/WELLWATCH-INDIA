import uuid
from typing import List
from fastapi import APIRouter, Depends, Header, HTTPException, status
from sqlalchemy.orm import Session
from backend.app.core.database import get_db
from backend.app.models.guardian import Guardianship, GuardianshipStatus
from backend.app.models.user import User, UserRole
from backend.app.schemas.guardian import GuardianshipCreate, GuardianshipRead, GuardianshipVerify
from backend.app.schemas.user import UserRead
from backend.app.security.rbac import get_current_user, require_roles
from backend.app.services.guardian_service import GuardianService

router = APIRouter()


@router.post("/link", response_model=GuardianshipRead, status_code=status.HTTP_201_CREATED)
def link_dependent(
    link_data: GuardianshipCreate,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
    x_correlation_id: str = Header(default_factory=lambda: str(uuid.uuid4()))
):
    """Submits a verification request to link a pediatric dependent."""
    try:
        guardianship = GuardianService.create_guardianship_request(
            db=db,
            guardian=current_user,
            data=link_data,
            correlation_id=x_correlation_id
        )
        return guardianship
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e),
        )


@router.post("/{guardianship_id}/verify", response_model=GuardianshipRead, status_code=status.HTTP_200_OK)
def verify_guardianship(
    guardianship_id: str,
    verification_data: GuardianshipVerify,
    db: Session = Depends(get_db),
    current_user: User = Depends(require_roles([UserRole.DATA_GOVERNANCE_OFFICER, UserRole.HEALTHCARE_PROVIDER])),
    x_correlation_id: str = Header(default_factory=lambda: str(uuid.uuid4()))
):
    """Verifies or rejects a guardianship relationship. Requires GOVERNANCE or PROVIDER role."""
    try:
        updated = GuardianService.verify_guardianship(
            db=db,
            verifier=current_user,
            guardianship_id=guardianship_id,
            verification=verification_data,
            correlation_id=x_correlation_id
        )
        return updated
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e),
        )


@router.get("/dependents", response_model=List[UserRead], status_code=status.HTTP_200_OK)
def list_verified_dependents(
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """Returns verified dependent children profiles linked to the current guardian."""
    links = db.query(Guardianship).filter(
        Guardianship.guardian_user_id == current_user.id,
        Guardianship.verification_status == GuardianshipStatus.VERIFIED.value
    ).all()
    
    dependent_ids = [l.dependent_user_id for l in links]
    if not dependent_ids:
        return []
    return db.query(User).filter(User.id.in_(dependent_ids)).all()
