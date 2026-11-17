import uuid
from typing import List
from fastapi import APIRouter, Depends, Header, HTTPException, status
from sqlalchemy.orm import Session
from backend.app.core.database import get_db
from backend.app.models.audit import AuditOutcome
from backend.app.models.organization import Organization
from backend.app.models.user import User
from backend.app.schemas.organization import OrganizationCreate, OrganizationRead
from backend.app.security.permissions import Permission
from backend.app.security.rbac import get_current_user, require_permission
from backend.app.services.audit_service import AuditService

router = APIRouter()


@router.post("", response_model=OrganizationRead, status_code=status.HTTP_201_CREATED)
def create_organization(
    org_in: OrganizationCreate,
    db: Session = Depends(get_db),
    current_user: User = Depends(require_permission(Permission.ORGANIZATION_MANAGE)),
    x_correlation_id: str = Header(default_factory=lambda: str(uuid.uuid4()))
):
    """Provisions a new healthcare organization or clinic. Requires ORGANIZATION_MANAGE."""
    existing = db.query(Organization).filter(Organization.license_number == org_in.license_number).first()
    if existing:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="An organization with this license number already exists.",
        )

    org = Organization(
        name=org_in.name,
        license_number=org_in.license_number,
        organization_type=org_in.organization_type,
        region=org_in.region,
        is_active=org_in.is_active
    )
    db.add(org)
    db.commit()
    db.refresh(org)

    AuditService.log_event(
        db=db,
        action="ORGANIZATION_CREATED",
        resource_type="ORGANIZATION",
        resource_id=org.id,
        actor_user_id=current_user.id,
        organization_id=org.id,
        outcome=AuditOutcome.SUCCESS,
        correlation_id=x_correlation_id,
        details={"name": org.name, "region": org.region}
    )
    return org


@router.get("", response_model=List[OrganizationRead], status_code=status.HTTP_200_OK)
def list_organizations(
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """Lists registered healthcare organizations across Abu Dhabi."""
    return db.query(Organization).filter(Organization.is_active == True).all()
