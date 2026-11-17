import json
import uuid
from typing import List, Optional
from fastapi import APIRouter, Depends, Header, HTTPException, status
from sqlalchemy.orm import Session
from backend.app.core.database import get_db
from backend.app.models.consent import ConsentRecord
from backend.app.models.user import User
from backend.app.schemas.consent import ConsentCreate, ConsentRead, ConsentRevoke
from backend.app.security.permissions import Permission
from backend.app.security.rbac import get_current_user, require_permission
from backend.app.services.consent_service import ConsentService

router = APIRouter()


def _to_consent_read(c: ConsentRecord) -> ConsentRead:
    return ConsentRead(
        id=c.id,
        user_id=c.user_id,
        authorized_representative_id=c.authorized_representative_id,
        purpose=c.purpose,
        data_categories=json.loads(c.data_categories) if c.data_categories else [],
        status=c.status,
        consent_version=c.consent_version,
        collection_source=c.collection_source,
        valid_from=c.valid_from,
        valid_until=c.valid_until,
        revoked_at=c.revoked_at,
        created_at=c.created_at
    )


@router.post("", response_model=ConsentRead, status_code=status.HTTP_201_CREATED)
def grant_consent(
    consent_in: ConsentCreate,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
    x_correlation_id: str = Header(default_factory=lambda: str(uuid.uuid4()))
):
    """Registers an explicit, purpose-based consent directive."""
    record = ConsentService.create_consent(
        db=db,
        user=current_user,
        data=consent_in,
        correlation_id=x_correlation_id
    )
    return _to_consent_read(record)


@router.post("/{consent_id}/revoke", response_model=ConsentRead, status_code=status.HTTP_200_OK)
def revoke_consent(
    consent_id: str,
    revoke_data: ConsentRevoke,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
    x_correlation_id: str = Header(default_factory=lambda: str(uuid.uuid4()))
):
    """Revokes an existing consent directive. Immediately ceases optional data processing."""
    try:
        revoked = ConsentService.revoke_consent(
            db=db,
            user=current_user,
            consent_id=consent_id,
            reason=revoke_data.reason,
            correlation_id=x_correlation_id
        )
        return _to_consent_read(revoked)
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e),
        )


@router.get("/me", response_model=List[ConsentRead], status_code=status.HTTP_200_OK)
def get_own_consents(
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """Retrieves all active and historical consent records for the current user."""
    records = db.query(ConsentRecord).filter(ConsentRecord.user_id == current_user.id).order_by(ConsentRecord.created_at.desc()).all()
    return [_to_consent_read(r) for r in records]


@router.get("/audit", response_model=List[ConsentRead], status_code=status.HTTP_200_OK)
def list_consents_for_audit(
    db: Session = Depends(get_db),
    current_user: User = Depends(require_permission(Permission.CONSENT_AUDIT))
):
    """Audits consent directives across the system. Requires CONSENT_AUDIT permission."""
    records = db.query(ConsentRecord).order_by(ConsentRecord.created_at.desc()).limit(100).all()
    return [_to_consent_read(r) for r in records]
