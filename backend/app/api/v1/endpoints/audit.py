from typing import List, Optional
from fastapi import APIRouter, Depends, Query, status
from sqlalchemy.orm import Session
from backend.app.core.database import get_db
from backend.app.models.audit import AuditLog
from backend.app.models.user import User
from backend.app.schemas.audit import AuditLogRead
from backend.app.security.permissions import Permission
from backend.app.security.rbac import require_permission

router = APIRouter()


@router.get("/logs", response_model=List[AuditLogRead], status_code=status.HTTP_200_OK)
def list_audit_logs(
    skip: int = Query(0, ge=0),
    limit: int = Query(50, ge=1, le=100),
    action: Optional[str] = None,
    resource_type: Optional[str] = None,
    db: Session = Depends(get_db),
    current_user: User = Depends(require_permission(Permission.AUDIT_READ))
):
    """
    Retrieves system-wide immutable audit trail logs.
    Requires AUDIT_READ permission (Data Governance Officer).
    """
    query = db.query(AuditLog)
    if action:
        query = query.filter(AuditLog.action == action)
    if resource_type:
        query = query.filter(AuditLog.resource_type == resource_type)
        
    return query.order_by(AuditLog.created_at.desc()).offset(skip).limit(limit).all()
