import uuid
from typing import List, Optional
from fastapi import APIRouter, Depends, Header, HTTPException, Query, status
from sqlalchemy.orm import Session
from backend.app.core.database import get_db
from backend.app.core.security import get_password_hash
from backend.app.models.audit import AuditOutcome
from backend.app.models.user import User, UserRole
from backend.app.schemas.user import UserCreate, UserRead
from backend.app.security.permissions import Permission
from backend.app.security.rbac import get_current_user, require_permission
from backend.app.services.audit_service import AuditService

router = APIRouter()


@router.post("", response_model=UserRead, status_code=status.HTTP_201_CREATED)
def create_user(
    user_in: UserCreate,
    db: Session = Depends(get_db),
    current_user: User = Depends(require_permission(Permission.USER_MANAGE)),
    x_correlation_id: str = Header(default_factory=lambda: str(uuid.uuid4()))
):
    """Provisions a new system user account. Requires USER_MANAGE permission."""
    existing = db.query(User).filter(User.email == user_in.email.lower().strip()).first()
    if existing:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="A user with this email address already exists.",
        )

    user = User(
        email=user_in.email.lower().strip(),
        full_name=user_in.full_name,
        password_hash=get_password_hash(user_in.password),
        role=user_in.role.value,
        phone_number=user_in.phone_number,
        organization_id=user_in.organization_id,
        is_active=True,
        is_verified=True
    )
    db.add(user)
    db.commit()
    db.refresh(user)

    AuditService.log_event(
        db=db,
        action="USER_CREATED",
        resource_type="USER",
        resource_id=user.id,
        actor_user_id=current_user.id,
        organization_id=user.organization_id,
        outcome=AuditOutcome.SUCCESS,
        correlation_id=x_correlation_id,
        details={"assigned_role": user.role}
    )
    return user


@router.get("", response_model=List[UserRead], status_code=status.HTTP_200_OK)
def list_users(
    skip: int = Query(0, ge=0),
    limit: int = Query(50, ge=1, le=100),
    role: Optional[UserRole] = None,
    db: Session = Depends(get_db),
    current_user: User = Depends(require_permission(Permission.USER_MANAGE))
):
    """Lists system users with pagination. Requires USER_MANAGE permission."""
    query = db.query(User)
    if role:
        query = query.filter(User.role == role.value)
    return query.offset(skip).limit(limit).all()


@router.get("/{user_id}", response_model=UserRead, status_code=status.HTTP_200_OK)
def get_user_by_id(
    user_id: str,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """Retrieves a specific user profile. Users can read themselves; otherwise requires USER_MANAGE."""
    if current_user.id != user_id and current_user.role != UserRole.SYSTEM_ADMIN.value:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Unauthorized to access this user profile.",
        )

    user = db.query(User).filter(User.id == user_id).first()
    if not user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="User not found.",
        )
    return user
