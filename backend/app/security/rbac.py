from typing import List, Set
from fastapi import Depends, HTTPException, Security, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from sqlalchemy.orm import Session
from backend.app.core.database import get_db
from backend.app.core.security import decode_token
from backend.app.models.user import User, UserRole
from backend.app.security.permissions import Permission

security_scheme = HTTPBearer(auto_error=True)

# Granular Role to Permission Mapping
ROLE_PERMISSIONS: dict[UserRole, Set[Permission]] = {
    UserRole.INDIVIDUAL: {
        Permission.PROFILE_READ_OWN,
        Permission.PROFILE_UPDATE_OWN,
    },
    UserRole.PARENT_GUARDIAN: {
        Permission.PROFILE_READ_OWN,
        Permission.PROFILE_UPDATE_OWN,
        Permission.DEPENDENT_READ,
    },
    UserRole.COMMUNITY_HEALTH_WORKER: {
        Permission.SCREENING_CREATE,
        Permission.SCREENING_READ_SCOPED,
        Permission.OUTREACH_MANAGE,
    },
    UserRole.HEALTHCARE_PROVIDER: {
        Permission.SCREENING_CREATE,
        Permission.SCREENING_READ_SCOPED,
        Permission.CLINICAL_REVIEW,
        Permission.PATHWAY_APPROVE,
        Permission.GAP_RESOLVE,
        Permission.GENOMIC_DATA_READ,
        Permission.GENOMIC_DATA_WRITE,
    },
    UserRole.CARE_COORDINATOR: {
        Permission.SCREENING_READ_SCOPED,
        Permission.OUTREACH_MANAGE,
        Permission.GAP_RESOLVE,
    },
    UserRole.POPULATION_HEALTH_ADMIN: {
        Permission.ANALYTICS_POPULATION,
    },
    UserRole.SYSTEM_ADMIN: {
        Permission.USER_MANAGE,
        Permission.ORGANIZATION_MANAGE,
    },
    UserRole.DATA_GOVERNANCE_OFFICER: {
        Permission.AUDIT_READ,
        Permission.CONSENT_AUDIT,
    },
}


def get_current_user(
    credentials: HTTPAuthorizationCredentials = Security(security_scheme),
    db: Session = Depends(get_db)
) -> User:
    """Authenticates the Bearer token and returns the User principal."""
    token = credentials.credentials
    try:
        payload = decode_token(token)
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=str(e),
            headers={"WWW-Authenticate": "Bearer"},
        )
        
    if payload.get("type") != "access":
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid token type: expected access token.",
            headers={"WWW-Authenticate": "Bearer"},
        )

    user_id = payload.get("sub")
    if not user_id:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Token subject missing.",
            headers={"WWW-Authenticate": "Bearer"},
        )

    user = db.query(User).filter(User.id == user_id).first()
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="User not found or account removed.",
            headers={"WWW-Authenticate": "Bearer"},
        )

    if not user.is_active:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="User account is deactivated.",
        )

    return user


def require_permission(required_perm: Permission):
    """Enforces that the authenticated user possesses the specific permission."""
    def permission_checker(current_user: User = Depends(get_current_user)) -> User:
        user_role = UserRole(current_user.role)
        perms = ROLE_PERMISSIONS.get(user_role, set())
        if required_perm not in perms:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Access denied: role '{current_user.role}' lacks permission '{required_perm.value}'.",
            )
        return current_user
    return permission_checker


def require_roles(allowed_roles: List[UserRole]):
    """Enforces that the authenticated user's role is in the allowed list."""
    def role_checker(current_user: User = Depends(get_current_user)) -> User:
        user_role = UserRole(current_user.role)
        if user_role not in allowed_roles:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Access denied: role '{current_user.role}' is not authorized for this resource.",
            )
        return current_user
    return role_checker


def enforce_organization_boundary(user: User, target_org_id: str):
    """
    Validates that clinical staff can only access data within their assigned organization.
    System Administrators and Data Governance Officers have cross-organizational operational scope.
    """
    cross_org_roles = {UserRole.SYSTEM_ADMIN, UserRole.DATA_GOVERNANCE_OFFICER, UserRole.POPULATION_HEALTH_ADMIN}
    user_role = UserRole(user.role)
    if user_role in cross_org_roles:
        return
        
    if not user.organization_id or user.organization_id != target_org_id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Access denied: cross-organizational record access is strictly forbidden.",
        )


def has_permission(role: str, perm: Permission) -> bool:
    """Checks whether a role string has the given permission."""
    try:
        user_role = UserRole(role)
        return perm in ROLE_PERMISSIONS.get(user_role, set())
    except (ValueError, TypeError):
        return False

