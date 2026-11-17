import uuid
from fastapi import APIRouter, Depends, Header, HTTPException, Request, status
from sqlalchemy.orm import Session
from backend.app.core.database import get_db
from backend.app.core.security import create_access_token, decode_token
from backend.app.models.user import User, UserRole
from backend.app.schemas.user import RefreshTokenRequest, TokenResponse, UserLogin, UserRead
from backend.app.security.rbac import get_current_user
from backend.app.services.auth_service import AuthService

router = APIRouter()


@router.post("/login", response_model=TokenResponse, status_code=status.HTTP_200_OK)
def login(
    login_data: UserLogin,
    request: Request,
    db: Session = Depends(get_db),
    x_correlation_id: str = Header(default_factory=lambda: str(uuid.uuid4()))
):
    """Authenticates user with email and password, returning JWT access & refresh tokens."""
    client_ip = request.client.host if request.client else "unknown"
    try:
        token_response = AuthService.authenticate_user(
            db=db,
            email=login_data.email,
            password=login_data.password,
            correlation_id=x_correlation_id,
            ip_address=client_ip
        )
        return token_response
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=str(e),
            headers={"WWW-Authenticate": "Bearer"},
        )


@router.post("/refresh", response_model=TokenResponse, status_code=status.HTTP_200_OK)
def refresh_token(
    refresh_req: RefreshTokenRequest,
    db: Session = Depends(get_db)
):
    """Exchanges a valid refresh token for a new access token."""
    try:
        payload = decode_token(refresh_req.refresh_token)
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=f"Invalid refresh token: {str(e)}",
            headers={"WWW-Authenticate": "Bearer"},
        )

    if payload.get("type") != "refresh":
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Token is not a valid refresh token.",
        )

    user_id = payload.get("sub")
    user = db.query(User).filter(User.id == user_id).first()
    if not user or not user.is_active:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="User account inactive or not found.",
        )

    new_access_token = create_access_token(
        subject=user.id,
        role=user.role,
        organization_id=user.organization_id
    )

    return TokenResponse(
        access_token=new_access_token,
        expires_in=1800,
        refresh_token=refresh_req.refresh_token,
        user_id=user.id,
        role=user.role,
        full_name=user.full_name,
        organization_id=user.organization_id
    )


@router.get("/me", response_model=UserRead, status_code=status.HTTP_200_OK)
def get_current_user_profile(
    current_user: User = Depends(get_current_user)
):
    """Retrieves profile of the currently authenticated principal."""
    return current_user
