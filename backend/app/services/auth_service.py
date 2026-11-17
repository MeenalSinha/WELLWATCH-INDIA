from datetime import datetime, timedelta, timezone
from typing import Optional
from sqlalchemy.orm import Session
from backend.app.core.config import settings
from backend.app.core.security import create_access_token, create_refresh_token, verify_password
from backend.app.models.audit import AuditOutcome
from backend.app.models.user import User
from backend.app.schemas.user import TokenResponse
from backend.app.services.audit_service import AuditService


class AuthService:
    @staticmethod
    def authenticate_user(
        db: Session,
        email: str,
        password: str,
        correlation_id: str,
        ip_address: Optional[str] = None
    ) -> TokenResponse:
        user = db.query(User).filter(User.email == email.lower().strip()).first()
        now = datetime.now(timezone.utc)

        # Check if user exists
        if not user:
            AuditService.log_event(
                db=db,
                action="USER_LOGIN_FAILED",
                resource_type="USER",
                outcome=AuditOutcome.FAILURE,
                correlation_id=correlation_id,
                ip_address=ip_address,
                details={"reason": "User not found"}
            )
            raise ValueError("Invalid email or password.")

        # Check if account is locked (handling naive/aware datetime compatibility with SQLite)
        if user.locked_until:
            locked_until = user.locked_until
            if locked_until.tzinfo is None:
                locked_until = locked_until.replace(tzinfo=timezone.utc)
            if locked_until > now:
                remaining_mins = int((locked_until - now).total_seconds() / 60) + 1
                AuditService.log_event(
                    db=db,
                    action="USER_LOGIN_LOCKED",
                    resource_type="USER",
                    resource_id=user.id,
                    actor_user_id=user.id,
                    organization_id=user.organization_id,
                    outcome=AuditOutcome.DENIED,
                    correlation_id=correlation_id,
                    ip_address=ip_address,
                    details={"reason": "Account locked due to excessive failed attempts"}
                )
                raise ValueError(f"Account is temporarily locked. Try again in {remaining_mins} minutes.")

        # Check password
        if not verify_password(password, user.password_hash):
            user.failed_login_count += 1
            if user.failed_login_count >= settings.MAX_LOGIN_ATTEMPTS:
                user.locked_until = now + timedelta(minutes=settings.LOCKOUT_DURATION_MINUTES)
            db.commit()

            AuditService.log_event(
                db=db,
                action="USER_LOGIN_FAILED",
                resource_type="USER",
                resource_id=user.id,
                actor_user_id=user.id,
                organization_id=user.organization_id,
                outcome=AuditOutcome.FAILURE,
                correlation_id=correlation_id,
                ip_address=ip_address,
                details={"failed_count": user.failed_login_count}
            )
            raise ValueError("Invalid email or password.")

        if not user.is_active:
            AuditService.log_event(
                db=db,
                action="USER_LOGIN_DEACTIVATED",
                resource_type="USER",
                resource_id=user.id,
                actor_user_id=user.id,
                organization_id=user.organization_id,
                outcome=AuditOutcome.DENIED,
                correlation_id=correlation_id,
                ip_address=ip_address,
                details={"reason": "Account is deactivated"}
            )
            raise ValueError("User account is deactivated. Contact administrator.")

        # Reset failed login count upon successful authentication
        user.failed_login_count = 0
        user.locked_until = None
        db.commit()

        # Generate tokens
        access_token = create_access_token(
            subject=user.id,
            role=user.role,
            organization_id=user.organization_id
        )
        refresh_token = create_refresh_token(subject=user.id)

        AuditService.log_event(
            db=db,
            action="USER_LOGIN_SUCCESS",
            resource_type="USER",
            resource_id=user.id,
            actor_user_id=user.id,
            organization_id=user.organization_id,
            outcome=AuditOutcome.SUCCESS,
            correlation_id=correlation_id,
            ip_address=ip_address,
            details={"role": user.role}
        )

        return TokenResponse(
            access_token=access_token,
            expires_in=settings.ACCESS_TOKEN_EXPIRE_MINUTES * 60,
            refresh_token=refresh_token,
            user_id=user.id,
            role=user.role,
            full_name=user.full_name,
            organization_id=user.organization_id
        )
