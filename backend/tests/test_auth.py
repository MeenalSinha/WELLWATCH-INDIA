import pytest
from backend.app.core.security import decode_token, verify_password
from backend.app.models.user import UserRole
from backend.app.services.auth_service import AuthService


def test_successful_authentication(db_session, test_users):
    user = test_users[UserRole.INDIVIDUAL]
    res = AuthService.authenticate_user(
        db=db_session,
        email=user.email,
        password="ValidPass123!",
        correlation_id="test-cid-auth-01"
    )
    assert res.user_id == user.id
    assert res.role == UserRole.INDIVIDUAL.value
    assert res.access_token is not None
    assert res.refresh_token is not None

    # Decode and verify token claims
    claims = decode_token(res.access_token)
    assert claims["sub"] == user.id
    assert claims["role"] == UserRole.INDIVIDUAL.value
    assert claims["type"] == "access"


def test_invalid_password_raises_error(db_session, test_users):
    user = test_users[UserRole.INDIVIDUAL]
    with pytest.raises(ValueError, match="Invalid email or password"):
        AuthService.authenticate_user(
            db=db_session,
            email=user.email,
            password="WrongPassword!",
            correlation_id="test-cid-auth-02"
        )


def test_account_lockout_after_5_failed_attempts(db_session, test_users):
    user = test_users[UserRole.INDIVIDUAL]
    for _ in range(5):
        try:
            AuthService.authenticate_user(
                db=db_session,
                email=user.email,
                password="WrongPassword!",
                correlation_id="test-cid-auth-03"
            )
        except ValueError:
            pass

    # 6th attempt should trigger account lockout
    with pytest.raises(ValueError, match="temporarily locked"):
        AuthService.authenticate_user(
            db=db_session,
            email=user.email,
            password="ValidPass123!",
            correlation_id="test-cid-auth-04"
        )


def test_deactivated_user_cannot_authenticate(db_session, test_users):
    user = test_users[UserRole.COMMUNITY_HEALTH_WORKER]
    user.is_active = False
    db_session.commit()

    with pytest.raises(ValueError, match="deactivated"):
        AuthService.authenticate_user(
            db=db_session,
            email=user.email,
            password="ValidPass123!",
            correlation_id="test-cid-auth-05"
        )


def test_refresh_token_validation(db_session, test_users):
    user = test_users[UserRole.HEALTHCARE_PROVIDER]
    res = AuthService.authenticate_user(
        db=db_session,
        email=user.email,
        password="ValidPass123!",
        correlation_id="test-cid-auth-06"
    )
    refresh_claims = decode_token(res.refresh_token)
    assert refresh_claims["sub"] == user.id
    assert refresh_claims["type"] == "refresh"
