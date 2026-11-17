import json
from backend.app.models.audit import AuditOutcome
from backend.app.models.user import UserRole
from backend.app.services.audit_service import AuditService


def test_audit_logging_and_sanitization(db_session, test_users):
    user = test_users[UserRole.HEALTHCARE_PROVIDER]

    # Attempt to log with sensitive fields
    raw_details = {
        "event_reason": "Routine screening check",
        "password": "SecretPassword123!",
        "access_token": "eyJhbGciOi...",
        "systolic_bp": 140.0,
        "fasting_glucose": 130.0,
        "safe_flag": "FLAG_NORMAL"
    }

    audit_entry = AuditService.log_event(
        db=db_session,
        action="TEST_ACTION",
        resource_type="PATIENT",
        resource_id="patient-uuid-123",
        actor_user_id=user.id,
        outcome=AuditOutcome.SUCCESS,
        correlation_id="test-corr-id-999",
        details=raw_details
    )

    assert audit_entry.id is not None
    assert audit_entry.correlation_id == "test-corr-id-999"
    assert audit_entry.action == "TEST_ACTION"

    # Verify sanitization in stored details
    stored = json.loads(audit_entry.details)
    assert "safe_flag" in stored
    assert "event_reason" in stored
    assert "password" not in stored
    assert "access_token" not in stored
    assert "systolic_bp" not in stored
    assert "fasting_glucose" not in stored
