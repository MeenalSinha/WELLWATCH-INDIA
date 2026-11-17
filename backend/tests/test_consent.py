import pytest
from backend.app.models.consent import ConsentPurpose, ConsentStatus
from backend.app.models.user import UserRole
from backend.app.schemas.consent import ConsentCreate
from backend.app.services.consent_service import ConsentService


def test_consent_lifecycle(db_session, test_users):
    user = test_users[UserRole.INDIVIDUAL]

    # 1. Grant consent
    consent_dto = ConsentCreate(
        purpose=ConsentPurpose.PREVENTIVE_SCREENING_ELIGIBILITY,
        data_categories=["vitals", "family_history"],
        consent_version="v1.0",
        collection_source="PORTAL_DIRECT"
    )
    consent = ConsentService.create_consent(
        db=db_session,
        user=user,
        data=consent_dto,
        correlation_id="cid-con-01"
    )
    assert consent.status == ConsentStatus.ACTIVE.value
    assert ConsentService.has_active_consent(db_session, user.id, ConsentPurpose.PREVENTIVE_SCREENING_ELIGIBILITY) is True
    assert ConsentService.has_active_consent(db_session, user.id, ConsentPurpose.CHRONIC_RISK_ASSESSMENT) is False

    # 2. Revoke consent
    revoked = ConsentService.revoke_consent(
        db=db_session,
        user=user,
        consent_id=consent.id,
        reason="Patient requested withdrawal",
        correlation_id="cid-con-02"
    )
    assert revoked.status == ConsentStatus.REVOKED.value
    assert revoked.revoked_at is not None
    assert ConsentService.has_active_consent(db_session, user.id, ConsentPurpose.PREVENTIVE_SCREENING_ELIGIBILITY) is False


def test_unauthorized_user_cannot_revoke_others_consent(db_session, test_users):
    user1 = test_users[UserRole.INDIVIDUAL]
    user2 = test_users[UserRole.COMMUNITY_HEALTH_WORKER]

    consent_dto = ConsentCreate(
        purpose=ConsentPurpose.CARE_COORDINATION_OUTREACH,
        data_categories=["contact"],
        consent_version="v1.0"
    )
    consent = ConsentService.create_consent(db=db_session, user=user1, data=consent_dto, correlation_id="cid-con-03")

    with pytest.raises(ValueError, match="Unauthorized"):
        ConsentService.revoke_consent(
            db=db_session,
            user=user2,
            consent_id=consent.id,
            reason="Malicious attempt",
            correlation_id="cid-con-04"
        )
