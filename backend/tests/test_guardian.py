import pytest
from backend.app.models.guardian import GuardianshipStatus, RelationshipType
from backend.app.models.user import UserRole
from backend.app.schemas.guardian import GuardianshipCreate, GuardianshipVerify
from backend.app.services.guardian_service import GuardianService


def test_guardianship_verification_workflow(db_session, test_users):
    parent = test_users[UserRole.PARENT_GUARDIAN]
    child = test_users["CHILD"]
    gov_officer = test_users[UserRole.DATA_GOVERNANCE_OFFICER]

    # 1. Create request
    create_dto = GuardianshipCreate(
        dependent_user_id=child.id,
        relationship_type=RelationshipType.FATHER,
        verification_evidence_ref="CIVIL-DOC-987"
    )
    guardianship = GuardianService.create_guardianship_request(
        db=db_session,
        guardian=parent,
        data=create_dto,
        correlation_id="cid-guard-01"
    )
    assert guardianship.verification_status == GuardianshipStatus.PENDING.value
    assert GuardianService.is_verified_guardian(db_session, parent.id, child.id) is False

    # 2. Verify request
    verify_dto = GuardianshipVerify(verification_status=GuardianshipStatus.VERIFIED)
    updated = GuardianService.verify_guardianship(
        db=db_session,
        verifier=gov_officer,
        guardianship_id=guardianship.id,
        verification=verify_dto,
        correlation_id="cid-guard-02"
    )
    assert updated.verification_status == GuardianshipStatus.VERIFIED.value
    assert updated.verified_by_user_id == gov_officer.id
    assert GuardianService.is_verified_guardian(db_session, parent.id, child.id) is True


def test_invalid_dependent_linkage_rejected(db_session, test_users):
    parent = test_users[UserRole.PARENT_GUARDIAN]
    create_dto = GuardianshipCreate(
        dependent_user_id="non-existent-user-id",
        relationship_type=RelationshipType.MOTHER
    )
    with pytest.raises(ValueError, match="does not exist"):
        GuardianService.create_guardianship_request(
            db=db_session,
            guardian=parent,
            data=create_dto,
            correlation_id="cid-guard-03"
        )
