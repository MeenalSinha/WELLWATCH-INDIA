from sqlalchemy.orm import Session
from backend.app.models.audit import AuditOutcome
from backend.app.models.guardian import Guardianship, GuardianshipStatus, RelationshipType
from backend.app.models.user import User, UserRole
from backend.app.schemas.guardian import GuardianshipCreate, GuardianshipVerify
from backend.app.services.audit_service import AuditService


class GuardianService:
    @staticmethod
    def create_guardianship_request(
        db: Session,
        guardian: User,
        data: GuardianshipCreate,
        correlation_id: str
    ) -> Guardianship:
        # Verify dependent exists
        dependent = db.query(User).filter(User.id == data.dependent_user_id).first()
        if not dependent:
            raise ValueError("Target dependent user account does not exist.")

        # Check existing guardianship
        existing = db.query(Guardianship).filter(
            Guardianship.guardian_user_id == guardian.id,
            Guardianship.dependent_user_id == dependent.id
        ).first()
        if existing and existing.verification_status in [GuardianshipStatus.VERIFIED.value, GuardianshipStatus.PENDING.value]:
            raise ValueError("A guardianship relationship already exists or is pending verification.")

        guardianship = Guardianship(
            guardian_user_id=guardian.id,
            dependent_user_id=dependent.id,
            relationship_type=data.relationship_type.value,
            verification_status=GuardianshipStatus.PENDING.value,
            verification_evidence_ref=data.verification_evidence_ref
        )
        db.add(guardianship)
        db.commit()
        db.refresh(guardianship)

        AuditService.log_event(
            db=db,
            action="GUARDIANSHIP_REQUEST_CREATED",
            resource_type="GUARDIANSHIP",
            resource_id=guardianship.id,
            actor_user_id=guardian.id,
            outcome=AuditOutcome.SUCCESS,
            correlation_id=correlation_id,
            details={"relationship": data.relationship_type.value}
        )
        return guardianship

    @staticmethod
    def verify_guardianship(
        db: Session,
        verifier: User,
        guardianship_id: str,
        verification: GuardianshipVerify,
        correlation_id: str
    ) -> Guardianship:
        guardianship = db.query(Guardianship).filter(Guardianship.id == guardianship_id).first()
        if not guardianship:
            raise ValueError("Guardianship record not found.")

        guardianship.verification_status = verification.verification_status.value
        guardianship.verified_by_user_id = verifier.id
        db.commit()
        db.refresh(guardianship)

        AuditService.log_event(
            db=db,
            action="GUARDIANSHIP_STATUS_UPDATED",
            resource_type="GUARDIANSHIP",
            resource_id=guardianship.id,
            actor_user_id=verifier.id,
            outcome=AuditOutcome.SUCCESS,
            correlation_id=correlation_id,
            details={"new_status": verification.verification_status.value}
        )
        return guardianship

    @staticmethod
    def is_verified_guardian(db: Session, guardian_id: str, dependent_id: str) -> bool:
        record = db.query(Guardianship).filter(
            Guardianship.guardian_user_id == guardian_id,
            Guardianship.dependent_user_id == dependent_id,
            Guardianship.verification_status == GuardianshipStatus.VERIFIED.value
        ).first()
        return record is not None
