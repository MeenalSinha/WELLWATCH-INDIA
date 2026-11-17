import json
from datetime import datetime, timezone
from typing import List, Optional
from sqlalchemy.orm import Session
from backend.app.models.audit import AuditOutcome
from backend.app.models.consent import ConsentPurpose, ConsentRecord, ConsentStatus
from backend.app.models.user import User
from backend.app.schemas.consent import ConsentCreate
from backend.app.services.audit_service import AuditService


class ConsentService:
    @staticmethod
    def create_consent(
        db: Session,
        user: User,
        data: ConsentCreate,
        correlation_id: str
    ) -> ConsentRecord:
        target_user_id = data.user_id if data.user_id else user.id

        # Deactivate any previous active consent for the same purpose
        existing_active = db.query(ConsentRecord).filter(
            ConsentRecord.user_id == target_user_id,
            ConsentRecord.purpose == data.purpose.value,
            ConsentRecord.status == ConsentStatus.ACTIVE.value
        ).all()
        now = datetime.now(timezone.utc)
        for prev in existing_active:
            prev.status = ConsentStatus.EXPIRED.value
            prev.valid_until = now

        consent = ConsentRecord(
            user_id=target_user_id,
            authorized_representative_id=data.authorized_representative_id,
            purpose=data.purpose.value,
            data_categories=json.dumps(data.data_categories),
            status=ConsentStatus.ACTIVE.value,
            consent_version=data.consent_version,
            valid_from=now,
            valid_until=data.valid_until,
            collection_source=data.collection_source
        )
        db.add(consent)
        db.commit()
        db.refresh(consent)

        AuditService.log_event(
            db=db,
            action="CONSENT_RECORD_CREATED",
            resource_type="CONSENT",
            resource_id=consent.id,
            actor_user_id=user.id,
            outcome=AuditOutcome.SUCCESS,
            correlation_id=correlation_id,
            details={"purpose": data.purpose.value, "version": data.consent_version}
        )
        return consent

    @staticmethod
    def revoke_consent(
        db: Session,
        user: User,
        consent_id: str,
        reason: Optional[str],
        correlation_id: str
    ) -> ConsentRecord:
        consent = db.query(ConsentRecord).filter(ConsentRecord.id == consent_id).first()
        if not consent:
            raise ValueError("Consent record not found.")

        # Ensure only the subject or guardian can revoke
        if consent.user_id != user.id and consent.authorized_representative_id != user.id and user.role != "DATA_GOVERNANCE_OFFICER":
            raise ValueError("Unauthorized to revoke this consent record.")

        now = datetime.now(timezone.utc)
        consent.status = ConsentStatus.REVOKED.value
        consent.revoked_at = now
        db.commit()
        db.refresh(consent)

        AuditService.log_event(
            db=db,
            action="CONSENT_REVOKED",
            resource_type="CONSENT",
            resource_id=consent.id,
            actor_user_id=user.id,
            outcome=AuditOutcome.SUCCESS,
            correlation_id=correlation_id,
            details={"reason": reason or "User withdrawal"}
        )
        return consent

    @staticmethod
    def has_active_consent(db: Session, user_id: str, purpose: ConsentPurpose) -> bool:
        """Verifies if patient has an active, valid consent record for a specific purpose."""
        now = datetime.now(timezone.utc)
        record = db.query(ConsentRecord).filter(
            ConsentRecord.user_id == user_id,
            ConsentRecord.purpose == purpose.value,
            ConsentRecord.status == ConsentStatus.ACTIVE.value
        ).first()

        if not record:
            return False
        if record.valid_until and record.valid_until < now:
            return False
        return True
