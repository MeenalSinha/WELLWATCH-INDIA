import enum
from datetime import datetime, timezone
from sqlalchemy import Column, DateTime, ForeignKey, Index, String, Text
from sqlalchemy.orm import relationship
from backend.app.models.base import BaseDBModel


class GuardianshipStatus(str, enum.Enum):
    PENDING = "PENDING"
    VERIFIED = "VERIFIED"
    REJECTED = "REJECTED"
    REVOKED = "REVOKED"


class RelationshipType(str, enum.Enum):
    MOTHER = "MOTHER"
    FATHER = "FATHER"
    LEGAL_GUARDIAN = "LEGAL_GUARDIAN"
    COURT_APPOINTED = "COURT_APPOINTED"


class Guardianship(BaseDBModel):
    """
    Verified parent/guardian relationship linking adult to pediatric dependent.
    Complies with UAE Child Rights Law (Wadeema's Law).
    Supports linkage by User ID (Phase 3 identity auth) and Person ID (Phase 4 longitudinal records).
    """
    __tablename__ = "guardianships"

    # User Account Links (Phase 3 Auth)
    guardian_user_id = Column(String(36), ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True)
    dependent_user_id = Column(String(36), ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True)
    
    # Person Subject Links (Phase 4 Healthcare Database)
    guardian_person_id = Column(String(36), ForeignKey("persons.id", ondelete="CASCADE"), nullable=True, index=True)
    dependent_person_id = Column(String(36), ForeignKey("persons.id", ondelete="CASCADE"), nullable=True, index=True)
    
    relationship_type = Column(String(50), nullable=False)
    verification_status = Column(String(50), default=GuardianshipStatus.PENDING.value, nullable=False, index=True)
    verification_evidence_ref = Column(String(255), nullable=True)
    verified_by_user_id = Column(String(36), ForeignKey("users.id", ondelete="SET NULL"), nullable=True)

    # Scope and Validity
    authorization_scope = Column(String(100), default="FULL_PREVENTIVE_HEALTHCARE_PROXY", nullable=False)
    valid_from = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc), nullable=False)
    valid_until = Column(DateTime(timezone=True), nullable=True)
    revoked_at = Column(DateTime(timezone=True), nullable=True)
    revocation_reason = Column(Text, nullable=True)

    # Relationships
    guardian = relationship("User", foreign_keys=[guardian_user_id])
    dependent = relationship("User", foreign_keys=[dependent_user_id])
    verifier = relationship("User", foreign_keys=[verified_by_user_id])
    
    guardian_person = relationship("Person", foreign_keys=[guardian_person_id])
    dependent_person = relationship("Person", foreign_keys=[dependent_person_id])

    __table_args__ = (
        Index("ix_guardianship_scope_status", "verification_status", "authorization_scope"),
    )
