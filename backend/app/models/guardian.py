import enum
from sqlalchemy import Column, ForeignKey, String
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
    """Verified parent/guardian relationship linking adult to pediatric dependent."""
    __tablename__ = "guardianships"

    guardian_user_id = Column(String(36), ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True)
    dependent_user_id = Column(String(36), ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True)
    relationship_type = Column(String(50), nullable=False)
    verification_status = Column(String(50), default=GuardianshipStatus.PENDING.value, nullable=False, index=True)
    verification_evidence_ref = Column(String(255), nullable=True)
    verified_by_user_id = Column(String(36), ForeignKey("users.id", ondelete="SET NULL"), nullable=True)

    guardian = relationship("User", foreign_keys=[guardian_user_id])
    dependent = relationship("User", foreign_keys=[dependent_user_id])
    verifier = relationship("User", foreign_keys=[verified_by_user_id])
