import enum
from sqlalchemy import Column, DateTime, ForeignKey, String, Text
from sqlalchemy.orm import relationship
from backend.app.models.base import BaseDBModel


class ConsentStatus(str, enum.Enum):
    ACTIVE = "ACTIVE"
    REVOKED = "REVOKED"
    EXPIRED = "EXPIRED"


class ConsentPurpose(str, enum.Enum):
    PREVENTIVE_SCREENING_ELIGIBILITY = "PREVENTIVE_SCREENING_ELIGIBILITY"
    CHRONIC_RISK_ASSESSMENT = "CHRONIC_RISK_ASSESSMENT"
    CARE_COORDINATION_OUTREACH = "CARE_COORDINATION_OUTREACH"
    POPULATION_HEALTH_ANALYTICS = "POPULATION_HEALTH_ANALYTICS"


class ConsentRecord(BaseDBModel):
    """Purpose-based patient consent record with granular scopes and audit history."""
    __tablename__ = "consent_records"

    user_id = Column(String(36), ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True)
    authorized_representative_id = Column(String(36), ForeignKey("users.id", ondelete="SET NULL"), nullable=True)
    
    purpose = Column(String(100), nullable=False, index=True)
    data_categories = Column(Text, nullable=False)  # JSON-encoded list of scopes (e.g. ["vitals", "family_history"])
    status = Column(String(50), default=ConsentStatus.ACTIVE.value, nullable=False, index=True)
    consent_version = Column(String(50), nullable=False)
    
    valid_from = Column(DateTime(timezone=True), nullable=False)
    valid_until = Column(DateTime(timezone=True), nullable=True)
    revoked_at = Column(DateTime(timezone=True), nullable=True)
    collection_source = Column(String(50), default="PORTAL_DIRECT", nullable=False)
    
    user = relationship("User", foreign_keys=[user_id], back_populates="consents")
    representative = relationship("User", foreign_keys=[authorized_representative_id])
