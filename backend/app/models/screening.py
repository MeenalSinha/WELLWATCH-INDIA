import enum
from sqlalchemy import Column, Float, ForeignKey, Integer, String, Text
from sqlalchemy.orm import relationship
from backend.app.models.base import BaseDBModel


class EpistemicStatus(str, enum.Enum):
    VERIFIED = "VERIFIED"
    UNVERIFIED = "UNVERIFIED"
    MISSING = "MISSING"
    UNKNOWN = "UNKNOWN"
    NOT_APPLICABLE = "NOT_APPLICABLE"
    OUTDATED = "OUTDATED"
    CONFLICTING = "CONFLICTING"
    UNAUTHORIZED = "UNAUTHORIZED"


class EncounterType(str, enum.Enum):
    IFHAS_COMPREHENSIVE = "IFHAS_COMPREHENSIVE"
    WELL_CHILD_VISIT = "WELL_CHILD_VISIT"
    COMMUNITY_VITALS_CHECK = "COMMUNITY_VITALS_CHECK"
    CLINIC_FOLLOWUP = "CLINIC_FOLLOWUP"


class ScreeningEncounter(BaseDBModel):
    """Clinical screening encounter capturing vitals, biometrics, and epistemic state."""
    __tablename__ = "screening_encounters"

    patient_id = Column(String(36), ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True)
    provider_id = Column(String(36), ForeignKey("users.id", ondelete="SET NULL"), nullable=True, index=True)
    facility_id = Column(String(36), ForeignKey("organizations.id", ondelete="RESTRICT"), nullable=False, index=True)
    
    encounter_type = Column(String(50), nullable=False, index=True)
    
    # Biometric Vitals
    systolic_bp = Column(Float, nullable=True)
    diastolic_bp = Column(Float, nullable=True)
    pulse_rate = Column(Integer, nullable=True)
    fasting_glucose = Column(Float, nullable=True)
    height_cm = Column(Float, nullable=True)
    weight_kg = Column(Float, nullable=True)
    bmi = Column(Float, nullable=True)
    
    # Epistemic Quality Classification
    epistemic_status = Column(String(50), default=EpistemicStatus.UNVERIFIED.value, nullable=False, index=True)
    notes = Column(Text, nullable=True)

    patient = relationship("User", foreign_keys=[patient_id], back_populates="screenings_as_patient")
    provider = relationship("User", foreign_keys=[provider_id])
    facility = relationship("Organization", back_populates="screenings")
