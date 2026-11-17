import enum
from datetime import date, datetime, timezone
from typing import Optional
from sqlalchemy import Column, Date, DateTime, Enum, Float, ForeignKey, Index, Integer, String, Text
from sqlalchemy.orm import relationship
from backend.app.models.base import BaseDBModel
from backend.app.models.screening import EpistemicStatus


class EncounterType(str, enum.Enum):
    WELL_CHILD_VISIT = "WELL_CHILD_VISIT"
    IFHAS_COMPREHENSIVE_SCREENING = "IFHAS_COMPREHENSIVE_SCREENING"
    OUTPATIENT_PREVENTIVE = "OUTPATIENT_PREVENTIVE"
    COMMUNITY_HEALTH_CHECK = "COMMUNITY_HEALTH_CHECK"
    TELEHEALTH_PREVENTIVE = "TELEHEALTH_PREVENTIVE"
    FOLLOW_UP_RECALL = "FOLLOW_UP_RECALL"


class EncounterStatus(str, enum.Enum):
    SCHEDULED = "SCHEDULED"
    IN_PROGRESS = "IN_PROGRESS"
    COMPLETED = "COMPLETED"
    CANCELLED = "CANCELLED"
    NO_SHOW = "NO_SHOW"


class ObservationType(str, enum.Enum):
    VITAL_SIGN = "VITAL_SIGN"
    LAB_RESULT = "LAB_RESULT"
    ANTHROPOMETRIC = "ANTHROPOMETRIC"
    LIFESTYLE = "LIFESTYLE"
    DEVELOPMENTAL = "DEVELOPMENTAL"


class ConditionClinicalStatus(str, enum.Enum):
    ACTIVE = "ACTIVE"
    RESOLVED = "RESOLVED"
    REMISSION = "REMISSION"


class ConditionVerificationStatus(str, enum.Enum):
    CONFIRMED = "CONFIRMED"
    PROVISIONAL = "PROVISIONAL"
    REFUTED = "REFUTED"


class FamilyRelationship(str, enum.Enum):
    FATHER = "FATHER"
    MOTHER = "MOTHER"
    SIBLING = "SIBLING"
    CHILD = "CHILD"
    PATERNAL_GRANDFATHER = "PATERNAL_GRANDFATHER"
    PATERNAL_GRANDMOTHER = "PATERNAL_GRANDMOTHER"
    MATERNAL_GRANDFATHER = "MATERNAL_GRANDFATHER"
    MATERNAL_GRANDMOTHER = "MATERNAL_GRANDMOTHER"
    PATERNAL_UNCLE_AUNT = "PATERNAL_UNCLE_AUNT"
    MATERNAL_UNCLE_AUNT = "MATERNAL_UNCLE_AUNT"


class ScreeningStatus(str, enum.Enum):
    COMPLETED = "COMPLETED"
    IN_PROGRESS = "IN_PROGRESS"
    SCHEDULED = "SCHEDULED"
    CANCELLED = "CANCELLED"
    DECLINED = "DECLINED"
    UNKNOWN = "UNKNOWN"
    NOT_RECORDED = "NOT_RECORDED"


class Encounter(BaseDBModel):
    """Healthcare encounter or preventive consultation."""
    __tablename__ = "encounters"

    patient_id = Column(String(36), ForeignKey("persons.id", ondelete="CASCADE"), nullable=False, index=True)
    facility_id = Column(String(36), ForeignKey("organizations.id", ondelete="RESTRICT"), nullable=False, index=True)
    provider_user_id = Column(String(36), ForeignKey("users.id", ondelete="SET NULL"), nullable=True)
    
    encounter_type = Column(Enum(EncounterType), default=EncounterType.OUTPATIENT_PREVENTIVE, nullable=False, index=True)
    encounter_date = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc), nullable=False, index=True)
    status = Column(Enum(EncounterStatus), default=EncounterStatus.COMPLETED, nullable=False)
    clinical_notes = Column(Text, nullable=True)
    
    provenance_id = Column(String(36), ForeignKey("data_provenances.id", ondelete="SET NULL"), nullable=True)

    person = relationship("Person", back_populates="encounters")
    facility = relationship("Organization")
    provider = relationship("User")
    provenance = relationship("DataProvenance")
    
    observations = relationship("ClinicalObservation", back_populates="encounter", cascade="all, delete-orphan")
    screenings = relationship("ScreeningRecord", back_populates="encounter", cascade="all, delete-orphan")
    documents = relationship("ClinicalDocument", back_populates="encounter", cascade="all, delete-orphan")


class ClinicalObservation(BaseDBModel):
    """
    Granular clinical measurement or observation.
    Supports LOINC/SNOMED CT coding, canonical unit normalization, collection vs recording timestamps,
    and the 8-state epistemic data uncertainty classification.
    """
    __tablename__ = "clinical_observations"

    patient_id = Column(String(36), ForeignKey("persons.id", ondelete="CASCADE"), nullable=False, index=True)
    encounter_id = Column(String(36), ForeignKey("encounters.id", ondelete="SET NULL"), nullable=True, index=True)
    observation_type = Column(Enum(ObservationType), nullable=False, index=True)
    
    # Standard Clinical Terminology
    code_system = Column(String(50), default="LOINC", nullable=False)  # LOINC or SNOMED_CT
    code = Column(String(50), nullable=False, index=True)              # e.g. "8480-6" (Systolic BP)
    display_name = Column(String(150), nullable=False)                 # e.g. "Systolic Blood Pressure"

    # Values (Numeric and String for categorical)
    numeric_value = Column(Float, nullable=True)
    string_value = Column(String(255), nullable=True)                  # e.g. "NEVER_SMOKED", "NORMAL"
    unit = Column(String(50), nullable=True)                           # Raw input unit, e.g. "mmHg", "mg/dL", "mmol/L"

    # Canonical Unit Normalization
    canonical_value = Column(Float, nullable=True)                     # Harmonized standard value
    canonical_unit = Column(String(50), nullable=True)                 # Harmonized standard unit

    # Timestamps
    collection_time = Column(DateTime(timezone=True), nullable=False, index=True)
    effective_time = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc), nullable=False)
    recording_time = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc), nullable=False)

    # Epistemic Uncertainty & Provenance
    epistemic_status = Column(Enum(EpistemicStatus), default=EpistemicStatus.VERIFIED, nullable=False, index=True)
    provenance_id = Column(String(36), ForeignKey("data_provenances.id", ondelete="SET NULL"), nullable=True)
    notes = Column(Text, nullable=True)

    person = relationship("Person", back_populates="observations")
    encounter = relationship("Encounter", back_populates="observations")
    provenance = relationship("DataProvenance")

    __table_args__ = (
        Index("ix_obs_patient_code_date", "patient_id", "code", "collection_time"),
    )


class ClinicalCondition(BaseDBModel):
    """Documented chronic or clinical condition with ICD-10/SNOMED CT coding."""
    __tablename__ = "clinical_conditions"

    patient_id = Column(String(36), ForeignKey("persons.id", ondelete="CASCADE"), nullable=False, index=True)
    code_system = Column(String(50), default="ICD-10", nullable=False)
    code = Column(String(50), nullable=False, index=True)              # e.g. "E11.9" (Type 2 Diabetes)
    clinical_description = Column(String(255), nullable=False)
    
    clinical_status = Column(Enum(ConditionClinicalStatus), default=ConditionClinicalStatus.ACTIVE, nullable=False)
    verification_status = Column(Enum(ConditionVerificationStatus), default=ConditionVerificationStatus.CONFIRMED, nullable=False)
    
    onset_date = Column(Date, nullable=True)
    resolution_date = Column(Date, nullable=True)
    
    recording_source_id = Column(String(36), ForeignKey("data_sources.id", ondelete="SET NULL"), nullable=True)
    diagnosing_provider_id = Column(String(36), ForeignKey("users.id", ondelete="SET NULL"), nullable=True)
    provenance_id = Column(String(36), ForeignKey("data_provenances.id", ondelete="SET NULL"), nullable=True)

    person = relationship("Person", back_populates="conditions")
    recording_source = relationship("DataSource")
    diagnosing_provider = relationship("User")
    provenance = relationship("DataProvenance")


class FamilyHistory(BaseDBModel):
    """Documented family medical history for inherited disease risk assessment."""
    __tablename__ = "family_histories"

    patient_id = Column(String(36), ForeignKey("persons.id", ondelete="CASCADE"), nullable=False, index=True)
    relationship_to_patient = Column(Enum(FamilyRelationship), nullable=False)
    condition_code = Column(String(50), nullable=True)                 # ICD-10 e.g. "I25.1", "E11"
    condition_name = Column(String(200), nullable=False)               # e.g. "Coronary Artery Disease", "Type 2 Diabetes"
    approximate_age_of_onset = Column(Integer, nullable=True)
    is_deceased = Column(Integer, default=0, nullable=False)
    
    verification_status = Column(Enum(EpistemicStatus), default=EpistemicStatus.UNVERIFIED, nullable=False)
    provenance_id = Column(String(36), ForeignKey("data_provenances.id", ondelete="SET NULL"), nullable=True)

    person = relationship("Person", back_populates="family_histories")
    provenance = relationship("DataProvenance")


class ScreeningRecord(BaseDBModel):
    """
    Formal preventive screening event record (e.g. Well-Child visit milestones, IFHAS cancer panels).
    """
    __tablename__ = "screening_records"

    patient_id = Column(String(36), ForeignKey("persons.id", ondelete="CASCADE"), nullable=False, index=True)
    encounter_id = Column(String(36), ForeignKey("encounters.id", ondelete="SET NULL"), nullable=True, index=True)
    
    screening_type = Column(String(100), nullable=False, index=True)    # e.g. "WELL_CHILD_12M", "IFHAS_DIABETES_PANEL"
    screening_date = Column(Date, nullable=False, index=True)
    screening_status = Column(Enum(ScreeningStatus), default=ScreeningStatus.COMPLETED, nullable=False, index=True)
    
    completion_evidence_ref = Column(String(255), nullable=True)       # e.g. Lab requisition / encounter number
    performing_facility_id = Column(String(36), ForeignKey("organizations.id", ondelete="SET NULL"), nullable=True)
    findings_summary = Column(Text, nullable=True)
    
    epistemic_status = Column(Enum(EpistemicStatus), default=EpistemicStatus.VERIFIED, nullable=False)
    provenance_id = Column(String(36), ForeignKey("data_provenances.id", ondelete="SET NULL"), nullable=True)

    person = relationship("Person", back_populates="screenings")
    encounter = relationship("Encounter", back_populates="screenings")
    facility = relationship("Organization")
    provenance = relationship("DataProvenance")

    __table_args__ = (
        Index("ix_screenings_patient_type_date", "patient_id", "screening_type", "screening_date"),
    )


class ClinicalDocument(BaseDBModel):
    """
    Metadata for authorized clinical documents, reports, and lab attachments.
    Stores metadata and secure storage URIs rather than large blobs in the relational DB.
    """
    __tablename__ = "clinical_documents"

    patient_id = Column(String(36), ForeignKey("persons.id", ondelete="CASCADE"), nullable=False, index=True)
    encounter_id = Column(String(36), ForeignKey("encounters.id", ondelete="SET NULL"), nullable=True)
    document_type = Column(String(100), nullable=False)                # "LAB_REPORT", "WELL_CHILD_RECORD", "IFHAS_SUMMARY"
    title = Column(String(255), nullable=False)
    mime_type = Column(String(100), default="application/pdf", nullable=False)
    storage_uri = Column(String(500), nullable=False)                  # Protected storage path
    file_size_bytes = Column(Integer, default=0, nullable=False)
    checksum_sha256 = Column(String(64), nullable=False)
    
    provenance_id = Column(String(36), ForeignKey("data_provenances.id", ondelete="SET NULL"), nullable=True)

    person = relationship("Person", back_populates="documents")
    encounter = relationship("Encounter", back_populates="documents")
    provenance = relationship("DataProvenance")
