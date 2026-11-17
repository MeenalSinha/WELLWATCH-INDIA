import enum
from datetime import date, datetime, timezone
from typing import Optional
from sqlalchemy import Column, Date, DateTime, Enum, Float, ForeignKey, Index, Integer, String, Text, UniqueConstraint
from sqlalchemy.orm import relationship
from backend.app.models.base import BaseDBModel


class RecordStatus(str, enum.Enum):
    ACTIVE = "ACTIVE"
    INACTIVE = "INACTIVE"
    MERGED = "MERGED"
    DECEASED = "DECEASED"


class Gender(str, enum.Enum):
    MALE = "MALE"
    FEMALE = "FEMALE"
    OTHER = "OTHER"
    UNKNOWN = "UNKNOWN"


class PopulationClassification(str, enum.Enum):
    PEDIATRIC = "PEDIATRIC"      # 0 to 17 years (Well-Child 0-6y focus)
    ADULT = "ADULT"              # 18 to 64 years (IFHAS comprehensive screening focus)
    GERIATRIC = "GERIATRIC"      # 65+ years (Targeted senior preventive focus)


class CompletenessStatus(str, enum.Enum):
    COMPLETE = "COMPLETE"
    PARTIALLY_COMPLETE = "PARTIALLY_COMPLETE"
    INCOMPLETE = "INCOMPLETE"
    MINIMAL = "MINIMAL"


class IdentifierNamespace(str, enum.Enum):
    EMIRATES_ID = "EMIRATES_ID"
    MRN = "MRN"
    MALAFFI_RECIPROCAL_ID = "MALAFFI_RECIPROCAL_ID"
    PASSPORT = "PASSPORT"
    INTERNAL_UUID = "INTERNAL_UUID"


class IdentifierVerificationStatus(str, enum.Enum):
    VERIFIED = "VERIFIED"
    UNVERIFIED = "UNVERIFIED"
    PENDING_VERIFICATION = "PENDING_VERIFICATION"
    REJECTED = "REJECTED"


class Person(BaseDBModel):
    """
    Core demographic healthcare subject.
    Separated from User login accounts: A Person does not need a User account to have
    medical records, and a User may manage or view multiple Person profiles under authorization.
    """
    __tablename__ = "persons"

    # Minimal identifiable demographic attributes (Data minimization principle)
    first_name = Column(String(100), nullable=False)
    last_name = Column(String(100), nullable=False)
    date_of_birth = Column(Date, nullable=False, index=True)
    gender = Column(Enum(Gender), nullable=False)
    preferred_language = Column(String(10), default="ar", nullable=False)  # 'ar', 'en'
    nationality = Column(String(50), default="ARE", nullable=True)
    emirate = Column(String(50), default="Abu Dhabi", nullable=True)
    district = Column(String(100), nullable=True)

    record_status = Column(Enum(RecordStatus), default=RecordStatus.ACTIVE, nullable=False, index=True)
    
    # Optional link to user login account (if the patient has their own portal login)
    user_id = Column(String(36), ForeignKey("users.id", ondelete="SET NULL"), nullable=True, unique=True)
    
    # Relationships
    identifiers = relationship("PersonIdentifier", back_populates="person", cascade="all, delete-orphan")
    patient_profile = relationship("PatientProfile", back_populates="person", uselist=False, cascade="all, delete-orphan")
    observations = relationship("ClinicalObservation", back_populates="person", cascade="all, delete-orphan")
    conditions = relationship("ClinicalCondition", back_populates="person", cascade="all, delete-orphan")
    family_histories = relationship("FamilyHistory", back_populates="person", cascade="all, delete-orphan")
    encounters = relationship("Encounter", back_populates="person", cascade="all, delete-orphan")
    screenings = relationship("ScreeningRecord", back_populates="person", cascade="all, delete-orphan")
    documents = relationship("ClinicalDocument", back_populates="person", cascade="all, delete-orphan")

    __table_args__ = (
        Index("ix_persons_dob_gender", "date_of_birth", "gender"),
        Index("ix_persons_emirate_status", "emirate", "record_status"),
    )

    @property
    def full_name(self) -> str:
        return f"{self.first_name} {self.last_name}".strip()

    @property
    def age_years(self) -> int:
        today = date.today()
        dob = self.date_of_birth
        return today.year - dob.year - ((today.month, today.day) < (dob.month, dob.day))

    @property
    def age_months(self) -> int:
        today = date.today()
        dob = self.date_of_birth
        years_diff = today.year - dob.year
        months_diff = today.month - dob.month
        total_months = years_diff * 12 + months_diff
        if today.day < dob.day:
            total_months -= 1
        return max(0, total_months)


class PersonIdentifier(BaseDBModel):
    """
    Multiple healthcare and national identifiers scoped by namespace and issuing authority.
    Sensitive identifiers can be hashed or masked for privacy protection.
    """
    __tablename__ = "person_identifiers"

    person_id = Column(String(36), ForeignKey("persons.id", ondelete="CASCADE"), nullable=False, index=True)
    namespace = Column(Enum(IdentifierNamespace), nullable=False)
    issuing_authority = Column(String(100), default="ICP UAE", nullable=False)
    identifier_value = Column(String(255), nullable=False)
    identifier_hash = Column(String(64), nullable=True, index=True)  # SHA-256 for zero-knowledge lookup
    verification_status = Column(Enum(IdentifierVerificationStatus), default=IdentifierVerificationStatus.UNVERIFIED, nullable=False)
    valid_from = Column(Date, nullable=True)
    valid_until = Column(Date, nullable=True)

    person = relationship("Person", back_populates="identifiers")

    __table_args__ = (
        UniqueConstraint("namespace", "identifier_value", name="uq_identifier_namespace_val"),
        Index("ix_identifiers_lookup", "namespace", "identifier_hash"),
    )


class PatientProfile(BaseDBModel):
    """
    Longitudinal healthcare profile tracking classification, completeness, and facility context.
    """
    __tablename__ = "patient_profiles"

    person_id = Column(String(36), ForeignKey("persons.id", ondelete="CASCADE"), nullable=False, unique=True)
    organization_id = Column(String(36), ForeignKey("organizations.id", ondelete="RESTRICT"), nullable=False, index=True)
    
    population_classification = Column(Enum(PopulationClassification), nullable=False, index=True)
    completeness_score = Column(Float, default=0.0, nullable=False)  # 0.0 to 100.0%
    completeness_status = Column(Enum(CompletenessStatus), default=CompletenessStatus.MINIMAL, nullable=False)
    
    last_assessment_date = Column(DateTime(timezone=True), nullable=True)
    primary_facility_name = Column(String(200), nullable=True)
    is_high_risk = Column(Integer, default=0, nullable=False)  # 0 or 1 indicator flag

    person = relationship("Person", back_populates="patient_profile")
    organization = relationship("Organization")


class Household(BaseDBModel):
    """Household unit for legitimate care coordination and community health worker allocation."""
    __tablename__ = "households"

    household_name = Column(String(150), nullable=False)
    organization_id = Column(String(36), ForeignKey("organizations.id", ondelete="RESTRICT"), nullable=False, index=True)
    primary_contact_person_id = Column(String(36), ForeignKey("persons.id", ondelete="SET NULL"), nullable=True)
    emirate = Column(String(50), default="Abu Dhabi", nullable=False)
    district = Column(String(100), nullable=True)
    notes = Column(Text, nullable=True)

    members = relationship("HouseholdMember", back_populates="household", cascade="all, delete-orphan")


class HouseholdMember(BaseDBModel):
    """Membership of an individual in a household."""
    __tablename__ = "household_members"

    household_id = Column(String(36), ForeignKey("households.id", ondelete="CASCADE"), nullable=False, index=True)
    person_id = Column(String(36), ForeignKey("persons.id", ondelete="CASCADE"), nullable=False, index=True)
    relationship_to_head = Column(String(50), default="MEMBER", nullable=False)

    household = relationship("Household", back_populates="members")
    person = relationship("Person")

    __table_args__ = (
        UniqueConstraint("household_id", "person_id", name="uq_household_person"),
    )
