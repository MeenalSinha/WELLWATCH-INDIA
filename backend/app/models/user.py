import enum
from sqlalchemy import Boolean, Column, DateTime, ForeignKey, Integer, String
from sqlalchemy.orm import relationship
from backend.app.models.base import BaseDBModel


class UserRole(str, enum.Enum):
    INDIVIDUAL = "INDIVIDUAL"
    PARENT_GUARDIAN = "PARENT_GUARDIAN"
    COMMUNITY_HEALTH_WORKER = "COMMUNITY_HEALTH_WORKER"
    HEALTHCARE_PROVIDER = "HEALTHCARE_PROVIDER"
    CARE_COORDINATOR = "CARE_COORDINATOR"
    POPULATION_HEALTH_ADMIN = "POPULATION_HEALTH_ADMIN"
    SYSTEM_ADMIN = "SYSTEM_ADMIN"
    DATA_GOVERNANCE_OFFICER = "DATA_GOVERNANCE_OFFICER"


class User(BaseDBModel):
    """User account principal across all 8 healthcare personas."""
    __tablename__ = "users"

    email = Column(String(255), unique=True, nullable=False, index=True)
    full_name = Column(String(255), nullable=False)
    password_hash = Column(String(255), nullable=False)
    role = Column(String(50), nullable=False, index=True)
    phone_number = Column(String(50), nullable=True)
    
    # Organization tenancy (null for independent citizens / patients)
    organization_id = Column(String(36), ForeignKey("organizations.id", ondelete="SET NULL"), nullable=True)
    
    # Status and Security
    is_active = Column(Boolean, default=True, nullable=False)
    is_verified = Column(Boolean, default=False, nullable=False)
    failed_login_count = Column(Integer, default=0, nullable=False)
    locked_until = Column(DateTime(timezone=True), nullable=True)
    mfa_enabled = Column(Boolean, default=False, nullable=False)
    
    organization = relationship("Organization", back_populates="users")
    consents = relationship("ConsentRecord", foreign_keys="ConsentRecord.user_id", back_populates="user")
    screenings_as_patient = relationship("ScreeningEncounter", foreign_keys="ScreeningEncounter.patient_id", back_populates="patient")
