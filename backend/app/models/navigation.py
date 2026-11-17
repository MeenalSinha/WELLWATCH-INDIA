import enum
from datetime import datetime
from typing import Optional, List, Dict, Any
from sqlalchemy import (
    Column,
    String,
    Boolean,
    Integer,
    DateTime,
    ForeignKey,
    Enum as SQLEnum,
    Text,
    JSON,
    Index,
)
from sqlalchemy.orm import relationship
from backend.app.models.base import BaseDBModel


class AppointmentStatus(str, enum.Enum):
    REQUESTED = "REQUESTED"
    BOOKED = "BOOKED"
    CONFIRMED = "CONFIRMED"
    RESCHEDULED = "RESCHEDULED"
    ATTENDED = "ATTENDED"
    NO_SHOW = "NO_SHOW"
    CANCELLED = "CANCELLED"
    COMPLETED = "COMPLETED"


class ReferralStatus(str, enum.Enum):
    DRAFT = "DRAFT"
    SUBMITTED = "SUBMITTED"
    ACCEPTED = "ACCEPTED"
    SCHEDULED = "SCHEDULED"
    COMPLETED = "COMPLETED"
    REJECTED = "REJECTED"
    CANCELLED = "CANCELLED"


class FollowUpStatus(str, enum.Enum):
    PENDING = "PENDING"
    ASSIGNED = "ASSIGNED"
    IN_PROGRESS = "IN_PROGRESS"
    RESOLVED = "RESOLVED"
    ESCALATED = "ESCALATED"
    CANCELLED = "CANCELLED"


class RecallStatus(str, enum.Enum):
    ACTIVE = "ACTIVE"
    DUE = "DUE"
    OVERDUE = "OVERDUE"
    SCHEDULED = "SCHEDULED"
    COMPLETED = "COMPLETED"
    CANCELLED = "CANCELLED"


class Appointment(BaseDBModel):
    __tablename__ = "appointments"

    organization_id = Column(String(36), ForeignKey("organizations.id"), nullable=False, index=True)
    person_id = Column(String(36), ForeignKey("persons.id"), nullable=False, index=True)
    recommendation_id = Column(String(36), ForeignKey("preventive_recommendations.id"), nullable=True, index=True)
    care_plan_id = Column(String(36), ForeignKey("preventive_care_plans.id"), nullable=True, index=True)
    service_type = Column(String(100), nullable=False)
    scheduled_time = Column(DateTime, nullable=False, index=True)
    location = Column(String(255), nullable=False, default="Primary Care Clinic - Abu Dhabi")
    provider_id = Column(String(36), ForeignKey("users.id"), nullable=True)
    status = Column(SQLEnum(AppointmentStatus), nullable=False, default=AppointmentStatus.BOOKED, index=True)
    attendance_notes = Column(Text, nullable=True)

    # Relationships
    person = relationship("Person", backref="appointments")
    organization = relationship("Organization")
    provider = relationship("User")
    recommendation = relationship("PreventiveRecommendation")


class Referral(BaseDBModel):
    __tablename__ = "referrals"

    organization_id = Column(String(36), ForeignKey("organizations.id"), nullable=False, index=True)
    person_id = Column(String(36), ForeignKey("persons.id"), nullable=False, index=True)
    recommendation_id = Column(String(36), ForeignKey("preventive_recommendations.id"), nullable=True, index=True)
    referring_provider_id = Column(String(36), ForeignKey("users.id"), nullable=False)
    receiving_org_id = Column(String(36), ForeignKey("organizations.id"), nullable=True)
    specialty = Column(String(100), nullable=False)
    clinical_indication = Column(Text, nullable=False)
    priority = Column(String(50), nullable=False, default="ROUTINE")  # ROUTINE, URGENT, EXPEDITED
    status = Column(SQLEnum(ReferralStatus), nullable=False, default=ReferralStatus.SUBMITTED, index=True)
    outcome_notes = Column(Text, nullable=True)

    # Relationships
    person = relationship("Person", backref="referrals")
    referring_provider = relationship("User", foreign_keys=[referring_provider_id])
    receiving_org = relationship("Organization", foreign_keys=[receiving_org_id])
    recommendation = relationship("PreventiveRecommendation")


class FollowUpTask(BaseDBModel):
    __tablename__ = "follow_up_tasks"

    organization_id = Column(String(36), ForeignKey("organizations.id"), nullable=False, index=True)
    person_id = Column(String(36), ForeignKey("persons.id"), nullable=False, index=True)
    recommendation_id = Column(String(36), ForeignKey("preventive_recommendations.id"), nullable=True, index=True)
    appointment_id = Column(String(36), ForeignKey("appointments.id"), nullable=True, index=True)
    referral_id = Column(String(36), ForeignKey("referrals.id"), nullable=True, index=True)
    trigger_reason = Column(String(100), nullable=False)  # e.g., "NO_SHOW", "OVERDUE_SCREENING", "INCOMPLETE_REFERRAL"
    assigned_role = Column(String(50), nullable=False, default="CARE_COORDINATOR")
    assigned_user_id = Column(String(36), ForeignKey("users.id"), nullable=True)
    due_date = Column(DateTime, nullable=False, index=True)
    priority = Column(String(50), nullable=False, default="MEDIUM")
    status = Column(SQLEnum(FollowUpStatus), nullable=False, default=FollowUpStatus.PENDING, index=True)
    resolution_notes = Column(Text, nullable=True)
    resolution_evidence_id = Column(String(100), nullable=True)

    # Relationships
    person = relationship("Person", backref="follow_up_tasks")
    appointment = relationship("Appointment")
    referral = relationship("Referral")
    recommendation = relationship("PreventiveRecommendation")
    assigned_user = relationship("User")


class RecallTask(BaseDBModel):
    __tablename__ = "recall_tasks"

    organization_id = Column(String(36), ForeignKey("organizations.id"), nullable=False, index=True)
    person_id = Column(String(36), ForeignKey("persons.id"), nullable=False, index=True)
    pathway_id = Column(String(100), nullable=False)
    screening_type = Column(String(100), nullable=False)
    due_date = Column(DateTime, nullable=False, index=True)
    status = Column(SQLEnum(RecallStatus), nullable=False, default=RecallStatus.ACTIVE, index=True)
    reminder_count = Column(Integer, nullable=False, default=0)
    last_reminded_at = Column(DateTime, nullable=True)

    # Relationships
    person = relationship("Person", backref="recall_tasks")


class CareGapClosureEvidence(BaseDBModel):
    __tablename__ = "care_gap_closure_evidence"

    organization_id = Column(String(36), ForeignKey("organizations.id"), nullable=False, index=True)
    person_id = Column(String(36), ForeignKey("persons.id"), nullable=False, index=True)
    care_gap_id = Column(String(100), nullable=False, index=True)
    recommendation_id = Column(String(36), ForeignKey("preventive_recommendations.id"), nullable=True, index=True)
    evidence_source = Column(String(100), nullable=False)  # SCREENING_RECORD, ENCOUNTER, LAB_RESULT, REFERRAL_OUTCOME
    evidence_id = Column(String(100), nullable=False)
    verification_method = Column(String(100), nullable=False)  # AUTOMATED_OBSERVATION_MATCH, CLINICIAN_ATTESTATION
    verifying_user_id = Column(String(36), ForeignKey("users.id"), nullable=False)
    verification_notes = Column(Text, nullable=True)
    verified_at = Column(DateTime, nullable=False, default=datetime.utcnow)
    is_valid = Column(Boolean, nullable=False, default=True)

    # Relationships
    person = relationship("Person")
    verifying_user = relationship("User")
    recommendation = relationship("PreventiveRecommendation")
