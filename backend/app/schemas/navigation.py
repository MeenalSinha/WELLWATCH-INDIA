from datetime import datetime
from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field, ConfigDict
from backend.app.models.navigation import (
    AppointmentStatus,
    ReferralStatus,
    FollowUpStatus,
    RecallStatus,
)


# Appointment Schemas
class AppointmentCreate(BaseModel):
    person_id: str
    recommendation_id: Optional[str] = None
    care_plan_id: Optional[str] = None
    service_type: str = "WELL_CHILD_VISIT"
    scheduled_time: datetime
    location: Optional[str] = "Primary Care Clinic - Abu Dhabi"
    provider_id: Optional[str] = None


class AppointmentStatusUpdate(BaseModel):
    status: AppointmentStatus
    attendance_notes: Optional[str] = None


class AppointmentResponse(BaseModel):
    id: str
    organization_id: str
    person_id: str
    recommendation_id: Optional[str] = None
    care_plan_id: Optional[str] = None
    service_type: str
    scheduled_time: datetime
    location: str
    provider_id: Optional[str] = None
    status: AppointmentStatus
    attendance_notes: Optional[str] = None
    created_at: datetime
    updated_at: datetime

    model_config = ConfigDict(from_attributes=True)


# Referral Schemas
class ReferralCreate(BaseModel):
    person_id: str
    recommendation_id: Optional[str] = None
    receiving_org_id: Optional[str] = None
    specialty: str
    clinical_indication: str
    priority: Optional[str] = "ROUTINE"


class ReferralStatusUpdate(BaseModel):
    status: ReferralStatus
    outcome_notes: Optional[str] = None


class ReferralResponse(BaseModel):
    id: str
    organization_id: str
    person_id: str
    recommendation_id: Optional[str] = None
    referring_provider_id: str
    receiving_org_id: Optional[str] = None
    specialty: str
    clinical_indication: str
    priority: str
    status: ReferralStatus
    outcome_notes: Optional[str] = None
    created_at: datetime
    updated_at: datetime

    model_config = ConfigDict(from_attributes=True)


# Follow-up Schemas
class FollowUpTaskCreate(BaseModel):
    person_id: str
    recommendation_id: Optional[str] = None
    appointment_id: Optional[str] = None
    referral_id: Optional[str] = None
    trigger_reason: str
    assigned_role: Optional[str] = "CARE_COORDINATOR"
    assigned_user_id: Optional[str] = None
    due_date: datetime
    priority: Optional[str] = "MEDIUM"


class FollowUpResolveRequest(BaseModel):
    resolution_notes: str = Field(..., min_length=5)
    resolution_evidence_id: Optional[str] = None


class FollowUpTaskResponse(BaseModel):
    id: str
    organization_id: str
    person_id: str
    recommendation_id: Optional[str] = None
    appointment_id: Optional[str] = None
    referral_id: Optional[str] = None
    trigger_reason: str
    assigned_role: str
    assigned_user_id: Optional[str] = None
    due_date: datetime
    priority: str
    status: FollowUpStatus
    resolution_notes: Optional[str] = None
    resolution_evidence_id: Optional[str] = None
    created_at: datetime
    updated_at: datetime

    model_config = ConfigDict(from_attributes=True)


# Recall Schemas
class RecallTaskCreate(BaseModel):
    person_id: str
    pathway_id: str
    screening_type: str
    due_date: datetime


class RecallStatusUpdate(BaseModel):
    status: RecallStatus


class RecallTaskResponse(BaseModel):
    id: str
    organization_id: str
    person_id: str
    pathway_id: str
    screening_type: str
    due_date: datetime
    status: RecallStatus
    reminder_count: int
    last_reminded_at: Optional[datetime] = None
    created_at: datetime
    updated_at: datetime

    model_config = ConfigDict(from_attributes=True)


# Care Gap Closure Evidence
class CareGapClosureRequest(BaseModel):
    person_id: str
    care_gap_id: str
    recommendation_id: Optional[str] = None
    evidence_source: str = Field(..., description="SCREENING_RECORD, ENCOUNTER, LAB_RESULT, REFERRAL_OUTCOME")
    evidence_id: str
    verification_method: str = Field("CLINICIAN_ATTESTATION", description="AUTOMATED_OBSERVATION_MATCH, CLINICIAN_ATTESTATION")
    verification_notes: Optional[str] = None


class CareGapClosureResponse(BaseModel):
    id: str
    organization_id: str
    person_id: str
    care_gap_id: str
    recommendation_id: Optional[str] = None
    evidence_source: str
    evidence_id: str
    verification_method: str
    verifying_user_id: str
    verification_notes: Optional[str] = None
    verified_at: datetime
    is_valid: bool
    status: str = "CLOSED"

    model_config = ConfigDict(from_attributes=True)


# Analytics Response
class NavigationAnalyticsResponse(BaseModel):
    total_appointments: int
    attended_appointments: int
    no_show_appointments: int
    attendance_rate: float
    total_referrals: int
    completed_referrals: int
    active_follow_up_tasks: int
    closed_care_gaps_count: int
    data_source_note: str
