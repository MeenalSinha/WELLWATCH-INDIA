from datetime import datetime
from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field, ConfigDict
from backend.app.models.outreach import (
    CommunicationChannel,
    OutreachAuthorizationStatus,
    OutreachDeliveryStatus,
)


class CommunicationPreferenceBase(BaseModel):
    preferred_language: str = Field("ar", pattern="^(ar|en)$")
    preferred_channel: CommunicationChannel = CommunicationChannel.IN_APP
    contact_window_start: str = "09:00"
    contact_window_end: str = "19:00"
    quiet_hours_start: str = "21:00"
    quiet_hours_end: str = "08:00"
    consent_opt_in: bool = False
    authorization_status: OutreachAuthorizationStatus = OutreachAuthorizationStatus.NOT_PROVIDED
    guardian_proxy_required: bool = False
    accessibility_needs: Optional[Dict[str, Any]] = None


class CommunicationPreferenceCreate(CommunicationPreferenceBase):
    person_id: str


class CommunicationPreferenceUpdate(BaseModel):
    preferred_language: Optional[str] = Field(None, pattern="^(ar|en)$")
    preferred_channel: Optional[CommunicationChannel] = None
    contact_window_start: Optional[str] = None
    contact_window_end: Optional[str] = None
    quiet_hours_start: Optional[str] = None
    quiet_hours_end: Optional[str] = None
    consent_opt_in: Optional[bool] = None
    guardian_proxy_required: Optional[bool] = None
    accessibility_needs: Optional[Dict[str, Any]] = None


class CommunicationPreferenceResponse(CommunicationPreferenceBase):
    id: str
    person_id: str
    opt_out_timestamp: Optional[datetime] = None
    created_at: datetime
    updated_at: datetime

    model_config = ConfigDict(from_attributes=True)


class OutreachTaskCreate(BaseModel):
    person_id: str
    recommendation_id: Optional[str] = None
    channel: Optional[CommunicationChannel] = None  # None = use preferred channel
    template_id: str
    scheduled_at: Optional[datetime] = None  # None = immediate
    recipient_role: Optional[str] = "PATIENT"


class OutreachTaskResponse(BaseModel):
    id: str
    organization_id: str
    person_id: str
    recommendation_id: Optional[str] = None
    recipient_id: Optional[str] = None
    recipient_role: str
    channel: CommunicationChannel
    template_id: str
    language: str
    message_body: str
    scheduled_at: datetime
    attempted_at: Optional[datetime] = None
    delivery_status: OutreachDeliveryStatus
    provider_status: str
    provider_message_id: Optional[str] = None
    failure_reason: Optional[str] = None
    retry_count: int
    max_retries: int
    consent_verified: bool
    created_at: datetime
    updated_at: datetime

    model_config = ConfigDict(from_attributes=True)


class OutreachTaskListResponse(BaseModel):
    total: int
    items: List[OutreachTaskResponse]


class OutreachAnalyticsResponse(BaseModel):
    total_generated: int
    total_attempted: int
    total_delivered: int
    total_failed: int
    total_suppressed: int
    by_channel: Dict[str, int]
    by_language: Dict[str, int]
    active_providers_count: int
    honest_boundary_note: str


class TemplatePreviewRequest(BaseModel):
    template_id: str
    language: str = "ar"
    variables: Dict[str, str] = {}


class TemplatePreviewResponse(BaseModel):
    template_id: str
    language: str
    direction: str  # "rtl" or "ltr"
    rendered_subject: str
    rendered_body: str
    governance_approved: bool


class InAppNotificationResponse(BaseModel):
    id: str
    organization_id: str
    person_id: str
    recipient_user_id: Optional[str] = None
    outreach_task_id: Optional[str] = None
    title: str
    message_body: str
    is_read: bool
    read_at: Optional[datetime] = None
    created_at: datetime

    model_config = ConfigDict(from_attributes=True)
