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


class CommunicationChannel(str, enum.Enum):
    IN_APP = "IN_APP"
    SMS = "SMS"
    EMAIL = "EMAIL"
    WHATSAPP = "WHATSAPP"


class OutreachAuthorizationStatus(str, enum.Enum):
    NOT_PROVIDED = "NOT_PROVIDED"
    PENDING = "PENDING"
    AUTHORIZED = "AUTHORIZED"
    OPTED_OUT = "OPTED_OUT"
    REVOKED = "REVOKED"
    EXPIRED = "EXPIRED"
    NOT_APPLICABLE = "NOT_APPLICABLE"


class OutreachDeliveryStatus(str, enum.Enum):
    DRAFT = "DRAFT"
    PENDING = "PENDING"
    SCHEDULED = "SCHEDULED"
    SENDING = "SENDING"
    SENT = "SENT"
    DELIVERED = "DELIVERED"
    FAILED = "FAILED"
    RETRY_PENDING = "RETRY_PENDING"
    CANCELLED = "CANCELLED"
    SUPPRESSED = "SUPPRESSED"
    EXPIRED = "EXPIRED"


class CommunicationPreference(BaseDBModel):
    __tablename__ = "communication_preferences"

    person_id = Column(String(36), ForeignKey("persons.id"), nullable=False, unique=True, index=True)
    preferred_language = Column(String(10), nullable=False, default="ar")  # "ar" or "en"
    preferred_channel = Column(SQLEnum(CommunicationChannel), nullable=False, default=CommunicationChannel.IN_APP)
    contact_window_start = Column(String(10), nullable=False, default="09:00")
    contact_window_end = Column(String(10), nullable=False, default="19:00")
    quiet_hours_start = Column(String(10), nullable=False, default="21:00")
    quiet_hours_end = Column(String(10), nullable=False, default="08:00")
    consent_opt_in = Column(Boolean, nullable=False, default=False)
    authorization_status = Column(
        SQLEnum(OutreachAuthorizationStatus),
        nullable=False,
        default=OutreachAuthorizationStatus.NOT_PROVIDED,
    )
    opt_out_timestamp = Column(DateTime, nullable=True)
    guardian_proxy_required = Column(Boolean, nullable=False, default=False)
    accessibility_needs = Column(JSON, nullable=True)

    # Relationships
    person = relationship("Person", backref="communication_preference")


class OutreachTask(BaseDBModel):
    __tablename__ = "outreach_tasks"

    organization_id = Column(String(36), ForeignKey("organizations.id"), nullable=False, index=True)
    person_id = Column(String(36), ForeignKey("persons.id"), nullable=False, index=True)
    recommendation_id = Column(String(36), ForeignKey("preventive_recommendations.id"), nullable=True, index=True)
    recipient_id = Column(String(36), nullable=True)
    recipient_role = Column(String(50), nullable=False, default="PATIENT")  # PATIENT, GUARDIAN, CAREGIVER
    channel = Column(SQLEnum(CommunicationChannel), nullable=False, default=CommunicationChannel.IN_APP)
    template_id = Column(String(100), nullable=False)
    language = Column(String(10), nullable=False, default="ar")  # "ar" or "en"
    message_body = Column(Text, nullable=False)
    scheduled_at = Column(DateTime, nullable=False)
    attempted_at = Column(DateTime, nullable=True)
    delivery_status = Column(
        SQLEnum(OutreachDeliveryStatus),
        nullable=False,
        default=OutreachDeliveryStatus.PENDING,
        index=True,
    )
    provider_status = Column(String(100), nullable=False, default="NOT_CONFIGURED")
    provider_message_id = Column(String(100), nullable=True)
    failure_reason = Column(String(255), nullable=True)
    retry_count = Column(Integer, nullable=False, default=0)
    max_retries = Column(Integer, nullable=False, default=3)
    consent_verified = Column(Boolean, nullable=False, default=True)

    # Relationships
    person = relationship("Person", backref="outreach_tasks")
    recommendation = relationship("PreventiveRecommendation", backref="outreach_tasks")
    organization = relationship("Organization")

    __table_args__ = (
        Index("ix_outreach_person_status", "person_id", "delivery_status"),
        Index("ix_outreach_scheduled", "scheduled_at", "delivery_status"),
    )


class InAppNotification(BaseDBModel):
    __tablename__ = "in_app_notifications"

    organization_id = Column(String(36), ForeignKey("organizations.id"), nullable=False, index=True)
    person_id = Column(String(36), ForeignKey("persons.id"), nullable=False, index=True)
    recipient_user_id = Column(String(36), ForeignKey("users.id"), nullable=True, index=True)
    outreach_task_id = Column(String(36), ForeignKey("outreach_tasks.id"), nullable=True, index=True)
    title = Column(String(255), nullable=False)
    message_body = Column(Text, nullable=False)
    is_read = Column(Boolean, nullable=False, default=False, index=True)
    read_at = Column(DateTime, nullable=True)

    # Relationships
    person = relationship("Person", backref="in_app_notifications")
    recipient_user = relationship("User")
    outreach_task = relationship("OutreachTask", backref="in_app_notifications")

