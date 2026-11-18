import enum
from datetime import datetime
from typing import Optional
from sqlalchemy import String, DateTime, ForeignKey, Text, Boolean, Integer, Index
from sqlalchemy.orm import Mapped, mapped_column
from backend.app.models.base import BaseDBModel


class ProviderType(str, enum.Enum):
    HIE = "HIE"
    EMR = "EMR"
    LAB = "LAB"
    IDENTITY = "IDENTITY"
    SCHEDULING = "SCHEDULING"


class IntegrationStatus(str, enum.Enum):
    NOT_CONFIGURED = "NOT_CONFIGURED"
    CONFIGURED = "CONFIGURED"
    CONNECTED = "CONNECTED"
    BLOCKED = "BLOCKED"
    ERROR = "ERROR"


class JobExecutionStatus(str, enum.Enum):
    SUCCESS = "SUCCESS"
    PARTIAL_FAILURE = "PARTIAL_FAILURE"
    FAILED = "FAILED"
    SKIPPED = "SKIPPED"


class IntegrationProvider(BaseDBModel):
    """External healthcare integration providers (Malaffi, SEHA EMR, Lab Network, UAE PASS)."""
    __tablename__ = "integration_providers"

    name: Mapped[str] = mapped_column(String(50), unique=True, index=True)
    display_name: Mapped[str] = mapped_column(String(100))
    provider_type: Mapped[str] = mapped_column(String(20), index=True)
    endpoint_url: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)
    auth_type: Mapped[str] = mapped_column(String(20), default="OAUTH2")
    is_enabled: Mapped[bool] = mapped_column(Boolean, default=False)
    status: Mapped[str] = mapped_column(String(20), default=IntegrationStatus.NOT_CONFIGURED)
    configuration_json: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    last_heartbeat_at: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)


class IntegrationJobLog(BaseDBModel):
    """Audit log of inbound and outbound data exchanges with external healthcare providers."""
    __tablename__ = "integration_job_logs"

    provider_id: Mapped[str] = mapped_column(String(36), ForeignKey("integration_providers.id"), index=True)
    job_type: Mapped[str] = mapped_column(String(50), index=True)  # INBOUND_INGESTION, OUTBOUND_EXPORT, HEALTH_CHECK
    status: Mapped[str] = mapped_column(String(20), default=JobExecutionStatus.SUCCESS)
    records_processed: Mapped[int] = mapped_column(Integer, default=0)
    records_failed: Mapped[int] = mapped_column(Integer, default=0)
    details_json: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    error_message: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    executed_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, index=True)


class ProvenanceRecord(BaseDBModel):
    """FHIR-compliant clinical data provenance for tracking external sources and integrity."""
    __tablename__ = "provenance_records"

    target_resource_type: Mapped[str] = mapped_column(String(50), index=True)
    target_resource_id: Mapped[str] = mapped_column(String(36), index=True)
    source_system: Mapped[str] = mapped_column(String(100), index=True)
    source_identifier: Mapped[Optional[str]] = mapped_column(String(100), nullable=True)
    agent_id: Mapped[str] = mapped_column(String(100))
    agent_type: Mapped[str] = mapped_column(String(50), default="INTEGRATION_ADAPTER")
    signature_hash: Mapped[Optional[str]] = mapped_column(String(128), nullable=True)
    recorded_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, index=True)
