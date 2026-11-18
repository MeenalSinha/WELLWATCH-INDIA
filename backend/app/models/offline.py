import enum
from datetime import datetime
from typing import Optional
from sqlalchemy import String, DateTime, ForeignKey, Text, Integer, Index
from sqlalchemy.orm import Mapped, mapped_column, relationship
from backend.app.models.base import BaseDBModel


class DeviceStatus(str, enum.Enum):
    ACTIVE = "ACTIVE"
    REVOKED = "REVOKED"
    EXPIRED = "EXPIRED"


class OfflineAssignmentStatus(str, enum.Enum):
    ACTIVE = "ACTIVE"
    EXPIRED = "EXPIRED"
    REVOKED = "REVOKED"
    COMPLETED = "COMPLETED"


class SyncOperationStatus(str, enum.Enum):
    PENDING = "PENDING"
    SYNCING = "SYNCING"
    SYNCED = "SYNCED"
    CONFLICT = "CONFLICT"
    REJECTED = "REJECTED"
    FAILED = "FAILED"


class ConflictResolutionStatus(str, enum.Enum):
    PENDING_REVIEW = "PENDING_REVIEW"
    RESOLVED_ACCEPTED = "RESOLVED_ACCEPTED"
    RESOLVED_REJECTED = "RESOLVED_REJECTED"


class DeviceRegistration(BaseDBModel):
    """Device registration for authenticated Community Health Workers performing offline visits."""
    __tablename__ = "device_registrations"

    device_id: Mapped[str] = mapped_column(String(128), index=True)
    user_id: Mapped[str] = mapped_column(String(36), ForeignKey("users.id"), index=True)
    organization_id: Mapped[str] = mapped_column(String(36), ForeignKey("organizations.id"), index=True)
    device_model: Mapped[str] = mapped_column(String(100), default="Generic Tablet/Mobile")
    os_platform: Mapped[str] = mapped_column(String(50), default="WebPWA")
    app_version: Mapped[str] = mapped_column(String(50), default="1.0.0")
    status: Mapped[DeviceStatus] = mapped_column(String(20), default=DeviceStatus.ACTIVE)
    last_sync_at: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)
    registered_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)


class OfflineAssignment(BaseDBModel):
    """Secure time-bounded assignment of patient cohorts to CHWs for authorized offline operations."""
    __tablename__ = "offline_assignments"

    user_id: Mapped[str] = mapped_column(String(36), ForeignKey("users.id"), index=True)
    patient_id: Mapped[str] = mapped_column(String(36), ForeignKey("persons.id"), index=True)
    organization_id: Mapped[str] = mapped_column(String(36), ForeignKey("organizations.id"), index=True)
    assignment_reason: Mapped[str] = mapped_column(String(255), default="FIELD_PREVENTIVE_VISIT")
    assigned_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    expires_at: Mapped[datetime] = mapped_column(DateTime, index=True)
    status: Mapped[OfflineAssignmentStatus] = mapped_column(String(20), default=OfflineAssignmentStatus.ACTIVE)


class SyncOperationAudit(BaseDBModel):
    """Idempotent outbox sync operations audit and replay protection."""
    __tablename__ = "sync_operation_audits"

    operation_id: Mapped[str] = mapped_column(String(64), unique=True, index=True)
    idempotency_key: Mapped[str] = mapped_column(String(128), index=True)
    user_id: Mapped[str] = mapped_column(String(36), ForeignKey("users.id"), index=True)
    device_id: Mapped[Optional[str]] = mapped_column(String(128), nullable=True)
    resource_type: Mapped[str] = mapped_column(String(50), index=True)
    resource_id: Mapped[Optional[str]] = mapped_column(String(36), nullable=True)
    operation_type: Mapped[str] = mapped_column(String(20), default="CREATE")
    status: Mapped[SyncOperationStatus] = mapped_column(String(20), default=SyncOperationStatus.SYNCED)
    payload_snapshot: Mapped[str] = mapped_column(Text)
    client_timestamp: Mapped[datetime] = mapped_column(DateTime)
    server_timestamp: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)


class SyncConflict(BaseDBModel):
    """Tracks clinical data discrepancies between offline submissions and updated server records."""
    __tablename__ = "sync_conflicts"

    operation_id: Mapped[str] = mapped_column(String(64), index=True)
    resource_type: Mapped[str] = mapped_column(String(50), index=True)
    resource_id: Mapped[str] = mapped_column(String(36), index=True)
    patient_id: Mapped[str] = mapped_column(String(36), ForeignKey("persons.id"), index=True)
    user_id: Mapped[str] = mapped_column(String(36), ForeignKey("users.id"), index=True)
    conflict_reason: Mapped[str] = mapped_column(String(255))
    client_payload: Mapped[str] = mapped_column(Text)
    server_payload: Mapped[str] = mapped_column(Text)
    resolution_status: Mapped[ConflictResolutionStatus] = mapped_column(
        String(30), default=ConflictResolutionStatus.PENDING_REVIEW
    )
    resolved_by: Mapped[Optional[str]] = mapped_column(String(36), ForeignKey("users.id"), nullable=True)
    resolution_notes: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    resolved_at: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
