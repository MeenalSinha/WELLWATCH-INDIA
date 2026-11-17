import enum
from datetime import datetime, timezone
from sqlalchemy import Column, DateTime, Enum, ForeignKey, Index, String, Text
from sqlalchemy.orm import relationship
from backend.app.models.base import BaseDBModel


class SourceType(str, enum.Enum):
    CLINIC_EMR = "CLINIC_EMR"
    COMMUNITY_SCREENING = "COMMUNITY_SCREENING"
    HIE_MALAFFI = "HIE_MALAFFI"
    PATIENT_ENTERED = "PATIENT_ENTERED"
    LAB_SYSTEM = "LAB_SYSTEM"
    GENOMIC_CORE = "GENOMIC_CORE"
    PUBLIC_HEALTH_SURVEILLANCE = "PUBLIC_HEALTH_SURVEILLANCE"


class TrustLevel(str, enum.Enum):
    OFFICIAL_REGULATORY = "OFFICIAL_REGULATORY"    # e.g. DoH / Malaffi
    ACCREDITED_CLINICAL = "ACCREDITED_CLINICAL"    # e.g. Hospital / Clinic EMR
    COMMUNITY_SCREENING = "COMMUNITY_SCREENING"    # e.g. Certified CHW screening
    SELF_REPORTED = "SELF_REPORTED"                # e.g. Patient portal entry


class ConnectionStatus(str, enum.Enum):
    NOT_CONFIGURED = "NOT_CONFIGURED"
    CONFIGURED = "CONFIGURED"
    CONNECTED = "CONNECTED"
    OPERATIONAL = "OPERATIONAL"
    DEGRADED = "DEGRADED"
    FAILED = "FAILED"


class ProvenanceVerificationStatus(str, enum.Enum):
    VERIFIED = "VERIFIED"
    UNVERIFIED = "UNVERIFIED"
    AUDITED = "AUDITED"
    REJECTED = "REJECTED"


class DataSource(BaseDBModel):
    """
    Catalog of authorized clinical data sources and ingestion systems.
    Prevents unverified or simulated external systems from being presented as active.
    """
    __tablename__ = "data_sources"

    source_type = Column(Enum(SourceType), nullable=False, index=True)
    organization_id = Column(String(36), ForeignKey("organizations.id", ondelete="SET NULL"), nullable=True)
    system_name = Column(String(150), nullable=False)
    system_version = Column(String(50), default="1.0", nullable=False)
    trust_level = Column(Enum(TrustLevel), default=TrustLevel.ACCREDITED_CLINICAL, nullable=False)
    connection_status = Column(Enum(ConnectionStatus), default=ConnectionStatus.NOT_CONFIGURED, nullable=False)
    last_sync_at = Column(DateTime(timezone=True), nullable=True)
    endpoint_contract_ref = Column(String(255), nullable=True)

    organization = relationship("Organization")
    provenances = relationship("DataProvenance", back_populates="data_source", cascade="all, delete-orphan")


class DataProvenance(BaseDBModel):
    """
    Cryptographic and operational audit provenance attached to every clinical entity.
    Maintains complete traceability of who, where, when, and how data entered the system.
    """
    __tablename__ = "data_provenances"

    data_source_id = Column(String(36), ForeignKey("data_sources.id", ondelete="RESTRICT"), nullable=False, index=True)
    collection_method = Column(String(100), default="DIRECT_CLINICAL_ENTRY", nullable=False)
    import_timestamp = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc), nullable=False)
    external_record_id = Column(String(150), nullable=True, index=True)
    verification_status = Column(Enum(ProvenanceVerificationStatus), default=ProvenanceVerificationStatus.UNVERIFIED, nullable=False)
    transformation_history = Column(Text, nullable=True)
    responsible_processor_id = Column(String(36), ForeignKey("users.id", ondelete="SET NULL"), nullable=True)

    data_source = relationship("DataSource", back_populates="provenances")
    processor = relationship("User")
