"""
Genomic Intelligence Data Model for WELLWATCH PREVENT.
Maintains authorized, structured genomic variants aligned with the Emirati Genome Program (EGP)
and enforce the VUS (Variant of Uncertain Significance) non-actionable safety invariant.
"""
from datetime import datetime
import enum
import uuid
from sqlalchemy import Column, String, DateTime, Enum as SQLEnum, ForeignKey, Text, Index
from sqlalchemy.orm import relationship

from backend.app.core.database import Base


class VariantClassification(str, enum.Enum):
    PATHOGENIC = "PATHOGENIC"
    LIKELY_PATHOGENIC = "LIKELY_PATHOGENIC"
    VUS = "VUS"  # Variant of Uncertain Significance
    LIKELY_BENIGN = "LIKELY_BENIGN"
    BENIGN = "BENIGN"


class GenomicReviewStatus(str, enum.Enum):
    PENDING_REVIEW = "PENDING_REVIEW"
    GENETIC_COUNSELOR_VERIFIED = "GENETIC_COUNSELOR_VERIFIED"
    REJECTED = "REJECTED"


class GenomicFinding(Base):
    __tablename__ = "genomic_findings"

    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    patient_id = Column(String(36), ForeignKey("persons.id", ondelete="CASCADE"), nullable=False, index=True)
    gene_symbol = Column(String(64), nullable=False, index=True)
    variant_id = Column(String(128), nullable=False)  # dbSNP rsID or HGVS string
    genomic_coordinates = Column(String(128), nullable=True)  # e.g. chr19:11200030:C>T
    variant_classification = Column(SQLEnum(VariantClassification), nullable=False, index=True)
    clinical_interpretation = Column(Text, nullable=False)
    actionable_guidance = Column(Text, nullable=True)
    source_laboratory = Column(String(255), nullable=False)
    report_reference = Column(String(128), nullable=True)
    provenance_id = Column(String(36), ForeignKey("data_provenances.id"), nullable=True)
    review_status = Column(SQLEnum(GenomicReviewStatus), default=GenomicReviewStatus.PENDING_REVIEW, nullable=False)
    reviewed_by_user_id = Column(String(36), ForeignKey("users.id"), nullable=True)
    reviewed_at = Column(DateTime, nullable=True)
    reviewer_notes = Column(Text, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)

    # Relationships
    patient = relationship("Person", backref="genomic_findings")
    provenance = relationship("DataProvenance")
    reviewer = relationship("User")

    __table_args__ = (
        Index("ix_genomic_findings_patient_gene", "patient_id", "gene_symbol"),
    )
