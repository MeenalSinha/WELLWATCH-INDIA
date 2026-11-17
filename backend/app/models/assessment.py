import enum
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
from sqlalchemy import Column, DateTime, Enum, ForeignKey, Index, String, Text
from sqlalchemy.orm import relationship
from backend.app.models.base import BaseDBModel


class AssessmentStatus(str, enum.Enum):
    COMPLETED = "COMPLETED"
    REQUIRES_REVIEW = "REQUIRES_REVIEW"
    NEEDS_DATA = "NEEDS_DATA"
    INSUFFICIENT_EVIDENCE = "INSUFFICIENT_EVIDENCE"
    NOT_EVALUATED = "NOT_EVALUATED"
    BLOCKED_BY_AUTHORIZATION = "BLOCKED_BY_AUTHORIZATION"


class ClinicalRiskTier(str, enum.Enum):
    LOW_RISK = "LOW_RISK"
    MODERATE_RISK = "MODERATE_RISK"
    ELEVATED_RISK = "ELEVATED_RISK"
    INSUFFICIENT_DATA = "INSUFFICIENT_DATA"


class HumanReviewStatus(str, enum.Enum):
    AWAITING_REVIEW = "AWAITING_REVIEW"
    REVIEWED = "REVIEWED"
    APPROVED = "APPROVED"
    REJECTED = "REJECTED"
    MORE_INFO_REQUIRED = "MORE_INFO_REQUIRED"


class PreventiveAssessment(BaseDBModel):
    """
    Immutable historical snapshot of an explainable preventive assessment.
    Strictly separates:
      1. Clinical Risk Tier
      2. Preventive Program Eligibility
      3. Care Gap Findings
      4. Data Confidence & Limitations
      5. Human Clinical Review State
    """
    __tablename__ = "preventive_assessments"

    patient_id = Column(String(36), ForeignKey("persons.id", ondelete="CASCADE"), nullable=False, index=True)
    assessment_type = Column(String(100), default="COMPREHENSIVE_PREVENTIVE_ASSESSMENT", nullable=False)
    population_category = Column(String(50), nullable=False, index=True)   # PEDIATRIC or ADULT
    
    pathway_id = Column(String(100), nullable=False)
    pathway_version = Column(String(50), nullable=False)
    
    assessment_status = Column(Enum(AssessmentStatus), default=AssessmentStatus.COMPLETED, nullable=False, index=True)
    clinical_risk_tier = Column(Enum(ClinicalRiskTier), default=ClinicalRiskTier.INSUFFICIENT_DATA, nullable=False, index=True)
    
    # Structured JSON-encoded audit evidence
    eligibility_summary = Column(Text, nullable=False)                     # JSON string
    care_gaps_summary = Column(Text, nullable=False)                       # JSON string
    contributing_factors = Column(Text, nullable=False)                    # JSON string: contributing clinical indicators with provenance
    data_quality_limitations = Column(Text, nullable=False)                # JSON string: missing, outdated, or conflicting observations
    
    # Human Clinical Review Workflow
    human_review_status = Column(Enum(HumanReviewStatus), default=HumanReviewStatus.AWAITING_REVIEW, nullable=False, index=True)
    reviewed_by_user_id = Column(String(36), ForeignKey("users.id", ondelete="SET NULL"), nullable=True)
    reviewed_at = Column(DateTime(timezone=True), nullable=True)
    reviewer_notes = Column(Text, nullable=True)
    
    recommended_action = Column(String(255), nullable=False)

    person = relationship("Person")
    reviewer = relationship("User")

    __table_args__ = (
        Index("ix_assessments_patient_status", "patient_id", "assessment_status"),
        Index("ix_assessments_review_queue", "human_review_status", "created_at"),
    )
