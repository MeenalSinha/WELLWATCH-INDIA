import enum
from datetime import datetime
from typing import Optional, List, Dict, Any
from sqlalchemy import (
    Column,
    String,
    Boolean,
    DateTime,
    ForeignKey,
    Enum as SQLEnum,
    Text,
    JSON,
    Index,
)
from sqlalchemy.orm import relationship
from backend.app.models.base import BaseDBModel


class RecommendationCategory(str, enum.Enum):
    WELL_CHILD_VISIT = "WELL_CHILD_VISIT"
    DEVELOPMENTAL_SCREENING = "DEVELOPMENTAL_SCREENING"
    ADULT_COMPREHENSIVE_SCREENING = "ADULT_COMPREHENSIVE_SCREENING"
    PREVENTIVE_LABORATORY_SCREENING = "PREVENTIVE_LABORATORY_SCREENING"
    CARDIOMETABOLIC_SCREENING = "CARDIOMETABOLIC_SCREENING"
    FAMILY_HISTORY_REVIEW = "FAMILY_HISTORY_REVIEW"
    GENOMIC_COUNSELING = "GENOMIC_COUNSELING"
    LIFESTYLE_WELLNESS = "LIFESTYLE_WELLNESS"
    SPECIALIST_REFERRAL = "SPECIALIST_REFERRAL"
    OVERDUE_RECALL = "OVERDUE_RECALL"
    ADDITIONAL_INFO_COLLECTION = "ADDITIONAL_INFO_COLLECTION"
    CLINICAL_REASSESSMENT = "CLINICAL_REASSESSMENT"


class RecommendationPriority(str, enum.Enum):
    CRITICAL = "CRITICAL"
    HIGH = "HIGH"
    MEDIUM = "MEDIUM"
    ROUTINE = "ROUTINE"


class RecommendationLifecycleStatus(str, enum.Enum):
    DRAFT = "DRAFT"
    PENDING_CLINICAL_REVIEW = "PENDING_CLINICAL_REVIEW"
    APPROVED = "APPROVED"
    MODIFIED = "MODIFIED"
    OFFERED = "OFFERED"
    ACCEPTED = "ACCEPTED"
    DECLINED = "DECLINED"
    SCHEDULED = "SCHEDULED"
    IN_PROGRESS = "IN_PROGRESS"
    COMPLETED = "COMPLETED"
    EXPIRED = "EXPIRED"
    CANCELLED = "CANCELLED"
    SUPERSEDED = "SUPERSEDED"


class PreventiveRecommendation(BaseDBModel):
    __tablename__ = "preventive_recommendations"

    organization_id = Column(String(36), ForeignKey("organizations.id"), nullable=False, index=True)
    person_id = Column(String(36), ForeignKey("persons.id"), nullable=False, index=True)
    assessment_id = Column(String(36), ForeignKey("preventive_assessments.id"), nullable=True, index=True)
    pathway_id = Column(String(100), nullable=False, index=True)
    care_gap_id = Column(String(100), nullable=True, index=True)
    
    category = Column(SQLEnum(RecommendationCategory), nullable=False)
    priority = Column(SQLEnum(RecommendationPriority), nullable=False, default=RecommendationPriority.ROUTINE)
    lifecycle_status = Column(
        SQLEnum(RecommendationLifecycleStatus),
        nullable=False,
        default=RecommendationLifecycleStatus.PENDING_CLINICAL_REVIEW,
        index=True,
    )
    
    recommended_action = Column(String(255), nullable=False)
    clinical_rationale = Column(Text, nullable=False)
    supporting_evidence = Column(JSON, nullable=False, default=dict)
    source_guideline = Column(String(255), nullable=False)
    uncertainty_status = Column(String(50), nullable=False, default="CONFIRMED")
    rule_id = Column(String(100), nullable=False)
    rule_version = Column(String(50), nullable=False)
    
    target_date = Column(DateTime, nullable=True)
    expiration_date = Column(DateTime, nullable=True)
    requires_human_review = Column(Boolean, nullable=False, default=True)
    
    declined_reason = Column(String(255), nullable=True)
    assigned_provider_id = Column(String(36), ForeignKey("users.id"), nullable=True)
    reviewer_notes = Column(Text, nullable=True)

    # Relationships
    person = relationship("Person", backref="recommendations")
    organization = relationship("Organization")
    reviews = relationship("RecommendationReview", back_populates="recommendation", cascade="all, delete-orphan")

    __table_args__ = (
        Index("ix_rec_person_status", "person_id", "lifecycle_status"),
        Index("ix_rec_org_priority", "organization_id", "priority"),
    )


class RecommendationReview(BaseDBModel):
    __tablename__ = "recommendation_reviews"

    recommendation_id = Column(String(36), ForeignKey("preventive_recommendations.id"), nullable=False, index=True)
    reviewer_id = Column(String(36), ForeignKey("users.id"), nullable=False, index=True)
    review_decision = Column(String(50), nullable=False)  # APPROVED, MODIFIED, REJECTED, INFO_REQUESTED
    rationale = Column(Text, nullable=False)
    modifications = Column(JSON, nullable=True)

    # Relationships
    recommendation = relationship("PreventiveRecommendation", back_populates="reviews")
    reviewer = relationship("User")


class PreventiveCarePlan(BaseDBModel):
    __tablename__ = "preventive_care_plans"

    organization_id = Column(String(36), ForeignKey("organizations.id"), nullable=False, index=True)
    person_id = Column(String(36), ForeignKey("persons.id"), nullable=False, index=True)
    plan_name = Column(String(255), nullable=False)
    pathway_id = Column(String(100), nullable=False)
    status = Column(String(50), nullable=False, default="ACTIVE")  # ACTIVE, COMPLETED, ON_HOLD, CANCELLED
    goals = Column(JSON, nullable=False, default=list)
    responsible_org_id = Column(String(36), ForeignKey("organizations.id"), nullable=False)
    assigned_care_team = Column(JSON, nullable=False, default=list)  # List of user_ids and roles
    target_date = Column(DateTime, nullable=True)
    review_schedule = Column(String(100), nullable=True)  # e.g., "EVERY_6_MONTHS"

    # Relationships
    person = relationship("Person", backref="care_plans")
    items = relationship("CarePlanItem", back_populates="care_plan", cascade="all, delete-orphan")


class CarePlanItem(BaseDBModel):
    __tablename__ = "care_plan_items"

    care_plan_id = Column(String(36), ForeignKey("preventive_care_plans.id"), nullable=False, index=True)
    recommendation_id = Column(String(36), ForeignKey("preventive_recommendations.id"), nullable=False, index=True)
    status = Column(String(50), nullable=False, default="PLANNED")  # PLANNED, IN_PROGRESS, COMPLETED, DEFERRED
    target_date = Column(DateTime, nullable=True)
    completed_at = Column(DateTime, nullable=True)

    # Relationships
    care_plan = relationship("PreventiveCarePlan", back_populates="items")
    recommendation = relationship("PreventiveRecommendation")
