from datetime import datetime
from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field, ConfigDict
from backend.app.models.recommendation import (
    RecommendationCategory,
    RecommendationPriority,
    RecommendationLifecycleStatus,
)


class RecommendationReviewRequest(BaseModel):
    review_decision: str = Field(..., description="APPROVED, MODIFIED, REJECTED, INFO_REQUESTED")
    rationale: str = Field(..., min_length=5)
    modifications: Optional[Dict[str, Any]] = None


class RecommendationStatusUpdate(BaseModel):
    lifecycle_status: RecommendationLifecycleStatus
    declined_reason: Optional[str] = None
    reviewer_notes: Optional[str] = None


class RecommendationResponse(BaseModel):
    id: str
    organization_id: str
    person_id: str
    assessment_id: Optional[str] = None
    pathway_id: str
    care_gap_id: Optional[str] = None
    category: RecommendationCategory
    priority: RecommendationPriority
    lifecycle_status: RecommendationLifecycleStatus
    recommended_action: str
    clinical_rationale: str
    supporting_evidence: Dict[str, Any]
    source_guideline: str
    uncertainty_status: str
    rule_id: str
    rule_version: str
    target_date: Optional[datetime] = None
    expiration_date: Optional[datetime] = None
    requires_human_review: bool
    declined_reason: Optional[str] = None
    assigned_provider_id: Optional[str] = None
    reviewer_notes: Optional[str] = None
    created_at: datetime
    updated_at: datetime

    model_config = ConfigDict(from_attributes=True)


class RecommendationListResponse(BaseModel):
    total: int
    items: List[RecommendationResponse]


class RecommendationExplanation(BaseModel):
    recommendation_id: str
    category: str
    priority: str
    clinical_rationale: str
    source_guideline: str
    trigger_rule_id: str
    trigger_care_gap: Optional[str]
    identified_evidence: Dict[str, Any]
    uncertainty_status: str
    requires_human_review: bool
    justification_summary: str


class CarePlanItemCreate(BaseModel):
    recommendation_id: str
    status: Optional[str] = "PLANNED"
    target_date: Optional[datetime] = None


class CarePlanItemResponse(BaseModel):
    id: str
    care_plan_id: str
    recommendation_id: str
    status: str
    target_date: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    recommendation: Optional[RecommendationResponse] = None

    model_config = ConfigDict(from_attributes=True)


class CarePlanCreate(BaseModel):
    person_id: str
    plan_name: str
    pathway_id: str
    goals: List[Dict[str, Any]] = []
    assigned_care_team: List[Dict[str, Any]] = []
    target_date: Optional[datetime] = None
    review_schedule: Optional[str] = "EVERY_6_MONTHS"
    recommendation_ids: List[str] = []


class CarePlanResponse(BaseModel):
    id: str
    organization_id: str
    person_id: str
    plan_name: str
    pathway_id: str
    status: str
    goals: List[Dict[str, Any]]
    responsible_org_id: str
    assigned_care_team: List[Dict[str, Any]]
    target_date: Optional[datetime] = None
    review_schedule: Optional[str] = None
    created_at: datetime
    updated_at: datetime
    items: List[CarePlanItemResponse] = []

    model_config = ConfigDict(from_attributes=True)
