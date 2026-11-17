from typing import Optional, List
from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy.orm import Session

from backend.app.core.database import get_db
from backend.app.security.rbac import get_current_user
from backend.app.models.user import User
from backend.app.models.recommendation import (
    RecommendationLifecycleStatus,
    RecommendationPriority,
)
from backend.app.schemas.recommendation import (
    RecommendationResponse,
    RecommendationListResponse,
    RecommendationReviewRequest,
    RecommendationStatusUpdate,
    RecommendationExplanation,
)
from backend.app.services.recommendation_service import RecommendationService

router = APIRouter()


@router.post("/generate/{patient_id}", response_model=List[RecommendationResponse], status_code=status.HTTP_201_CREATED)
def generate_recommendations(
    patient_id: str,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """
    Generate evidence-backed preventive recommendations for a patient.
    Consumes pathway evaluation, preventive risk assessments, and clinical observations.
    """
    recs = RecommendationService.generate_recommendations_for_patient(
        db=db,
        person_id=patient_id,
        current_user=current_user,
    )
    return recs


@router.get("", response_model=RecommendationListResponse)
def list_recommendations(
    person_id: Optional[str] = Query(None, description="Filter by patient ID"),
    status: Optional[RecommendationLifecycleStatus] = Query(None, description="Filter by lifecycle status"),
    priority: Optional[RecommendationPriority] = Query(None, description="Filter by priority"),
    limit: int = Query(50, ge=1, le=200),
    offset: int = Query(0, ge=0),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """
    Retrieve recommendations with pagination, filtering by status and priority.
    Enforces multi-tenant organizational boundary.
    """
    total, items = RecommendationService.get_recommendations(
        db=db,
        person_id=person_id,
        organization_id=current_user.organization_id if current_user.role.value != "SYSTEM_ADMIN" else None,
        status=status,
        priority=priority,
        limit=limit,
        offset=offset,
    )
    return RecommendationListResponse(total=total, items=items)


@router.get("/{recommendation_id}", response_model=RecommendationResponse)
def get_recommendation(
    recommendation_id: str,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """
    Retrieve single recommendation detail with clinical rationale, supporting evidence, and source guideline.
    """
    return RecommendationService.get_recommendation_by_id(
        db=db,
        recommendation_id=recommendation_id,
        current_user=current_user,
    )


@router.get("/{recommendation_id}/explain", response_model=RecommendationExplanation)
def explain_recommendation(
    recommendation_id: str,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """
    Explainable justification for recommendation: care gap, clinical evidence, pathway rule, and uncertainty.
    """
    rec = RecommendationService.get_recommendation_by_id(
        db=db,
        recommendation_id=recommendation_id,
        current_user=current_user,
    )
    return RecommendationExplanation(
        recommendation_id=rec.id,
        category=rec.category.value,
        priority=rec.priority.value,
        clinical_rationale=rec.clinical_rationale,
        source_guideline=rec.source_guideline,
        trigger_rule_id=rec.rule_id,
        trigger_care_gap=rec.care_gap_id,
        identified_evidence=rec.supporting_evidence or {},
        uncertainty_status=rec.uncertainty_status,
        requires_human_review=rec.requires_human_review,
        justification_summary=(
            f"Generated under {rec.source_guideline} (Rule {rec.rule_id}) for care gap '{rec.care_gap_id or 'GENERAL'}'. "
            f"Priority {rec.priority.value} with clinical confidence status '{rec.uncertainty_status}'."
        ),
    )


@router.post("/{recommendation_id}/review", response_model=RecommendationResponse)
def review_recommendation(
    recommendation_id: str,
    payload: RecommendationReviewRequest,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """
    Physician/Clinical review workflow.
    Approves, modifies, or rejects recommendation. Requires clinical role.
    """
    return RecommendationService.review_recommendation(
        db=db,
        recommendation_id=recommendation_id,
        reviewer_user=current_user,
        review_decision=payload.review_decision,
        rationale=payload.rationale,
        modifications=payload.modifications,
    )


@router.post("/{recommendation_id}/status", response_model=RecommendationResponse)
def update_recommendation_status(
    recommendation_id: str,
    payload: RecommendationStatusUpdate,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """
    Update recommendation lifecycle status (e.g. OFFERED, ACCEPTED, DECLINED).
    Declined transitions require explicit reason.
    """
    return RecommendationService.update_status(
        db=db,
        recommendation_id=recommendation_id,
        current_user=current_user,
        new_status=payload.lifecycle_status,
        declined_reason=payload.declined_reason,
        reviewer_notes=payload.reviewer_notes,
    )
