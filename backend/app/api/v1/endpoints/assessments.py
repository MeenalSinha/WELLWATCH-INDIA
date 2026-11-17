import json
from datetime import date
from typing import Any, Dict, List, Optional
from fastapi import APIRouter, Depends, HTTPException, Query, status
from pydantic import BaseModel
from sqlalchemy.orm import Session

from backend.app.core.database import get_db
from backend.app.models.assessment import HumanReviewStatus
from backend.app.models.user import User, UserRole
from backend.app.security.rbac import get_current_user
from backend.app.services.person_service import PersonService
from backend.app.services.preventive_assessment_service import PreventiveAssessmentService

router = APIRouter()


class ReviewSubmission(BaseModel):
    review_status: str                  # "APPROVED", "REJECTED", "MORE_INFO_REQUIRED"
    notes: str


@router.post("/generate/{patient_id}", status_code=status.HTTP_201_CREATED)
def generate_assessment(
    patient_id: str,
    pathway_id: Optional[str] = Query(None),
    eval_date: Optional[date] = Query(None),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    Generates an explainable, multi-factor preventive assessment.
    Persists an immutable historical snapshot.
    """
    try:
        person = PersonService.get_person_by_id(
            db=db,
            person_id=patient_id,
            requesting_user_id=current_user.id,
            requesting_role=current_user.role,
            requesting_org_id=current_user.organization_id
        )
    except PermissionError as e:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail=str(e))

    if not person:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Patient not found.")

    try:
        assessment = PreventiveAssessmentService.generate_assessment(
            db=db,
            person_id=person.id,
            pathway_id=pathway_id,
            evaluation_date=eval_date,
            requesting_user_id=current_user.id
        )
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))

    return {
        "assessment_id": assessment.id,
        "patient_id": assessment.patient_id,
        "population_category": assessment.population_category,
        "pathway_id": assessment.pathway_id,
        "pathway_version": assessment.pathway_version,
        "assessment_status": assessment.assessment_status.value,
        "clinical_risk_tier": assessment.clinical_risk_tier.value,
        "human_review_status": assessment.human_review_status.value,
        "recommended_action": assessment.recommended_action,
        "eligibility": json.loads(assessment.eligibility_summary),
        "care_gaps": json.loads(assessment.care_gaps_summary),
        "contributing_factors": json.loads(assessment.contributing_factors),
        "data_quality_limitations": json.loads(assessment.data_quality_limitations),
        "created_at": assessment.created_at.isoformat()
    }


@router.get("/{assessment_id}")
def get_assessment(
    assessment_id: str,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Retrieves full explainable assessment details by ID."""
    from backend.app.models.assessment import PreventiveAssessment
    assessment = db.query(PreventiveAssessment).filter(PreventiveAssessment.id == assessment_id).first()
    if not assessment:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Assessment not found.")

    # Check access to the patient
    try:
        PersonService.get_person_by_id(
            db=db,
            person_id=assessment.patient_id,
            requesting_user_id=current_user.id,
            requesting_role=current_user.role,
            requesting_org_id=current_user.organization_id
        )
    except PermissionError as e:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail=str(e))

    return {
        "assessment_id": assessment.id,
        "patient_id": assessment.patient_id,
        "population_category": assessment.population_category,
        "pathway_id": assessment.pathway_id,
        "pathway_version": assessment.pathway_version,
        "assessment_status": assessment.assessment_status.value,
        "clinical_risk_tier": assessment.clinical_risk_tier.value,
        "human_review_status": assessment.human_review_status.value,
        "recommended_action": assessment.recommended_action,
        "reviewed_by_user_id": assessment.reviewed_by_user_id,
        "reviewed_at": assessment.reviewed_at.isoformat() if assessment.reviewed_at else None,
        "reviewer_notes": assessment.reviewer_notes,
        "eligibility": json.loads(assessment.eligibility_summary),
        "care_gaps": json.loads(assessment.care_gaps_summary),
        "contributing_factors": json.loads(assessment.contributing_factors),
        "data_quality_limitations": json.loads(assessment.data_quality_limitations),
        "created_at": assessment.created_at.isoformat()
    }


@router.get("/patient/{patient_id}")
def list_patient_assessments(
    patient_id: str,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Retrieves all historical assessments for a patient in reverse chronological order."""
    from backend.app.models.assessment import PreventiveAssessment
    try:
        PersonService.get_person_by_id(
            db=db,
            person_id=patient_id,
            requesting_user_id=current_user.id,
            requesting_role=current_user.role,
            requesting_org_id=current_user.organization_id
        )
    except PermissionError as e:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail=str(e))

    records = db.query(PreventiveAssessment).filter(
        PreventiveAssessment.patient_id == patient_id
    ).order_by(PreventiveAssessment.created_at.desc()).all()

    return [
        {
            "assessment_id": a.id,
            "pathway_id": a.pathway_id,
            "pathway_version": a.pathway_version,
            "assessment_status": a.assessment_status.value,
            "clinical_risk_tier": a.clinical_risk_tier.value,
            "human_review_status": a.human_review_status.value,
            "recommended_action": a.recommended_action,
            "created_at": a.created_at.isoformat()
        }
        for a in records
    ]


@router.post("/{assessment_id}/review")
def submit_review(
    assessment_id: str,
    payload: ReviewSubmission,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    Submits authorized human clinical review.
    Restricted to Healthcare Providers (Physicians).
    """
    if current_user.role != UserRole.HEALTHCARE_PROVIDER.value:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Forbidden: Only qualified healthcare providers can submit clinical reviews."
        )

    status_map = {
        "APPROVED": HumanReviewStatus.APPROVED,
        "REJECTED": HumanReviewStatus.REJECTED,
        "MORE_INFO_REQUIRED": HumanReviewStatus.MORE_INFO_REQUIRED,
    }
    if payload.review_status not in status_map:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=f"Invalid review status: {payload.review_status}")

    try:
        updated = PreventiveAssessmentService.submit_clinical_review(
            db=db,
            assessment_id=assessment_id,
            reviewer_user=current_user,
            new_status=status_map[payload.review_status],
            notes=payload.notes
        )
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(e))

    return {
        "success": True,
        "assessment_id": updated.id,
        "human_review_status": updated.human_review_status.value,
        "reviewed_at": updated.reviewed_at.isoformat()
    }
