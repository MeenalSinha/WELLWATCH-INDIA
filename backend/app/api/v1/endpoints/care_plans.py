from typing import List, Optional
from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy.orm import Session

from backend.app.core.database import get_db
from backend.app.security.rbac import get_current_user
from backend.app.models.user import User
from backend.app.schemas.recommendation import (
    CarePlanCreate,
    CarePlanResponse,
)
from backend.app.services.recommendation_service import RecommendationService

router = APIRouter()


@router.post("", response_model=CarePlanResponse, status_code=status.HTTP_201_CREATED)
def create_care_plan(
    payload: CarePlanCreate,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """
    Assemble and persist a structured PreventiveCarePlan linking active recommendations.
    """
    care_plan = RecommendationService.create_care_plan(
        db=db,
        person_id=payload.person_id,
        plan_name=payload.plan_name,
        pathway_id=payload.pathway_id,
        goals=payload.goals,
        assigned_care_team=payload.assigned_care_team,
        recommendation_ids=payload.recommendation_ids,
        current_user=current_user,
        target_date=payload.target_date,
        review_schedule=payload.review_schedule,
    )
    return care_plan


@router.get("/patient/{patient_id}", response_model=List[CarePlanResponse])
def get_patient_care_plans(
    patient_id: str,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """
    Retrieve all preventive care plans for a specific patient.
    """
    return RecommendationService.get_care_plans_for_patient(
        db=db,
        person_id=patient_id,
        current_user=current_user,
    )
