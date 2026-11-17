from datetime import date
from typing import Any, Dict, List, Optional
from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy.orm import Session

from backend.app.core.database import get_db
from backend.app.models.user import User, UserRole
from backend.app.security.rbac import get_current_user
from backend.app.services.pathway_engine import PathwayEngine
from backend.app.services.person_service import PersonService

router = APIRouter()


@router.get("", response_model=List[Dict[str, Any]])
def list_pathways(
    current_user: User = Depends(get_current_user),
):
    """Lists all active and versioned preventive care pathways."""
    return PathwayEngine.list_available_pathways()


@router.get("/{pathway_id}", response_model=Dict[str, Any])
def get_pathway(
    pathway_id: str,
    current_user: User = Depends(get_current_user),
):
    """Retrieves full specification of a versioned preventive care pathway."""
    pathway = PathwayEngine.load_pathway(pathway_id)
    if not pathway:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Preventive pathway '{pathway_id}' not found."
        )
    return pathway


@router.post("/{pathway_id}/evaluate-eligibility/{patient_id}")
def evaluate_patient_eligibility(
    pathway_id: str,
    patient_id: str,
    eval_date: Optional[date] = Query(None, description="Explicit evaluation date for reproducible back-testing"),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Evaluates whether an individual is eligible for a preventive pathway."""
    pathway = PathwayEngine.load_pathway(pathway_id)
    if not pathway:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Pathway not found.")

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

    return PathwayEngine.evaluate_eligibility(pathway=pathway, person=person, evaluation_date=eval_date)


@router.post("/{pathway_id}/evaluate-care-gaps/{patient_id}")
def evaluate_patient_care_gaps(
    pathway_id: str,
    patient_id: str,
    eval_date: Optional[date] = Query(None, description="Explicit evaluation date"),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Evaluates milestone care gaps, distinguishing completed, due, overdue, and uncertain records."""
    pathway = PathwayEngine.load_pathway(pathway_id)
    if not pathway:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Pathway not found.")

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

    gaps = PathwayEngine.evaluate_care_gaps(db=db, pathway=pathway, person=person, evaluation_date=eval_date)
    return {
        "patient_id": person.id,
        "pathway_id": pathway.get("pathway_id"),
        "pathway_version": pathway.get("version"),
        "evaluation_date": (eval_date or date.today()).isoformat(),
        "care_gaps": gaps
    }


@router.post("/{pathway_id}/evaluate-all/{patient_id}")
def evaluate_all(
    pathway_id: str,
    patient_id: str,
    eval_date: Optional[date] = Query(None, description="Explicit evaluation date"),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    Combined deterministic evaluation separating Eligibility from Care Gaps.
    Returns complete explainability, reasoning codes, and authoritative guidelines citations.
    """
    pathway = PathwayEngine.load_pathway(pathway_id)
    if not pathway:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Pathway not found.")

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

    eligibility = PathwayEngine.evaluate_eligibility(pathway=pathway, person=person, evaluation_date=eval_date)
    care_gaps = []
    if eligibility.get("is_eligible"):
        care_gaps = PathwayEngine.evaluate_care_gaps(db=db, pathway=pathway, person=person, evaluation_date=eval_date)

    return {
        "patient_id": person.id,
        "pathway_id": pathway.get("pathway_id"),
        "pathway_name": pathway.get("name"),
        "pathway_version": pathway.get("version"),
        "clinical_authority": pathway.get("publishing_authority"),
        "guideline_reference": pathway.get("guideline_reference"),
        "eligibility": eligibility,
        "care_gaps": care_gaps
    }
