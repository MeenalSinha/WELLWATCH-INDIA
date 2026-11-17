from typing import Optional
from fastapi import APIRouter, Depends, HTTPException, Query, Request, status
from sqlalchemy.orm import Session

from backend.app.core.database import get_db
from backend.app.security.rbac import get_current_user
from backend.app.models.clinical import ObservationType
from backend.app.models.user import User, UserRole
from backend.app.schemas.person import (
    AdultProfileResponse,
    ObservationCreate,
    PediatricProfileResponse,
    PersonCreate,
    PersonListResponse,
    PersonSummaryResponse,
    TimelineResponse,
)
from backend.app.services.person_service import PersonService

router = APIRouter()


@router.post("", response_model=PersonSummaryResponse, status_code=status.HTTP_201_CREATED)
def register_patient(
    payload: PersonCreate,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    Registers a new healthcare subject (Person).
    Separates healthcare subjects from application login accounts.
    Available to Healthcare Providers, Community Health Workers, and Care Coordinators.
    """
    allowed_roles = [
        UserRole.HEALTHCARE_PROVIDER.value,
        UserRole.COMMUNITY_HEALTH_WORKER.value,
        UserRole.CARE_COORDINATOR.value,
        UserRole.POPULATION_HEALTH_ADMIN.value,
    ]
    if current_user.role not in allowed_roles:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Forbidden: Insufficient permissions to register new patients."
        )

    # Use provider's organization
    org_id = current_user.organization_id
    if not org_id:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="User must be associated with a registered healthcare facility to enroll patients."
        )

    primary_id = None
    if payload.identifier_value:
        primary_id = (payload.identifier_namespace, payload.identifier_value)

    try:
        person = PersonService.register_person(
            db=db,
            first_name=payload.first_name,
            last_name=payload.last_name,
            date_of_birth=payload.date_of_birth,
            gender=payload.gender,
            organization_id=org_id,
            emirate=payload.emirate,
            district=payload.district,
            preferred_language=payload.preferred_language,
            nationality=payload.nationality,
            primary_identifier=primary_id,
            guardian_user_id=payload.guardian_user_id,
            relationship_to_guardian=payload.relationship_to_guardian
        )
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(e))

    return PersonSummaryResponse(
        id=person.id,
        full_name=person.full_name,
        date_of_birth=person.date_of_birth,
        age_years=person.age_years,
        age_months=person.age_months,
        gender=person.gender,
        emirate=person.emirate or "Abu Dhabi",
        population_classification=person.patient_profile.population_classification,
        completeness_score=person.patient_profile.completeness_score,
        completeness_status=person.patient_profile.completeness_status,
        record_status=person.record_status,
        created_at=person.created_at
    )


@router.get("", response_model=PersonListResponse)
def list_patients(
    classification: Optional[str] = Query(None, description="Filter by PEDIATRIC, ADULT, GERIATRIC"),
    q: Optional[str] = Query(None, description="Search query by name or district"),
    skip: int = Query(0, ge=0),
    limit: int = Query(50, ge=1, le=100),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    Lists patients with facility boundary isolation and population filtering.
    """
    try:
        items, total = PersonService.list_persons(
            db=db,
            requesting_role=current_user.role,
            requesting_org_id=current_user.organization_id,
            classification_filter=classification,
            search_query=q,
            skip=skip,
            limit=limit
        )
    except PermissionError as e:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail=str(e))

    response_items = [
        PersonSummaryResponse(
            id=p.id,
            full_name=p.full_name,
            date_of_birth=p.date_of_birth,
            age_years=p.age_years,
            age_months=p.age_months,
            gender=p.gender,
            emirate=p.emirate or "Abu Dhabi",
            population_classification=p.patient_profile.population_classification,
            completeness_score=p.patient_profile.completeness_score,
            completeness_status=p.patient_profile.completeness_status,
            record_status=p.record_status,
            created_at=p.created_at
        )
        for p in items
    ]

    return PersonListResponse(items=response_items, total=total, skip=skip, limit=limit)


@router.get("/{patient_id}", response_model=PersonSummaryResponse)
def get_patient(
    patient_id: str,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Retrieves patient summary enforcing patient authorization boundaries."""
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
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Patient record not found.")

    return PersonSummaryResponse(
        id=person.id,
        full_name=person.full_name,
        date_of_birth=person.date_of_birth,
        age_years=person.age_years,
        age_months=person.age_months,
        gender=person.gender,
        emirate=person.emirate or "Abu Dhabi",
        population_classification=person.patient_profile.population_classification,
        completeness_score=person.patient_profile.completeness_score,
        completeness_status=person.patient_profile.completeness_status,
        record_status=person.record_status,
        created_at=person.created_at
    )


@router.get("/{patient_id}/pediatric", response_model=PediatricProfileResponse)
def get_pediatric_profile(
    patient_id: str,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Retrieves pediatric profile including Well-Child visits and developmental milestones."""
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
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Patient record not found.")

    pediatric_data = PersonService.get_pediatric_profile(db=db, person=person)

    summary = PersonSummaryResponse(
        id=person.id,
        full_name=person.full_name,
        date_of_birth=person.date_of_birth,
        age_years=person.age_years,
        age_months=person.age_months,
        gender=person.gender,
        emirate=person.emirate or "Abu Dhabi",
        population_classification=person.patient_profile.population_classification,
        completeness_score=person.patient_profile.completeness_score,
        completeness_status=person.patient_profile.completeness_status,
        record_status=person.record_status,
        created_at=person.created_at
    )

    return PediatricProfileResponse(person=summary, **pediatric_data)


@router.get("/{patient_id}/adult", response_model=AdultProfileResponse)
def get_adult_profile(
    patient_id: str,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Retrieves adult profile including vitals, chronic conditions, lifestyle, and IFHAS screening history."""
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
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Patient record not found.")

    adult_data = PersonService.get_adult_profile(db=db, person=person)

    summary = PersonSummaryResponse(
        id=person.id,
        full_name=person.full_name,
        date_of_birth=person.date_of_birth,
        age_years=person.age_years,
        age_months=person.age_months,
        gender=person.gender,
        emirate=person.emirate or "Abu Dhabi",
        population_classification=person.patient_profile.population_classification,
        completeness_score=person.patient_profile.completeness_score,
        completeness_status=person.patient_profile.completeness_status,
        record_status=person.record_status,
        created_at=person.created_at
    )

    return AdultProfileResponse(person=summary, **adult_data)


@router.get("/{patient_id}/timeline", response_model=TimelineResponse)
def get_patient_timeline(
    patient_id: str,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Retrieves unified longitudinal timeline of all encounters, observations, and screenings."""
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
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Patient record not found.")

    events = PersonService.get_longitudinal_timeline(db=db, person_id=person.id)
    return TimelineResponse(patient_id=person.id, total_events=len(events), events=events)


@router.post("/{patient_id}/observations", status_code=status.HTTP_201_CREATED)
def add_clinical_observation(
    patient_id: str,
    payload: ObservationCreate,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Records a structured clinical observation for a patient."""
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
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Patient record not found.")

    obs_type = ObservationType.VITAL_SIGN
    if payload.observation_type in ObservationType.__members__:
        obs_type = ObservationType[payload.observation_type]

    try:
        obs = PersonService.record_clinical_observation(
            db=db,
            person_id=person.id,
            observation_type=obs_type,
            code=payload.code,
            display_name=payload.display_name,
            numeric_value=payload.numeric_value,
            unit=payload.unit,
            string_value=payload.string_value,
            collection_time=payload.collection_time,
            processor_user_id=current_user.id
        )
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(e))

    return {
        "success": True,
        "observation_id": obs.id,
        "code": obs.code,
        "canonical_value": obs.canonical_value,
        "canonical_unit": obs.canonical_unit,
        "epistemic_status": obs.epistemic_status.value
    }
