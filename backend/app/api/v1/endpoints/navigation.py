from typing import Optional, List
from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy.orm import Session

from backend.app.core.database import get_db
from backend.app.security.rbac import get_current_user
from backend.app.models.user import User
from backend.app.models.navigation import (
    Appointment,
    Referral,
    FollowUpTask,
    RecallTask,
    AppointmentStatus,
    ReferralStatus,
    FollowUpStatus,
    RecallStatus,
)
from backend.app.schemas.navigation import (
    AppointmentCreate,
    AppointmentResponse,
    AppointmentStatusUpdate,
    ReferralCreate,
    ReferralResponse,
    ReferralStatusUpdate,
    FollowUpTaskCreate,
    FollowUpTaskResponse,
    FollowUpResolveRequest,
    RecallTaskCreate,
    RecallStatusUpdate,
    RecallTaskResponse,
    CareGapClosureRequest,
    CareGapClosureResponse,
    NavigationAnalyticsResponse,
)
from backend.app.services.navigation_service import NavigationService

router = APIRouter()


# Appointments
@router.post("/appointments", response_model=AppointmentResponse, status_code=status.HTTP_201_CREATED)
def book_appointment(
    payload: AppointmentCreate,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """
    Schedule a preventive care appointment. Links to patient, recommendation, and care plan.
    """
    return NavigationService.book_appointment(
        db=db,
        person_id=payload.person_id,
        service_type=payload.service_type,
        scheduled_time=payload.scheduled_time,
        current_user=current_user,
        recommendation_id=payload.recommendation_id,
        care_plan_id=payload.care_plan_id,
        location=payload.location or "Primary Care Clinic - Abu Dhabi",
        provider_id=payload.provider_id,
    )


@router.get("/appointments", response_model=List[AppointmentResponse])
def list_appointments(
    person_id: Optional[str] = Query(None),
    status: Optional[AppointmentStatus] = Query(None),
    limit: int = Query(50, ge=1, le=200),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """
    List scheduled appointments with attendance status.
    """
    query = db.query(Appointment)
    if person_id:
        query = query.filter(Appointment.person_id == person_id)
    if status:
        query = query.filter(Appointment.status == status)
    user_role = current_user.role.value if hasattr(current_user.role, "value") else str(current_user.role)
    if user_role != "SYSTEM_ADMIN":
        query = query.filter(Appointment.organization_id == current_user.organization_id)
    return query.order_by(Appointment.scheduled_time.desc()).limit(limit).all()


@router.put("/appointments/{appointment_id}/status", response_model=AppointmentResponse)
def update_appointment_status(
    appointment_id: str,
    payload: AppointmentStatusUpdate,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """
    Update appointment attendance. If NO_SHOW is marked, automatically initiates missed-care follow-up.
    """
    return NavigationService.update_appointment_status(
        db=db,
        appointment_id=appointment_id,
        new_status=payload.status,
        current_user=current_user,
        attendance_notes=payload.attendance_notes,
    )


# Referrals
@router.post("/referrals", response_model=ReferralResponse, status_code=status.HTTP_201_CREATED)
def create_referral(
    payload: ReferralCreate,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """
    Issue clinical specialist referral. Enforces clinical role.
    """
    return NavigationService.create_referral(
        db=db,
        person_id=payload.person_id,
        specialty=payload.specialty,
        clinical_indication=payload.clinical_indication,
        current_user=current_user,
        recommendation_id=payload.recommendation_id,
        receiving_org_id=payload.receiving_org_id,
        priority=payload.priority or "ROUTINE",
    )


@router.get("/referrals", response_model=List[ReferralResponse])
def list_referrals(
    person_id: Optional[str] = Query(None),
    status: Optional[ReferralStatus] = Query(None),
    limit: int = Query(50, ge=1, le=200),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """
    List specialist referrals.
    """
    query = db.query(Referral)
    if person_id:
        query = query.filter(Referral.person_id == person_id)
    if status:
        query = query.filter(Referral.status == status)
    user_role = current_user.role.value if hasattr(current_user.role, "value") else str(current_user.role)
    if user_role != "SYSTEM_ADMIN":
        query = query.filter(Referral.organization_id == current_user.organization_id)
    return query.order_by(Referral.created_at.desc()).limit(limit).all()


@router.put("/referrals/{referral_id}/status", response_model=ReferralResponse)
def update_referral_status(
    referral_id: str,
    payload: ReferralStatusUpdate,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """
    Update referral lifecycle. Incomplete/rejected referrals trigger care coordination follow-up.
    """
    return NavigationService.update_referral_status(
        db=db,
        referral_id=referral_id,
        new_status=payload.status,
        current_user=current_user,
        outcome_notes=payload.outcome_notes,
    )


# Follow-up Management
@router.get("/follow-ups", response_model=List[FollowUpTaskResponse])
def list_follow_ups(
    person_id: Optional[str] = Query(None),
    status: Optional[FollowUpStatus] = Query(None),
    limit: int = Query(50, ge=1, le=200),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """
    List automated follow-up tasks for care coordinators and clinicians.
    """
    query = db.query(FollowUpTask)
    if person_id:
        query = query.filter(FollowUpTask.person_id == person_id)
    if status:
        query = query.filter(FollowUpTask.status == status)
    user_role = current_user.role.value if hasattr(current_user.role, "value") else str(current_user.role)
    if user_role != "SYSTEM_ADMIN":
        query = query.filter(FollowUpTask.organization_id == current_user.organization_id)
    return query.order_by(FollowUpTask.due_date.asc()).limit(limit).all()


@router.post("/follow-ups/{task_id}/resolve", response_model=FollowUpTaskResponse)
def resolve_follow_up(
    task_id: str,
    payload: FollowUpResolveRequest,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """
    Resolve follow-up task with resolution notes and audit evidence.
    """
    return NavigationService.resolve_follow_up(
        db=db,
        task_id=task_id,
        resolution_notes=payload.resolution_notes,
        current_user=current_user,
        resolution_evidence_id=payload.resolution_evidence_id,
    )


# Verified Care-Gap Closure (Module G)
@router.post("/care-gaps/close", response_model=CareGapClosureResponse)
def close_care_gap_with_evidence(
    payload: CareGapClosureRequest,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """
    CRITICAL MODULE G: Verified Care-Gap Closure.
    Only closes care gaps when supported by verifiable clinical evidence.
    Enforces human clinical oversight.
    """
    return NavigationService.verify_and_close_care_gap(
        db=db,
        person_id=payload.person_id,
        care_gap_id=payload.care_gap_id,
        evidence_source=payload.evidence_source,
        evidence_id=payload.evidence_id,
        verification_method=payload.verification_method,
        current_user=current_user,
        recommendation_id=payload.recommendation_id,
        verification_notes=payload.verification_notes,
    )


@router.get("/analytics", response_model=NavigationAnalyticsResponse)
def get_navigation_analytics(
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """
    Closed-loop navigation analytics: attendance rate, completed referrals, and verified care gaps.
    """
    user_role = current_user.role.value if hasattr(current_user.role, "value") else str(current_user.role)
    stats = NavigationService.get_navigation_analytics(
        db=db,
        organization_id=current_user.organization_id if user_role != "SYSTEM_ADMIN" else None,
    )
    return NavigationAnalyticsResponse(**stats)


# Screening Recalls
@router.post("/recalls", response_model=RecallTaskResponse, status_code=status.HTTP_201_CREATED)
def create_recall_task(
    payload: RecallTaskCreate,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """
    Schedule an automated recall task for due/overdue screening.
    """
    return NavigationService.create_recall_task(
        db=db,
        person_id=payload.person_id,
        pathway_id=payload.pathway_id,
        screening_type=payload.screening_type,
        due_date=payload.due_date,
        current_user=current_user,
    )


@router.get("/recalls", response_model=List[RecallTaskResponse])
def list_recalls(
    person_id: Optional[str] = Query(None),
    status: Optional[RecallStatus] = Query(None),
    limit: int = Query(50, ge=1, le=200),
    offset: int = Query(0, ge=0),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """
    List screening recall tasks scoped by organization.
    """
    user_role = current_user.role.value if hasattr(current_user.role, "value") else str(current_user.role)
    org_id = current_user.organization_id if user_role != "SYSTEM_ADMIN" else None
    return NavigationService.get_recalls(
        db=db,
        person_id=person_id,
        organization_id=org_id,
        status=status,
        limit=limit,
        offset=offset,
    )


@router.patch("/recalls/{recall_id}/status", response_model=RecallTaskResponse)
def update_recall_status(
    recall_id: str,
    payload: RecallStatusUpdate,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """
    Update recall status (SCHEDULED, NOTIFIED, COMPLETED, OVERDUE, CANCELLED).
    """
    return NavigationService.update_recall_status(
        db=db,
        recall_id=recall_id,
        new_status=payload.status,
        current_user=current_user,
    )
