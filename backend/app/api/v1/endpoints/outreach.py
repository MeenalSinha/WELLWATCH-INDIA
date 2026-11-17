from typing import Optional
from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy.orm import Session

from backend.app.core.database import get_db
from backend.app.security.rbac import get_current_user
from backend.app.models.user import User
from backend.app.models.outreach import OutreachDeliveryStatus
from backend.app.schemas.outreach import (
    CommunicationPreferenceResponse,
    CommunicationPreferenceUpdate,
    OutreachTaskCreate,
    OutreachTaskResponse,
    OutreachTaskListResponse,
    OutreachAnalyticsResponse,
    TemplatePreviewRequest,
    TemplatePreviewResponse,
    InAppNotificationResponse,
)
from backend.app.services.outreach_service import OutreachService

router = APIRouter()


@router.get("/preferences/{person_id}", response_model=CommunicationPreferenceResponse)
def get_preferences(
    person_id: str,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """
    Retrieve cultural communication and accessibility preferences for a patient.
    Defaults to Arabic language in Abu Dhabi.
    """
    return OutreachService.get_or_create_preference(db, person_id)


@router.put("/preferences/{person_id}", response_model=CommunicationPreferenceResponse)
def update_preferences(
    person_id: str,
    payload: CommunicationPreferenceUpdate,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """
    Update communication preferences (language, channel, quiet hours, opt-in/opt-out consent).
    """
    data = payload.dict(exclude_unset=True)
    return OutreachService.update_preference(db, person_id, data, current_user)


@router.post("/tasks", response_model=OutreachTaskResponse, status_code=status.HTTP_201_CREATED)
def create_outreach_task(
    payload: OutreachTaskCreate,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """
    Schedule or immediately dispatch a culturally localized outreach task.
    Enforces consent validation, Wadeema's Law guardian proxy, quiet hours, and duplicate suppression.
    """
    return OutreachService.schedule_outreach_task(
        db=db,
        person_id=payload.person_id,
        template_id=payload.template_id,
        current_user=current_user,
        recommendation_id=payload.recommendation_id,
        override_channel=payload.channel,
        scheduled_at=payload.scheduled_at,
    )


@router.get("/tasks", response_model=OutreachTaskListResponse)
def list_outreach_tasks(
    person_id: Optional[str] = Query(None, description="Filter by patient ID"),
    delivery_status: Optional[OutreachDeliveryStatus] = Query(None, description="Filter by delivery status"),
    limit: int = Query(50, ge=1, le=200),
    offset: int = Query(0, ge=0),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """
    List outreach tasks with delivery status and pagination.
    """
    total, items = OutreachService.get_outreach_tasks(
        db=db,
        person_id=person_id,
        organization_id=current_user.organization_id if current_user.role.value != "SYSTEM_ADMIN" else None,
        delivery_status=delivery_status,
        limit=limit,
        offset=offset,
    )
    return OutreachTaskListResponse(total=total, items=items)


@router.post("/tasks/{task_id}/retry", response_model=OutreachTaskResponse)
def retry_outreach_task(
    task_id: str,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """
    Retry a failed outreach task adhering to maximum retry limits.
    """
    return OutreachService.retry_task(db, task_id, current_user)


@router.post("/preview", response_model=TemplatePreviewResponse)
def preview_template(
    payload: TemplatePreviewRequest,
):
    """
    Preview approved Arabic or English template with variable substitutions and RTL/LTR direction.
    Checks actual governance approval status from template registry.
    """
    templates = OutreachService.load_templates()
    tmpl = templates.get(payload.template_id)
    if not tmpl:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Template {payload.template_id} not found")
    is_approved = tmpl.get("governance_approved", False)

    subj, body, direction = OutreachService.render_template(
        template_id=payload.template_id,
        language=payload.language,
        variables=payload.variables,
    )
    return TemplatePreviewResponse(
        template_id=payload.template_id,
        language=payload.language,
        direction=direction,
        rendered_subject=subj,
        rendered_body=body,
        governance_approved=is_approved,
    )


@router.get("/inbox", response_model=list[InAppNotificationResponse])
def get_inbox(
    person_id: Optional[str] = Query(None, description="Patient person ID"),
    limit: int = Query(50, ge=1, le=100),
    offset: int = Query(0, ge=0),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """
    Retrieves genuinely persisted in-app notifications for patient or guardian portal.
    """
    return OutreachService.get_inbox(
        db=db,
        person_id=person_id,
        recipient_user_id=current_user.id,
        limit=limit,
        offset=offset,
    )


@router.patch("/inbox/{notification_id}/read", response_model=InAppNotificationResponse)
def mark_inbox_read(
    notification_id: str,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """
    Marks an in-app notification as read.
    """
    return OutreachService.mark_notification_read(db, notification_id, current_user)


@router.get("/analytics", response_model=OutreachAnalyticsResponse)
def get_outreach_analytics(
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """
    Operational outreach metrics: delivery rates, suppression, channel breakdowns, and honest provider status.
    """
    stats = OutreachService.get_analytics(
        db=db,
        organization_id=current_user.organization_id if current_user.role.value != "SYSTEM_ADMIN" else None,
    )
    return OutreachAnalyticsResponse(**stats)
