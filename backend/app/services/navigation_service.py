import logging
from datetime import datetime, timezone, timedelta
from typing import Optional, List, Dict, Any, Tuple
from sqlalchemy.orm import Session
from fastapi import HTTPException, status

from backend.app.models.person import Person
from backend.app.models.user import User, UserRole
from backend.app.models.recommendation import (
    PreventiveRecommendation,
    RecommendationLifecycleStatus,
)
from backend.app.models.navigation import (
    Appointment,
    Referral,
    FollowUpTask,
    RecallTask,
    CareGapClosureEvidence,
    AppointmentStatus,
    ReferralStatus,
    FollowUpStatus,
    RecallStatus,
)
from backend.app.models.audit import AuditOutcome
from backend.app.services.audit_service import AuditService

logger = logging.getLogger(__name__)


def _get_role_str(user: User) -> str:
    return user.role.value if hasattr(user.role, "value") else str(user.role)


class NavigationService:

    @staticmethod
    def book_appointment(
        db: Session,
        person_id: str,
        service_type: str,
        scheduled_time: datetime,
        current_user: User,
        recommendation_id: Optional[str] = None,
        care_plan_id: Optional[str] = None,
        location: str = "Primary Care Clinic - Abu Dhabi",
        provider_id: Optional[str] = None,
    ) -> Appointment:
        person = db.query(Person).filter(Person.id == person_id).first()
        if not person:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Patient not found")

        role_str = _get_role_str(current_user)
        # Org boundary check
        if role_str != UserRole.SYSTEM_ADMIN.value and person.primary_organization_id and person.primary_organization_id != current_user.organization_id:
            raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Cross-organization appointment booking denied")

        # If recommendation linked, transition its lifecycle state to SCHEDULED
        if recommendation_id:
            rec = db.query(PreventiveRecommendation).filter(PreventiveRecommendation.id == recommendation_id).first()
            if rec:
                rec.lifecycle_status = RecommendationLifecycleStatus.SCHEDULED

        appt = Appointment(
            organization_id=person.primary_organization_id or current_user.organization_id,
            person_id=person_id,
            recommendation_id=recommendation_id,
            care_plan_id=care_plan_id,
            service_type=service_type,
            scheduled_time=scheduled_time,
            location=location,
            provider_id=provider_id,
            status=AppointmentStatus.BOOKED,
        )
        db.add(appt)
        db.commit()
        db.refresh(appt)

        AuditService.log(
            db=db,
            user_id=current_user.id,
            action="APPOINTMENT_BOOKED",
            resource_type="Appointment",
            resource_id=appt.id,
            organization_id=appt.organization_id,
            outcome=AuditOutcome.SUCCESS,
            details={"service_type": service_type, "scheduled_time": scheduled_time.isoformat()},
        )
        return appt

    @staticmethod
    def update_appointment_status(
        db: Session,
        appointment_id: str,
        new_status: AppointmentStatus,
        current_user: User,
        attendance_notes: Optional[str] = None,
    ) -> Appointment:
        appt = db.query(Appointment).filter(Appointment.id == appointment_id).first()
        if not appt:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Appointment not found")

        prev_status = appt.status
        appt.status = new_status
        if attendance_notes:
            appt.attendance_notes = attendance_notes

        # Missed-Care Detection (Module F):
        # If marked NO_SHOW, automatically generate follow-up task
        if new_status == AppointmentStatus.NO_SHOW and prev_status != AppointmentStatus.NO_SHOW:
            follow_up = FollowUpTask(
                organization_id=appt.organization_id,
                person_id=appt.person_id,
                appointment_id=appt.id,
                recommendation_id=appt.recommendation_id,
                trigger_reason="NO_SHOW",
                assigned_role="CARE_COORDINATOR",
                due_date=datetime.now(timezone.utc) + timedelta(days=3),
                priority="HIGH",
                status=FollowUpStatus.PENDING,
            )
            db.add(follow_up)

        db.commit()
        db.refresh(appt)

        AuditService.log(
            db=db,
            user_id=current_user.id,
            action="APPOINTMENT_STATUS_UPDATED",
            resource_type="Appointment",
            resource_id=appt.id,
            organization_id=appt.organization_id,
            outcome=AuditOutcome.SUCCESS,
            details={"previous_status": prev_status.value, "new_status": new_status.value},
        )
        return appt

    @staticmethod
    def create_referral(
        db: Session,
        person_id: str,
        specialty: str,
        clinical_indication: str,
        current_user: User,
        recommendation_id: Optional[str] = None,
        receiving_org_id: Optional[str] = None,
        priority: str = "ROUTINE",
    ) -> Referral:
        role_str = _get_role_str(current_user)
        # Enforce clinical role for creating specialist referrals
        if role_str != UserRole.HEALTHCARE_PROVIDER.value:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Role '{role_str}' is not authorized to originate clinical specialist referrals.",
            )

        person = db.query(Person).filter(Person.id == person_id).first()
        if not person:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Patient not found")

        referral = Referral(
            organization_id=person.primary_organization_id or current_user.organization_id,
            person_id=person_id,
            recommendation_id=recommendation_id,
            referring_provider_id=current_user.id,
            receiving_org_id=receiving_org_id,
            specialty=specialty,
            clinical_indication=clinical_indication,
            priority=priority,
            status=ReferralStatus.SUBMITTED,
        )
        db.add(referral)
        db.commit()
        db.refresh(referral)

        AuditService.log(
            db=db,
            user_id=current_user.id,
            action="REFERRAL_CREATED",
            resource_type="Referral",
            resource_id=referral.id,
            organization_id=referral.organization_id,
            outcome=AuditOutcome.SUCCESS,
            details={"specialty": specialty, "priority": priority},
        )
        return referral

    @staticmethod
    def update_referral_status(
        db: Session,
        referral_id: str,
        new_status: ReferralStatus,
        current_user: User,
        outcome_notes: Optional[str] = None,
    ) -> Referral:
        referral = db.query(Referral).filter(Referral.id == referral_id).first()
        if not referral:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Referral not found")

        referral.status = new_status
        if outcome_notes:
            referral.outcome_notes = outcome_notes

        # Incomplete or rejected referral detection
        if new_status in [ReferralStatus.REJECTED]:
            follow_up = FollowUpTask(
                organization_id=referral.organization_id,
                person_id=referral.person_id,
                referral_id=referral.id,
                recommendation_id=referral.recommendation_id,
                trigger_reason="INCOMPLETE_REFERRAL",
                assigned_role="CARE_COORDINATOR",
                due_date=datetime.now(timezone.utc) + timedelta(days=2),
                priority="HIGH",
                status=FollowUpStatus.PENDING,
            )
            db.add(follow_up)

        db.commit()
        db.refresh(referral)

        AuditService.log(
            db=db,
            user_id=current_user.id,
            action="REFERRAL_STATUS_UPDATED",
            resource_type="Referral",
            resource_id=referral.id,
            organization_id=referral.organization_id,
            outcome=AuditOutcome.SUCCESS,
            details={"new_status": new_status.value},
        )
        return referral

    @staticmethod
    def verify_and_close_care_gap(
        db: Session,
        person_id: str,
        care_gap_id: str,
        evidence_source: str,
        evidence_id: str,
        verification_method: str,
        current_user: User,
        recommendation_id: Optional[str] = None,
        verification_notes: Optional[str] = None,
    ) -> CareGapClosureEvidence:
        role_str = _get_role_str(current_user)
        # Authorization check: only authorized healthcare providers or system workflows can attest closure
        if role_str not in [UserRole.HEALTHCARE_PROVIDER.value, UserRole.SYSTEM_ADMIN.value]:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Role '{role_str}' is not authorized to certify clinical care gap closure.",
            )

        if not evidence_source or not evidence_id:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Care-gap closure requires verified evidence_source and evidence_id.",
            )

        person = db.query(Person).filter(Person.id == person_id).first()
        if not person:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Patient not found")

        closure_evidence = CareGapClosureEvidence(
            organization_id=person.primary_organization_id or current_user.organization_id,
            person_id=person_id,
            care_gap_id=care_gap_id,
            recommendation_id=recommendation_id,
            evidence_source=evidence_source,
            evidence_id=evidence_id,
            verification_method=verification_method,
            verifying_user_id=current_user.id,
            verification_notes=verification_notes,
            verified_at=datetime.now(timezone.utc),
            is_valid=True,
        )
        db.add(closure_evidence)

        # Update recommendation lifecycle status to COMPLETED if linked
        if recommendation_id:
            rec = db.query(PreventiveRecommendation).filter(PreventiveRecommendation.id == recommendation_id).first()
            if rec:
                rec.lifecycle_status = RecommendationLifecycleStatus.COMPLETED
                rec.reviewer_notes = f"Care gap closed via evidence {evidence_source}:{evidence_id} by {current_user.email}"

        db.commit()
        db.refresh(closure_evidence)

        AuditService.log(
            db=db,
            user_id=current_user.id,
            action="CARE_GAP_CLOSED_WITH_EVIDENCE",
            resource_type="CareGapClosureEvidence",
            resource_id=closure_evidence.id,
            organization_id=closure_evidence.organization_id,
            outcome=AuditOutcome.SUCCESS,
            details={
                "care_gap_id": care_gap_id,
                "evidence_source": evidence_source,
                "evidence_id": evidence_id,
            },
        )
        return closure_evidence

    @staticmethod
    def resolve_follow_up(
        db: Session,
        task_id: str,
        resolution_notes: str,
        current_user: User,
        resolution_evidence_id: Optional[str] = None,
    ) -> FollowUpTask:
        task = db.query(FollowUpTask).filter(FollowUpTask.id == task_id).first()
        if not task:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Follow-up task not found")

        task.status = FollowUpStatus.RESOLVED
        task.resolution_notes = resolution_notes
        task.resolution_evidence_id = resolution_evidence_id
        db.commit()
        db.refresh(task)

        AuditService.log(
            db=db,
            user_id=current_user.id,
            action="FOLLOW_UP_TASK_RESOLVED",
            resource_type="FollowUpTask",
            resource_id=task.id,
            organization_id=task.organization_id,
            outcome=AuditOutcome.SUCCESS,
            details={"resolution_notes": resolution_notes},
        )
        return task

    @staticmethod
    def get_navigation_analytics(db: Session, organization_id: Optional[str] = None) -> Dict[str, Any]:
        appts_q = db.query(Appointment)
        refs_q = db.query(Referral)
        followups_q = db.query(FollowUpTask)
        closure_q = db.query(CareGapClosureEvidence)

        if organization_id:
            appts_q = appts_q.filter(Appointment.organization_id == organization_id)
            refs_q = refs_q.filter(Referral.organization_id == organization_id)
            followups_q = followups_q.filter(FollowUpTask.organization_id == organization_id)
            closure_q = closure_q.filter(CareGapClosureEvidence.organization_id == organization_id)

        all_appts = appts_q.all()
        total_appts = len(all_appts)
        attended_appts = sum(1 for a in all_appts if a.status in [AppointmentStatus.ATTENDED, AppointmentStatus.COMPLETED])
        no_shows = sum(1 for a in all_appts if a.status == AppointmentStatus.NO_SHOW)
        attendance_rate = round(attended_appts / total_appts * 100, 1) if total_appts > 0 else 0.0

        all_refs = refs_q.all()
        total_refs = len(all_refs)
        completed_refs = sum(1 for r in all_refs if r.status == ReferralStatus.COMPLETED)

        active_followups = followups_q.filter(FollowUpTask.status.in_([FollowUpStatus.PENDING, FollowUpStatus.ASSIGNED, FollowUpStatus.IN_PROGRESS])).count()
        closed_care_gaps = closure_q.count()

        return {
            "total_appointments": total_appts,
            "attended_appointments": attended_appts,
            "no_show_appointments": no_shows,
            "attendance_rate": attendance_rate,
            "total_referrals": total_refs,
            "completed_referrals": completed_refs,
            "active_follow_up_tasks": active_followups,
            "closed_care_gaps_count": closed_care_gaps,
            "data_source_note": "Calculated strictly from persisted relational records in PostgreSQL/SQLite database.",
        }
