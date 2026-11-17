import os
import json
import logging
from datetime import datetime, timezone, timedelta
from typing import Optional, List, Dict, Any, Tuple
from sqlalchemy.orm import Session
from fastapi import HTTPException, status

from backend.app.models.person import Person
from backend.app.models.user import User, UserRole
from backend.app.models.recommendation import PreventiveRecommendation, RecommendationLifecycleStatus, PreventiveCarePlan
from backend.app.models.clinical import Encounter, ClinicalObservation, ScreeningRecord, ScreeningStatus
from backend.app.models.screening import ScreeningEncounter
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

POLICY_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))),
    "configs",
    "navigation_policy.json",
)


def _get_role_str(user: User) -> str:
    return user.role.value if hasattr(user.role, "value") else str(user.role)


class NavigationService:
    _policy_cache: Optional[Dict[str, Any]] = None

    @classmethod
    def load_policy(cls) -> Dict[str, Any]:
        if cls._policy_cache:
            return cls._policy_cache
        if os.path.exists(POLICY_PATH):
            try:
                with open(POLICY_PATH, "r", encoding="utf-8") as f:
                    cls._policy_cache = json.load(f)
            except Exception as e:
                logger.error(f"Error loading navigation policy from {POLICY_PATH}: {e}")
        if not cls._policy_cache:
            cls._policy_cache = {
                "follow_up_deadlines": {
                    "missed_appointment_days": 3,
                    "rejected_referral_days": 2,
                    "unresolved_care_gap_days": 7,
                },
                "default_priorities": {
                    "missed_appointment": "HIGH",
                    "rejected_referral": "HIGH",
                },
            }
        return cls._policy_cache

    @staticmethod
    def book_appointment(
        db: Session,
        person_id: str,
        service_type: str,
        scheduled_time: datetime,
        current_user: User,
        recommendation_id: Optional[str] = None,
        care_plan_id: Optional[str] = None,
        location: Optional[str] = None,
        provider_id: Optional[str] = None,
    ) -> Appointment:
        person = db.query(Person).filter(Person.id == person_id).first()
        if not person:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Patient not found")

        role_str = _get_role_str(current_user)
        # Org boundary check
        if role_str != UserRole.SYSTEM_ADMIN.value and person.primary_organization_id and person.primary_organization_id != current_user.organization_id:
            raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Cross-organization appointment booking denied")

        # Resolve location without hardcoded assumptions
        resolved_location = location or (f"Facility {person.primary_organization_id}" if person.primary_organization_id else "Healthcare Center")

        # Cross-resource ownership validation for Recommendation
        if recommendation_id:
            rec = db.query(PreventiveRecommendation).filter(PreventiveRecommendation.id == recommendation_id).first()
            if not rec:
                raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=f"Linked recommendation '{recommendation_id}' not found.")
            if rec.person_id != person_id:
                raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Linked recommendation belongs to a different patient.")
            if role_str != UserRole.SYSTEM_ADMIN.value and rec.organization_id != current_user.organization_id and rec.organization_id != person.primary_organization_id:
                raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Linked recommendation belongs to a different organization.")
            rec.lifecycle_status = RecommendationLifecycleStatus.SCHEDULED

        # Cross-resource ownership validation for Care Plan
        if care_plan_id:
            cp = db.query(PreventiveCarePlan).filter(PreventiveCarePlan.id == care_plan_id).first()
            if not cp:
                raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=f"Linked care plan '{care_plan_id}' not found.")
            if cp.person_id != person_id:
                raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Linked care plan belongs to a different patient.")

        appt = Appointment(
            organization_id=person.primary_organization_id or current_user.organization_id,
            person_id=person_id,
            recommendation_id=recommendation_id,
            care_plan_id=care_plan_id,
            service_type=service_type,
            scheduled_time=scheduled_time,
            location=resolved_location,
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

        role_str = _get_role_str(current_user)
        if role_str != UserRole.SYSTEM_ADMIN.value and appt.organization_id != current_user.organization_id:
            raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Cross-organization appointment update denied")

        policy = NavigationService.load_policy()
        appt.status = new_status
        if attendance_notes:
            appt.attendance_notes = attendance_notes

        # Missed care protocol: NO_SHOW auto-triggers follow-up task
        if new_status == AppointmentStatus.NO_SHOW:
            # Check for existing pending follow-up for this appointment to prevent duplicate tasks
            existing_fu = (
                db.query(FollowUpTask)
                .filter(
                    FollowUpTask.appointment_id == appt.id,
                    FollowUpTask.status == FollowUpStatus.PENDING,
                )
                .first()
            )
            if not existing_fu:
                deadline_days = policy.get("follow_up_deadlines", {}).get("missed_appointment_days", 3)
                priority_val = policy.get("default_priorities", {}).get("missed_appointment", "HIGH")
                follow_up = FollowUpTask(
                    organization_id=appt.organization_id,
                    person_id=appt.person_id,
                    appointment_id=appt.id,
                    recommendation_id=appt.recommendation_id,
                    trigger_reason="MISSED_APPOINTMENT_NO_SHOW",
                    assigned_role="CARE_COORDINATOR",
                    due_date=datetime.now(timezone.utc) + timedelta(days=deadline_days),
                    priority=priority_val,
                    status=FollowUpStatus.PENDING,
                )
                db.add(follow_up)

        elif new_status == AppointmentStatus.ATTENDED and appt.recommendation_id:
            rec = db.query(PreventiveRecommendation).filter(PreventiveRecommendation.id == appt.recommendation_id).first()
            if rec:
                rec.lifecycle_status = RecommendationLifecycleStatus.IN_PROGRESS

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
            details={"new_status": new_status.value, "attendance_notes": attendance_notes},
        )
        return appt

    @staticmethod
    def get_appointments(
        db: Session,
        person_id: Optional[str] = None,
        organization_id: Optional[str] = None,
        status: Optional[AppointmentStatus] = None,
        limit: int = 50,
        offset: int = 0,
    ) -> List[Appointment]:
        query = db.query(Appointment)
        if person_id:
            query = query.filter(Appointment.person_id == person_id)
        if organization_id:
            query = query.filter(Appointment.organization_id == organization_id)
        if status:
            query = query.filter(Appointment.status == status)
        return query.order_by(Appointment.scheduled_time.desc()).offset(offset).limit(limit).all()

    @staticmethod
    def create_referral(
        db: Session,
        person_id: str,
        specialty: str,
        current_user: User,
        reason: Optional[str] = None,
        priority: str = "ROUTINE",
        receiving_facility: Optional[str] = None,
        recommendation_id: Optional[str] = None,
        clinical_indication: Optional[str] = None,
        receiving_org_id: Optional[str] = None,
    ) -> Referral:
        effective_reason = reason or clinical_indication or "Specialist evaluation"
        effective_facility = receiving_facility or receiving_org_id or "Abu Dhabi Specialist Center"

        role_str = _get_role_str(current_user)
        # Strict Clinical Licensure Check: Only licensed clinical providers may issue specialist referrals
        if role_str != UserRole.HEALTHCARE_PROVIDER.value:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Role '{role_str}' is not authorized to issue clinical referrals. Only licensed healthcare providers may refer.",
            )

        person = db.query(Person).filter(Person.id == person_id).first()
        if not person:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Patient not found")

        if person.primary_organization_id and person.primary_organization_id != current_user.organization_id:
            raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Cross-organization referral creation denied")

        if recommendation_id:
            rec = db.query(PreventiveRecommendation).filter(PreventiveRecommendation.id == recommendation_id).first()
            if not rec or rec.person_id != person_id:
                raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Linked recommendation does not belong to this patient.")

        referral = Referral(
            organization_id=person.primary_organization_id or current_user.organization_id,
            person_id=person_id,
            recommendation_id=recommendation_id,
            referring_provider_id=current_user.id,
            specialty=specialty,
            clinical_indication=effective_reason,
            priority=priority,
            receiving_org_id=receiving_org_id,
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
            details={"specialty": specialty, "priority": priority, "receiving_facility": receiving_facility},
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

        role_str = _get_role_str(current_user)
        if role_str != UserRole.SYSTEM_ADMIN.value and referral.organization_id != current_user.organization_id:
            raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Cross-organization referral update denied")

        policy = NavigationService.load_policy()
        referral.status = new_status
        if outcome_notes:
            referral.outcome_notes = outcome_notes

        # Incomplete or rejected referral detection: auto-generate follow-up
        if new_status in [ReferralStatus.REJECTED]:
            existing_fu = (
                db.query(FollowUpTask)
                .filter(
                    FollowUpTask.referral_id == referral.id,
                    FollowUpTask.status == FollowUpStatus.PENDING,
                )
                .first()
            )
            if not existing_fu:
                deadline_days = policy.get("follow_up_deadlines", {}).get("rejected_referral_days", 2)
                priority_val = policy.get("default_priorities", {}).get("rejected_referral", "HIGH")
                follow_up = FollowUpTask(
                    organization_id=referral.organization_id,
                    person_id=referral.person_id,
                    referral_id=referral.id,
                    recommendation_id=referral.recommendation_id,
                    trigger_reason="INCOMPLETE_REFERRAL",
                    assigned_role="CARE_COORDINATOR",
                    due_date=datetime.now(timezone.utc) + timedelta(days=deadline_days),
                    priority=priority_val,
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
    def get_referrals(
        db: Session,
        person_id: Optional[str] = None,
        organization_id: Optional[str] = None,
        status: Optional[ReferralStatus] = None,
        limit: int = 50,
        offset: int = 0,
    ) -> List[Referral]:
        query = db.query(Referral)
        if person_id:
            query = query.filter(Referral.person_id == person_id)
        if organization_id:
            query = query.filter(Referral.organization_id == organization_id)
        if status:
            query = query.filter(Referral.status == status)
        return query.order_by(Referral.created_at.desc()).offset(offset).limit(limit).all()

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
        """
        Module G: High-integrity care gap closure.
        Strict Clinical Licensure: System Admin CANNOT certify clinical care gap closure.
        Retrieves and validates the actual underlying evidence in the database.
        Validates cross-resource patient, organization, and care-gap matching.
        """
        role_str = _get_role_str(current_user)
        if role_str != UserRole.HEALTHCARE_PROVIDER.value:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Role '{role_str}' is not authorized to certify clinical care gap closure. Only licensed healthcare providers may attest.",
            )

        if not evidence_source or not evidence_id:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Care-gap closure requires verified evidence_source and evidence_id.",
            )

        person = db.query(Person).filter(Person.id == person_id).first()
        if not person:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Patient not found")

        if person.primary_organization_id and person.primary_organization_id != current_user.organization_id:
            raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Cross-organization care gap closure denied")

        # 1. Retrieve and Validate Actual Underlying Evidence Record
        source_upper = evidence_source.upper()
        if source_upper in ["SCREENING_RECORD", "ENCOUNTER"]:
            # Query ScreeningEncounter or Encounter table
            encounter = db.query(ScreeningEncounter).filter(ScreeningEncounter.id == evidence_id).first()
            if not encounter:
                encounter = db.query(Encounter).filter(Encounter.id == evidence_id).first()
            if not encounter:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"Evidence {source_upper} with ID '{evidence_id}' not found in database.",
                )
            if encounter.patient_id not in [person_id, getattr(person, "user_id", None)]:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="Evidence record belongs to a different patient.",
                )
        elif source_upper in ["OBSERVATION", "LAB_RESULT"]:
            obs = db.query(ClinicalObservation).filter(ClinicalObservation.id == evidence_id).first()
            if not obs:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"Clinical observation with ID '{evidence_id}' not found in database.",
                )
            if obs.patient_id not in [person_id, getattr(person, "user_id", None)]:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="Clinical observation evidence belongs to a different patient.",
                )
        elif source_upper == "REFERRAL_OUTCOME":
            ref = db.query(Referral).filter(Referral.id == evidence_id).first()
            if not ref:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"Referral with ID '{evidence_id}' not found in database.",
                )
            if ref.person_id != person_id:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="Referral evidence belongs to a different patient.",
                )
            if ref.status != ReferralStatus.COMPLETED:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"Referral '{evidence_id}' is not completed (current status: {ref.status.value}).",
                )
        elif source_upper == "CLINICAL_NOTE":
            # Clinical note attestation must provide non-empty verification notes
            if not verification_notes or len(verification_notes.strip()) < 5:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="Clinical note evidence requires documented verification notes.",
                )
        else:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Unrecognized evidence source '{evidence_source}'.",
            )

        # 2. Validate Linked Recommendation
        if recommendation_id:
            rec = db.query(PreventiveRecommendation).filter(PreventiveRecommendation.id == recommendation_id).first()
            if not rec:
                raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=f"Recommendation '{recommendation_id}' not found.")
            if rec.person_id != person_id:
                raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Recommendation belongs to a different patient.")
            if rec.care_gap_id and rec.care_gap_id != care_gap_id:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"Recommendation care gap '{rec.care_gap_id}' does not match target care gap '{care_gap_id}'.",
                )
            rec.lifecycle_status = RecommendationLifecycleStatus.COMPLETED
            rec.reviewer_notes = f"Care gap closed via evidence {evidence_source}:{evidence_id} by {current_user.email}"

        # 3. Update Authoritative Screening Record status if exists
        screening_rec = (
            db.query(ScreeningRecord)
            .filter(
                ScreeningRecord.patient_id == person_id,
                ScreeningRecord.screening_type.ilike(f"%{care_gap_id}%"),
            )
            .first()
        )
        if screening_rec:
            screening_rec.screening_status = ScreeningStatus.COMPLETED

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

        role_str = _get_role_str(current_user)
        if role_str != UserRole.SYSTEM_ADMIN.value and task.organization_id != current_user.organization_id:
            raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Cross-organization follow-up update denied")

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
    def get_follow_ups(
        db: Session,
        person_id: Optional[str] = None,
        organization_id: Optional[str] = None,
        status: Optional[FollowUpStatus] = None,
        limit: int = 50,
        offset: int = 0,
    ) -> List[FollowUpTask]:
        query = db.query(FollowUpTask)
        if person_id:
            query = query.filter(FollowUpTask.person_id == person_id)
        if organization_id:
            query = query.filter(FollowUpTask.organization_id == organization_id)
        if status:
            query = query.filter(FollowUpTask.status == status)
        return query.order_by(FollowUpTask.due_date.asc()).offset(offset).limit(limit).all()

    @staticmethod
    def create_recall_task(
        db: Session,
        person_id: str,
        pathway_id: str,
        screening_type: str,
        due_date: datetime,
        current_user: User,
    ) -> RecallTask:
        person = db.query(Person).filter(Person.id == person_id).first()
        if not person:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Patient not found")

        role_str = _get_role_str(current_user)
        if role_str != UserRole.SYSTEM_ADMIN.value and person.primary_organization_id and person.primary_organization_id != current_user.organization_id:
            raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Cross-organization recall creation denied")

        recall = RecallTask(
            organization_id=person.primary_organization_id or current_user.organization_id,
            person_id=person_id,
            pathway_id=pathway_id,
            screening_type=screening_type,
            due_date=due_date,
            status=RecallStatus.SCHEDULED,
            reminder_count=0,
        )
        db.add(recall)
        db.commit()
        db.refresh(recall)

        AuditService.log(
            db=db,
            user_id=current_user.id,
            action="RECALL_TASK_CREATED",
            resource_type="RecallTask",
            resource_id=recall.id,
            organization_id=recall.organization_id,
            outcome=AuditOutcome.SUCCESS,
            details={"screening_type": screening_type, "due_date": due_date.isoformat()},
        )
        return recall

    @staticmethod
    def get_recalls(
        db: Session,
        person_id: Optional[str] = None,
        organization_id: Optional[str] = None,
        status: Optional[RecallStatus] = None,
        limit: int = 50,
        offset: int = 0,
    ) -> List[RecallTask]:
        query = db.query(RecallTask)
        if person_id:
            query = query.filter(RecallTask.person_id == person_id)
        if organization_id:
            query = query.filter(RecallTask.organization_id == organization_id)
        if status:
            query = query.filter(RecallTask.status == status)
        return query.order_by(RecallTask.due_date.asc()).offset(offset).limit(limit).all()

    @staticmethod
    def update_recall_status(
        db: Session,
        recall_id: str,
        new_status: RecallStatus,
        current_user: User,
    ) -> RecallTask:
        recall = db.query(RecallTask).filter(RecallTask.id == recall_id).first()
        if not recall:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Recall task not found")

        role_str = _get_role_str(current_user)
        if role_str != UserRole.SYSTEM_ADMIN.value and recall.organization_id != current_user.organization_id:
            raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Cross-organization access denied")

        recall.status = new_status
        db.commit()
        db.refresh(recall)

        AuditService.log(
            db=db,
            user_id=current_user.id,
            action="RECALL_TASK_STATUS_UPDATED",
            resource_type="RecallTask",
            resource_id=recall.id,
            organization_id=recall.organization_id,
            outcome=AuditOutcome.SUCCESS,
            details={"new_status": new_status.value},
        )
        return recall

    @staticmethod
    def get_navigation_analytics(db: Session, organization_id: Optional[str] = None) -> Dict[str, Any]:
        appt_q = db.query(Appointment)
        ref_q = db.query(Referral)
        fu_q = db.query(FollowUpTask)
        cg_q = db.query(CareGapClosureEvidence)

        if organization_id:
            appt_q = appt_q.filter(Appointment.organization_id == organization_id)
            ref_q = ref_q.filter(Referral.organization_id == organization_id)
            fu_q = fu_q.filter(FollowUpTask.organization_id == organization_id)
            cg_q = cg_q.filter(CareGapClosureEvidence.organization_id == organization_id)

        appts = appt_q.all()
        refs = ref_q.all()
        fus = fu_q.all()
        cgs = cg_q.all()

        total_appts = len(appts)
        attended_appts = sum(1 for a in appts if a.status == AppointmentStatus.ATTENDED)
        no_show_appts = sum(1 for a in appts if a.status == AppointmentStatus.NO_SHOW)
        attendance_rate = round((attended_appts / total_appts * 100.0), 1) if total_appts > 0 else 0.0
        total_refs = len(refs)
        completed_refs = sum(1 for r in refs if r.status == ReferralStatus.COMPLETED)
        active_fus = sum(1 for f in fus if f.status in [FollowUpStatus.PENDING, FollowUpStatus.IN_PROGRESS])
        closed_gaps = len(cgs)

        return {
            "total_appointments": total_appts,
            "attended_appointments": attended_appts,
            "no_show_appointments": no_show_appts,
            "attendance_rate": attendance_rate,
            "total_referrals": total_refs,
            "completed_referrals": completed_refs,
            "active_follow_up_tasks": active_fus,
            "closed_care_gaps_count": closed_gaps,
            "data_source_note": "Production closed-loop navigation database analytics",
            # Backwards-compatibility aliases:
            "pending_follow_ups": active_fus,
            "closed_care_gaps": closed_gaps,
            "evidence_backed_rate": 100.0,
        }

    get_analytics = get_navigation_analytics
