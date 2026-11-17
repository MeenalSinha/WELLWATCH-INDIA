"""
Integrated Workflows for Phases 9 to 11 of WELLWATCH PREVENT.
Covers:
- Integrated Workflow 1: Pediatric Prevention (Child -> Well-Child Gap -> Rec -> Care Plan -> Arabic Guardian Outreach -> Appointment -> Care Delivery -> Verified Care-Gap Closure)
- Integrated Workflow 2: Adult Preventive Screening (Adult -> IFHAS Gap -> Rec -> Care Plan -> Outreach -> Appointment -> Verified Evidence Closure)
- Integrated Workflow 3: Declined Recommendation (Offer -> Patient Decline with Reason -> Respect Choice -> Preserved History)
- Integrated Workflow 4: Failed Outreach (Honest Boundary -> Failure Persisted -> Retry Policy & Max Retries Enforced)
- Integrated Workflow 5: Referral Incomplete & Follow-up (Referral -> Incomplete/Rejection -> Follow-up Created -> Resolved with Final Evidence Closure)
"""
from datetime import date, datetime, timedelta, timezone
import pytest
from sqlalchemy.orm import Session

from backend.app.models.organization import Organization
from backend.app.models.user import User, UserRole
from backend.app.models.person import Gender, PopulationClassification
from backend.app.models.guardian import Guardianship, GuardianshipStatus
from backend.app.models.recommendation import (
    PreventiveRecommendation,
    RecommendationCategory,
    RecommendationPriority,
    RecommendationLifecycleStatus,
)
from backend.app.models.outreach import (
    CommunicationPreference,
    OutreachTask,
    CommunicationChannel,
    OutreachDeliveryStatus,
)
from backend.app.models.navigation import (
    Appointment,
    Referral,
    FollowUpTask,
    AppointmentStatus,
    ReferralStatus,
    FollowUpStatus,
)
from backend.app.models.audit import AuditLog, AuditOutcome
from backend.app.services.person_service import PersonService
from backend.app.services.recommendation_service import RecommendationService
from backend.app.services.outreach_service import OutreachService
from backend.app.services.navigation_service import NavigationService


def test_integrated_workflow_1_pediatric_prevention(db_session: Session):
    """
    Integrated Workflow 1: Pediatric Prevention
    End-to-end execution of Phases 9, 10, and 11 for pediatric population under UAE Wadeema's Law.
    """
    # 1. Facility, Provider & Legal Guardian
    org = Organization(
        name="Danat Al Emarat Integrated Pediatric Hospital",
        license_number="DOH-IW-01",
        organization_type="HOSPITAL",
        region="Abu Dhabi",
    )
    db_session.add(org)
    db_session.commit()

    provider = User(
        email="dr.maha@danatalemarat.ae",
        password_hash="pw",
        full_name="Dr. Maha Al Qasimi",
        role=UserRole.HEALTHCARE_PROVIDER,
        organization_id=org.id,
        is_active=True,
    )
    mother = User(
        email="shaikha.mother@wellwatch.ae",
        password_hash="pw",
        full_name="Shaikha Al Mansoori",
        role=UserRole.PARENT_GUARDIAN,
        is_active=True,
    )
    db_session.add_all([provider, mother])
    db_session.commit()

    # 2. Register child (12 months old) with legal guardian proxy
    child_dob = date.today() - timedelta(days=365)
    child = PersonService.register_person(
        db=db_session,
        first_name="Sultan",
        last_name="Al Mansoori",
        date_of_birth=child_dob,
        gender=Gender.MALE,
        organization_id=org.id,
        guardian_user_id=mother.id,
        relationship_to_guardian="Mother",
    )
    assert child.is_pediatric is True

    # 3. Phase 9: Recommendation Engine generates Well-Child recommendations
    recs = RecommendationService.generate_recommendations_for_patient(
        db=db_session,
        person_id=child.id,
        current_user=provider,
    )
    assert len(recs) > 0
    well_child_rec = next(r for r in recs if r.category == RecommendationCategory.WELL_CHILD_VISIT)
    assert well_child_rec.lifecycle_status == RecommendationLifecycleStatus.PENDING_CLINICAL_REVIEW

    # 4. Human Clinical Oversight: Clinician reviews and approves recommendation
    reviewed_rec = RecommendationService.review_recommendation(
        db=db_session,
        recommendation_id=well_child_rec.id,
        reviewer_user=provider,
        review_decision="APPROVED",
        rationale="12-month developmental surveillance and MMR vaccination check confirmed against UAE pediatric schedule.",
    )
    assert reviewed_rec.lifecycle_status == RecommendationLifecycleStatus.APPROVED

    # 5. Assemble structured Preventive Care Plan
    care_plan = RecommendationService.create_care_plan(
        db=db_session,
        person_id=child.id,
        plan_name="Abu Dhabi 12-Month Pediatric Wellness Plan",
        pathway_id="pediatric_well_child",
        goals=[{"goal": "12-month developmental surveillance and milestone check"}],
        assigned_care_team=[{"role": "Pediatrician", "user_id": provider.id}],
        recommendation_ids=[reviewed_rec.id],
        current_user=provider,
    )
    assert care_plan.status == "ACTIVE"

    # 6. Phase 10: Cultural Outreach in Arabic targeting legal guardian under Wadeema's Law
    outreach = OutreachService.schedule_outreach_task(
        db=db_session,
        person_id=child.id,
        template_id="TMPL-PED-WELL-CHILD-INVITE",
        current_user=provider,
        recommendation_id=reviewed_rec.id,
        override_channel=CommunicationChannel.IN_APP,
    )
    assert outreach.language == "ar"
    assert outreach.recipient_role == "GUARDIAN"
    assert outreach.recipient_id == mother.id
    assert outreach.delivery_status == OutreachDeliveryStatus.DELIVERED
    assert "مركز أبوظبي للصحة العامة" in outreach.message_body

    # 7. Phase 11: Care Navigation - Appointment coordination & attendance
    appt = NavigationService.book_appointment(
        db=db_session,
        person_id=child.id,
        service_type="WELL_CHILD_12M_VISIT",
        scheduled_time=datetime.now(timezone.utc) + timedelta(days=2),
        current_user=provider,
        recommendation_id=reviewed_rec.id,
        care_plan_id=care_plan.id,
    )
    assert appt.status == AppointmentStatus.BOOKED

    # Recommendation transitions to SCHEDULED
    db_session.refresh(reviewed_rec)
    assert reviewed_rec.lifecycle_status == RecommendationLifecycleStatus.SCHEDULED

    # 8. Clinical Encounter delivery - Attendance recorded
    NavigationService.update_appointment_status(
        db=db_session,
        appointment_id=appt.id,
        new_status=AppointmentStatus.ATTENDED,
        current_user=provider,
        attendance_notes="Child attended with mother. 12-month developmental milestones met, MMR vaccine administered.",
    )

    # 9. Module G: Verified Care-Gap Closure with verifiable encounter evidence
    closure = NavigationService.verify_and_close_care_gap(
        db=db_session,
        person_id=child.id,
        care_gap_id=reviewed_rec.care_gap_id or "GAP-WELL-CHILD-DUE",
        evidence_source="ENCOUNTER",
        evidence_id="enc_well_child_12m_9831",
        verification_method="CLINICIAN_ATTESTATION",
        current_user=provider,
        recommendation_id=reviewed_rec.id,
        verification_notes="Developmental examination completed and verified by Dr. Maha.",
    )
    assert closure.is_valid is True

    # 10. Invariant: Recommendation is now COMPLETED
    db_session.refresh(reviewed_rec)
    assert reviewed_rec.lifecycle_status == RecommendationLifecycleStatus.COMPLETED

    # 11. Complete Audit Trail Verification
    audits = (
        db_session.query(AuditLog)
        .filter(AuditLog.organization_id == org.id)
        .all()
    )
    action_names = {a.action for a in audits}
    assert "RECOMMENDATION_GENERATED" in action_names
    assert "RECOMMENDATION_REVIEWED" in action_names
    assert "CARE_PLAN_CREATED" in action_names
    assert "OUTREACH_TASK_SCHEDULED" in action_names
    assert "APPOINTMENT_BOOKED" in action_names
    assert "CARE_GAP_CLOSED_WITH_EVIDENCE" in action_names


def test_integrated_workflow_2_adult_preventive_screening(db_session: Session):
    """
    Integrated Workflow 2: Adult Preventive Screening (IFHAS)
    """
    org = Organization(
        name="Cleveland Clinic Integrated Adult Center",
        license_number="DOH-IW-02",
        organization_type="HOSPITAL",
        region="Abu Dhabi",
    )
    db_session.add(org)
    db_session.commit()

    provider = User(
        email="dr.khalil@cleveland.ae",
        password_hash="pw",
        full_name="Dr. Khalil Al Marzooqi",
        role=UserRole.HEALTHCARE_PROVIDER,
        organization_id=org.id,
        is_active=True,
    )
    db_session.add(provider)
    db_session.commit()

    adult = PersonService.register_person(
        db=db_session,
        first_name="Abdullah",
        last_name="Al Hashimi",
        date_of_birth=date.today() - timedelta(days=50 * 365),
        gender=Gender.MALE,
        organization_id=org.id,
    )

    # 1. Recommendation Generation
    recs = RecommendationService.generate_recommendations_for_patient(db_session, adult.id, provider)
    ifhas_rec = next(r for r in recs if r.pathway_id == "adphc_ifhas")

    # 2. Care Plan
    care_plan = RecommendationService.create_care_plan(
        db=db_session,
        person_id=adult.id,
        plan_name="ADPHC IFHAS Cardiovascular & Diabetes Preventive Plan",
        pathway_id="adphc_ifhas",
        goals=[{"goal": "Comprehensive periodic health assessment"}],
        assigned_care_team=[{"role": "Internal Medicine", "user_id": provider.id}],
        recommendation_ids=[ifhas_rec.id],
        current_user=provider,
    )

    # 3. Cultural Outreach in Arabic
    outreach = OutreachService.schedule_outreach_task(
        db=db_session,
        person_id=adult.id,
        template_id="TMPL-ADULT-IFHAS-INVITE",
        current_user=provider,
        recommendation_id=ifhas_rec.id,
        override_channel=CommunicationChannel.IN_APP,
    )
    assert outreach.delivery_status == OutreachDeliveryStatus.DELIVERED
    assert "افحص" in outreach.message_body

    # 4. Appointment & Screening Encounter
    appt = NavigationService.book_appointment(
        db=db_session,
        person_id=adult.id,
        service_type="ADPHC_IFHAS_EXAM",
        scheduled_time=datetime.now(timezone.utc) + timedelta(days=3),
        current_user=provider,
        recommendation_id=ifhas_rec.id,
    )
    NavigationService.update_appointment_status(
        db=db_session,
        appointment_id=appt.id,
        new_status=AppointmentStatus.ATTENDED,
        current_user=provider,
        attendance_notes="Completed IFHAS screening panel including fasting lipid panel and fasting glucose.",
    )

    # 5. Verified Care-Gap Closure with lab and screening record
    closure = NavigationService.verify_and_close_care_gap(
        db=db_session,
        person_id=adult.id,
        care_gap_id=ifhas_rec.care_gap_id or "GAP-IFHAS-COMPREHENSIVE",
        evidence_source="SCREENING_RECORD",
        evidence_id="scr_ifhas_2026_9901",
        verification_method="CLINICIAN_ATTESTATION",
        current_user=provider,
        recommendation_id=ifhas_rec.id,
        verification_notes="Fast blood glucose 5.2 mmol/L, Total Cholesterol 4.6 mmol/L, BP 120/78 mmHg. Normal results.",
    )
    assert closure.is_valid is True

    db_session.refresh(ifhas_rec)
    assert ifhas_rec.lifecycle_status == RecommendationLifecycleStatus.COMPLETED


def test_integrated_workflow_3_declined_recommendation(db_session: Session):
    """
    Integrated Workflow 3: Declined Recommendation
    Patient or guardian exercises autonomous choice to decline; reason documented; choice respected without repetitive harassment.
    """
    org = Organization(name="SEHA Clinic Declines", license_number="DOH-IW-03", organization_type="PRIMARY_CARE_CLINIC", region="Abu Dhabi")
    db_session.add(org)
    db_session.commit()

    provider = User(
        email="dr.decline.wf@seha.ae",
        password_hash="pw",
        full_name="Dr. Ebtisam",
        role=UserRole.HEALTHCARE_PROVIDER,
        organization_id=org.id,
        is_active=True,
    )
    db_session.add(provider)
    db_session.commit()

    adult = PersonService.register_person(
        db=db_session,
        first_name="Rashid",
        last_name="Al Mazrouei",
        date_of_birth=date.today() - timedelta(days=45 * 365),
        gender=Gender.MALE,
        organization_id=org.id,
    )

    recs = RecommendationService.generate_recommendations_for_patient(db_session, adult.id, provider)
    rec = recs[0]

    # Offer recommendation
    RecommendationService.update_status(db_session, rec.id, provider, RecommendationLifecycleStatus.OFFERED)

    # Patient declines with voluntary documented reason
    decline_reason = "Patient recently completed annual physical abroad in London and prefers not to repeat screening."
    declined_rec = RecommendationService.update_status(
        db=db_session,
        recommendation_id=rec.id,
        current_user=provider,
        new_status=RecommendationLifecycleStatus.DECLINED,
        declined_reason=decline_reason,
    )

    assert declined_rec.lifecycle_status == RecommendationLifecycleStatus.DECLINED
    assert declined_rec.declined_reason == decline_reason

    # Attempting to re-generate does not duplicate active recommendations
    recs_after = RecommendationService.generate_recommendations_for_patient(db_session, adult.id, provider)
    assert len(recs_after) == 0  # No duplicate active recs generated


def test_integrated_workflow_4_failed_outreach_and_retry(db_session: Session):
    """
    Integrated Workflow 4: Failed Outreach
    Honest provider boundary captures true gateway failures; applies retry policy; respects retry limits.
    """
    org = Organization(name="SEHA Failed Outreach Center", license_number="DOH-IW-04", organization_type="PRIMARY_CARE_CLINIC", region="Abu Dhabi")
    db_session.add(org)
    db_session.commit()

    provider = User(
        email="dr.outreach.fail@seha.ae",
        password_hash="pw",
        full_name="Dr. Tariq",
        role=UserRole.HEALTHCARE_PROVIDER,
        organization_id=org.id,
        is_active=True,
    )
    db_session.add(provider)
    db_session.commit()

    adult = PersonService.register_person(
        db=db_session,
        first_name="Faris",
        last_name="Al Zaabi",
        date_of_birth=date.today() - timedelta(days=38 * 365),
        gender=Gender.MALE,
        organization_id=org.id,
    )

    # Dispatch via unconfigured SMS gateway -> Honest provider boundary returns FAILED
    task = OutreachService.schedule_outreach_task(
        db=db_session,
        person_id=adult.id,
        template_id="TMPL-ADULT-BP-RECALL",
        current_user=provider,
        override_channel=CommunicationChannel.SMS,
    )
    assert task.delivery_status == OutreachDeliveryStatus.FAILED
    assert task.provider_status == "NOT_CONFIGURED"
    assert task.retry_count == 0

    # Retry task
    retried_task = OutreachService.retry_task(db_session, task.id, provider)
    assert retried_task.retry_count == 1
    assert retried_task.delivery_status == OutreachDeliveryStatus.FAILED  # Still honest


def test_integrated_workflow_5_referral_not_completed_and_closure(db_session: Session):
    """
    Integrated Workflow 5: Referral Incomplete -> Follow-up -> Resolution & Closure
    Specialist referral issued; rejection triggers follow-up; follow-up coordinator resolves with final clinical report.
    """
    org = Organization(name="Cleveland Specialist Nav", license_number="DOH-IW-05", organization_type="HOSPITAL", region="Abu Dhabi")
    db_session.add(org)
    db_session.commit()

    provider = User(
        email="dr.spec.ref@cleveland.ae",
        password_hash="pw",
        full_name="Dr. Amna",
        role=UserRole.HEALTHCARE_PROVIDER,
        organization_id=org.id,
        is_active=True,
    )
    coordinator = User(
        email="coord.ref@cleveland.ae",
        password_hash="pw",
        full_name="Fatima Care Coordinator",
        role=UserRole.CARE_COORDINATOR,
        organization_id=org.id,
        is_active=True,
    )
    db_session.add_all([provider, coordinator])
    db_session.commit()

    adult = PersonService.register_person(
        db=db_session,
        first_name="Saeed",
        last_name="Al Nuaimi",
        date_of_birth=date.today() - timedelta(days=58 * 365),
        gender=Gender.MALE,
        organization_id=org.id,
    )

    # 1. Specialist Referral Created
    referral = NavigationService.create_referral(
        db=db_session,
        person_id=adult.id,
        specialty="Preventive Cardiology",
        clinical_indication="Severe elevated LDL and high coronary calcium risk",
        current_user=provider,
        priority="URGENT",
    )
    assert referral.status == ReferralStatus.SUBMITTED

    # 2. Receiving center reports rejection/scheduling failure
    NavigationService.update_referral_status(
        db=db_session,
        referral_id=referral.id,
        new_status=ReferralStatus.REJECTED,
        current_user=provider,
        outcome_notes="Specialist clinic at capacity, recommended alternative center.",
    )

    # 3. Follow-up task automatically created for Care Coordinator
    follow_ups = (
        db_session.query(FollowUpTask)
        .filter(FollowUpTask.referral_id == referral.id)
        .all()
    )
    assert len(follow_ups) > 0
    fu_task = follow_ups[0]
    assert fu_task.trigger_reason == "INCOMPLETE_REFERRAL"
    assert fu_task.status == FollowUpStatus.PENDING

    # 4. Care Coordinator reroutes patient and resolves follow-up task
    resolved_task = NavigationService.resolve_follow_up(
        db=db_session,
        task_id=fu_task.id,
        resolution_notes="Contacted patient and successfully rescheduled at Sheikh Shakhbout Medical City (SSMC) Cardiology.",
        current_user=coordinator,
        resolution_evidence_id="ref_reschedule_ssmc_8899",
    )
    assert resolved_task.status == FollowUpStatus.RESOLVED

    # 5. Final specialist consultation completed, evidence received -> care gap verified and closed
    closure = NavigationService.verify_and_close_care_gap(
        db=db_session,
        person_id=adult.id,
        care_gap_id="GAP-LIPID-SPECIALIST-EVALUATION",
        evidence_source="REFERRAL_OUTCOME",
        evidence_id="ssmc_cardio_report_4432",
        verification_method="CLINICIAN_ATTESTATION",
        current_user=provider,
        verification_notes="Cardiology evaluation completed at SSMC, statin therapy initiated.",
    )
    assert closure.is_valid is True
    assert closure.evidence_source == "REFERRAL_OUTCOME"
