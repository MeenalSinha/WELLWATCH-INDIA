from datetime import date, datetime, timedelta, timezone
import pytest
from sqlalchemy.orm import Session
from fastapi import HTTPException

from backend.app.models.organization import Organization
from backend.app.models.user import User, UserRole
from backend.app.models.person import Gender
from backend.app.models.navigation import (
    AppointmentStatus,
    ReferralStatus,
    FollowUpStatus,
    RecallTask,
    RecallStatus,
)
from backend.app.models.clinical import Encounter, EncounterType, EncounterStatus
from backend.app.models.recommendation import (
    PreventiveRecommendation,
    RecommendationCategory,
    RecommendationPriority,
    RecommendationLifecycleStatus,
)
from backend.app.services.person_service import PersonService
from backend.app.services.navigation_service import NavigationService


def test_appointment_booking_and_status_update(db_session: Session):
    org = Organization(name="SEHA Al Bateen Nav", license_number="DOH-HOSP-N01", organization_type="PRIMARY_CARE_CLINIC", region="Abu Dhabi")
    db_session.add(org)
    db_session.commit()

    provider = User(
        email="dr.nav.appt@seha.ae",
        password_hash="pw",
        full_name="Dr. Zayed",
        role=UserRole.HEALTHCARE_PROVIDER,
        organization_id=org.id,
        is_active=True,
    )
    db_session.add(provider)
    db_session.commit()

    adult = PersonService.register_person(
        db=db_session,
        first_name="Salem",
        last_name="Al Nuaimi",
        date_of_birth=date.today() - timedelta(days=42 * 365),
        gender=Gender.MALE,
        organization_id=org.id,
    )

    sched_time = datetime.now(timezone.utc) + timedelta(days=2)
    appt = NavigationService.book_appointment(
        db=db_session,
        person_id=adult.id,
        service_type="ADULT_IFHAS_COMPREHENSIVE",
        scheduled_time=sched_time,
        current_user=provider,
    )
    assert appt.person_id == adult.id
    assert appt.status == AppointmentStatus.BOOKED

    # Update appointment status to ATTENDED
    updated = NavigationService.update_appointment_status(
        db=db_session,
        appointment_id=appt.id,
        new_status=AppointmentStatus.ATTENDED,
        current_user=provider,
        attendance_notes="Patient attended on time, completed intake and vitals.",
    )
    assert updated.status == AppointmentStatus.ATTENDED
    assert "completed intake" in updated.attendance_notes


def test_missed_appointment_no_show_triggers_follow_up(db_session: Session):
    org = Organization(name="Danat Al Emarat Nav", license_number="DOH-HOSP-N02", organization_type="HOSPITAL", region="Abu Dhabi")
    db_session.add(org)
    db_session.commit()

    provider = User(
        email="dr.noshow@danatalemarat.ae",
        password_hash="pw",
        full_name="Dr. Hessa",
        role=UserRole.HEALTHCARE_PROVIDER,
        organization_id=org.id,
        is_active=True,
    )
    db_session.add(provider)
    db_session.commit()

    child = PersonService.register_person(
        db=db_session,
        first_name="Rashid",
        last_name="Al Zaabi",
        date_of_birth=date.today() - timedelta(days=300),
        gender=Gender.MALE,
        organization_id=org.id,
    )

    appt = NavigationService.book_appointment(
        db=db_session,
        person_id=child.id,
        service_type="WELL_CHILD_VISIT",
        scheduled_time=datetime.now(timezone.utc) + timedelta(days=1),
        current_user=provider,
    )

    # Patient missed appointment -> Marked NO_SHOW
    NavigationService.update_appointment_status(
        db=db_session,
        appointment_id=appt.id,
        new_status=AppointmentStatus.NO_SHOW,
        current_user=provider,
        attendance_notes="Guardian did not show up and did not answer outreach phone call.",
    )

    # Automated Missed-Care Detection must have created a FollowUpTask
    tasks = NavigationService.get_navigation_analytics(db_session, org.id)
    assert tasks["no_show_appointments"] == 1
    assert tasks["active_follow_up_tasks"] >= 1


def test_clinical_referral_creation_and_rbac(db_session: Session):
    org = Organization(name="Cleveland Clinic Nav", license_number="DOH-HOSP-N03", organization_type="HOSPITAL", region="Abu Dhabi")
    db_session.add(org)
    db_session.commit()

    provider = User(
        email="dr.referral.doc@cleveland.ae",
        password_hash="pw",
        full_name="Dr. Omar",
        role=UserRole.HEALTHCARE_PROVIDER,
        organization_id=org.id,
        is_active=True,
    )
    chw = User(
        email="chw.referral@cleveland.ae",
        password_hash="pw",
        full_name="CHW Fatima",
        role=UserRole.COMMUNITY_HEALTH_WORKER,
        organization_id=org.id,
        is_active=True,
    )
    db_session.add_all([provider, chw])
    db_session.commit()

    adult = PersonService.register_person(
        db=db_session,
        first_name="Hamad",
        last_name="Al Qubaisi",
        date_of_birth=date.today() - timedelta(days=55 * 365),
        gender=Gender.MALE,
        organization_id=org.id,
    )

    # CHW attempts to create referral -> BLOCKED
    with pytest.raises(HTTPException) as excinfo:
        NavigationService.create_referral(
            db=db_session,
            person_id=adult.id,
            specialty="Preventive Cardiology",
            clinical_indication="Persistent Stage 2 HTN",
            current_user=chw,
        )
    assert excinfo.value.status_code == 403

    # Licensed Provider creates referral -> SUCCEEDS
    ref = NavigationService.create_referral(
        db=db_session,
        person_id=adult.id,
        specialty="Preventive Cardiology",
        clinical_indication="Persistent Stage 2 HTN",
        current_user=provider,
        priority="URGENT",
    )
    assert ref.specialty == "Preventive Cardiology"
    assert ref.status == ReferralStatus.SUBMITTED


def test_care_gap_cannot_be_closed_without_evidence(db_session: Session):
    org = Organization(name="SEHA Evidence Clinic", license_number="DOH-HOSP-N04", organization_type="PRIMARY_CARE_CLINIC", region="Abu Dhabi")
    db_session.add(org)
    db_session.commit()

    provider = User(
        email="dr.evidence@seha.ae",
        password_hash="pw",
        full_name="Dr. Mona",
        role=UserRole.HEALTHCARE_PROVIDER,
        organization_id=org.id,
        is_active=True,
    )
    db_session.add(provider)
    db_session.commit()

    adult = PersonService.register_person(
        db=db_session,
        first_name="Ali",
        last_name="Al Mansoori",
        date_of_birth=date.today() - timedelta(days=45 * 365),
        gender=Gender.MALE,
        organization_id=org.id,
    )

    # Attempt to close without evidence -> FAILS
    with pytest.raises(HTTPException) as excinfo:
        NavigationService.verify_and_close_care_gap(
            db=db_session,
            person_id=adult.id,
            care_gap_id="GAP-BP-OVERDUE",
            evidence_source="ENCOUNTER",
            evidence_id="",  # empty
            verification_method="CLINICIAN_ATTESTATION",
            current_user=provider,
        )
    assert excinfo.value.status_code == 400


def test_verified_care_gap_closure_completes_recommendation(db_session: Session):
    org = Organization(name="SEHA Closure Clinic", license_number="DOH-HOSP-N05", organization_type="PRIMARY_CARE_CLINIC", region="Abu Dhabi")
    db_session.add(org)
    db_session.commit()

    provider = User(
        email="dr.closure@seha.ae",
        password_hash="pw",
        full_name="Dr. Khaled",
        role=UserRole.HEALTHCARE_PROVIDER,
        organization_id=org.id,
        is_active=True,
    )
    db_session.add(provider)
    db_session.commit()

    adult = PersonService.register_person(
        db=db_session,
        first_name="Moza",
        last_name="Al Kaabi",
        date_of_birth=date.today() - timedelta(days=40 * 365),
        gender=Gender.FEMALE,
        organization_id=org.id,
    )

    rec = PreventiveRecommendation(
        organization_id=org.id,
        person_id=adult.id,
        pathway_id="adphc_ifhas",
        care_gap_id="GAP-IFHAS-COMPREHENSIVE",
        category=RecommendationCategory.ADULT_COMPREHENSIVE_SCREENING,
        priority=RecommendationPriority.MEDIUM,
        lifecycle_status=RecommendationLifecycleStatus.SCHEDULED,
        recommended_action="Complete IFHAS screening",
        clinical_rationale="Periodic screening due",
        supporting_evidence={},
        source_guideline="ADPHC IFHAS",
        uncertainty_status="CONFIRMED",
        rule_id="REC-ADULT-001",
        rule_version="1.0.0",
        requires_human_review=False,
    )
    db_session.add(rec)
    db_session.commit()
    db_session.refresh(rec)

    # Persist actual clinical encounter evidence
    encounter = Encounter(
        patient_id=adult.id,
        facility_id=org.id,
        provider_user_id=provider.id,
        encounter_type=EncounterType.OUTPATIENT_PREVENTIVE,
        status=EncounterStatus.COMPLETED,
        clinical_notes="Completed IFHAS screening encounter with vitals and labs",
    )
    db_session.add(encounter)
    db_session.commit()
    db_session.refresh(encounter)

    # Verify and close care gap with verified clinical screening encounter evidence
    evidence = NavigationService.verify_and_close_care_gap(
        db=db_session,
        person_id=adult.id,
        care_gap_id="GAP-IFHAS-COMPREHENSIVE",
        evidence_source="ENCOUNTER",
        evidence_id=encounter.id,
        verification_method="CLINICIAN_ATTESTATION",
        current_user=provider,
        recommendation_id=rec.id,
        verification_notes="Clinical encounter completed, fasting glucose and lipids reviewed by Dr. Khaled.",
    )

    assert evidence.care_gap_id == "GAP-IFHAS-COMPREHENSIVE"
    assert evidence.is_valid is True

    # Recommendation lifecycle status must now be COMPLETED
    db_session.refresh(rec)
    assert rec.lifecycle_status == RecommendationLifecycleStatus.COMPLETED
    assert "Care gap closed via evidence" in rec.reviewer_notes


def test_system_admin_rejected_from_closing_care_gap(db_session: Session):
    org = Organization(name="Admin Reject Clinic", license_number="DOH-HOSP-N06", organization_type="HOSPITAL", region="Abu Dhabi")
    db_session.add(org)
    db_session.commit()

    admin = User(
        email="sysadmin.nav@doh.gov.ae",
        password_hash="pw",
        full_name="Admin Saeed",
        role=UserRole.SYSTEM_ADMIN,
        organization_id=org.id,
        is_active=True,
    )
    db_session.add(admin)
    db_session.commit()

    patient = PersonService.register_person(
        db=db_session,
        first_name="Fatima",
        last_name="Al Mansoori",
        date_of_birth=date.today() - timedelta(days=35 * 365),
        gender=Gender.FEMALE,
        organization_id=org.id,
    )

    with pytest.raises(HTTPException) as excinfo:
        NavigationService.verify_and_close_care_gap(
            db=db_session,
            person_id=patient.id,
            care_gap_id="GAP-MAMMO-001",
            evidence_source="ENCOUNTER",
            evidence_id="fake_enc",
            verification_method="CLINICIAN_ATTESTATION",
            current_user=admin,
        )
    assert excinfo.value.status_code == 403
    assert "Only licensed healthcare providers may attest" in excinfo.value.detail


def test_cross_patient_evidence_rejected(db_session: Session):
    org = Organization(name="Cross Patient Clinic", license_number="DOH-HOSP-N07", organization_type="HOSPITAL", region="Abu Dhabi")
    db_session.add(org)
    db_session.commit()

    provider = User(
        email="dr.crosspatient@seha.ae",
        password_hash="pw",
        full_name="Dr. Noura",
        role=UserRole.HEALTHCARE_PROVIDER,
        organization_id=org.id,
        is_active=True,
    )
    db_session.add(provider)
    db_session.commit()

    patient_a = PersonService.register_person(
        db=db_session,
        first_name="Patient",
        last_name="A",
        date_of_birth=date.today() - timedelta(days=30 * 365),
        gender=Gender.FEMALE,
        organization_id=org.id,
    )
    patient_b = PersonService.register_person(
        db=db_session,
        first_name="Patient",
        last_name="B",
        date_of_birth=date.today() - timedelta(days=32 * 365),
        gender=Gender.MALE,
        organization_id=org.id,
    )

    # Encounter belongs to Patient B
    enc_b = Encounter(
        patient_id=patient_b.id,
        facility_id=org.id,
        provider_user_id=provider.id,
        encounter_type=EncounterType.OUTPATIENT_PREVENTIVE,
        status=EncounterStatus.COMPLETED,
    )
    db_session.add(enc_b)
    db_session.commit()

    # Attempt to close Patient A's gap with Patient B's encounter
    with pytest.raises(HTTPException) as excinfo:
        NavigationService.verify_and_close_care_gap(
            db=db_session,
            person_id=patient_a.id,
            care_gap_id="GAP-A",
            evidence_source="ENCOUNTER",
            evidence_id=enc_b.id,
            verification_method="CLINICIAN_ATTESTATION",
            current_user=provider,
        )
    assert excinfo.value.status_code == 400
    assert "belongs to a different patient" in excinfo.value.detail


def test_recall_task_lifecycle(db_session: Session):
    org = Organization(name="Recall Clinic", license_number="DOH-HOSP-N08", organization_type="HOSPITAL", region="Abu Dhabi")
    db_session.add(org)
    db_session.commit()

    provider = User(
        email="dr.recall@danatalemarat.ae",
        password_hash="pw",
        full_name="Dr. Shamma",
        role=UserRole.HEALTHCARE_PROVIDER,
        organization_id=org.id,
        is_active=True,
    )
    db_session.add(provider)
    db_session.commit()

    child = PersonService.register_person(
        db=db_session,
        first_name="Zayed",
        last_name="Al Dhaheri",
        date_of_birth=date.today() - timedelta(days=180),
        gender=Gender.MALE,
        organization_id=org.id,
    )

    recall = NavigationService.create_recall_task(
        db=db_session,
        person_id=child.id,
        pathway_id="adphc_well_child",
        screening_type="IMMUNIZATION_DUE",
        due_date=datetime.now(timezone.utc) + timedelta(days=14),
        current_user=provider,
    )
    assert recall.status == RecallStatus.SCHEDULED
    assert recall.screening_type == "IMMUNIZATION_DUE"

    recalls = NavigationService.get_recalls(db=db_session, person_id=child.id)
    assert len(recalls) >= 1

    updated = NavigationService.update_recall_status(
        db=db_session,
        recall_id=recall.id,
        new_status=RecallStatus.COMPLETED,
        current_user=provider,
    )
    assert updated.status == RecallStatus.COMPLETED
