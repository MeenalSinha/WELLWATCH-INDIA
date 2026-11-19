import uuid
import pytest
from datetime import date, datetime, timedelta
from sqlalchemy.orm import Session
from fastapi import HTTPException

from backend.app.models.organization import Organization
from backend.app.models.user import User, UserRole
from backend.app.models.person import Person, Gender
from backend.app.models.clinical import Encounter, ScreeningRecord, ScreeningStatus, EncounterType, EncounterStatus
from backend.app.models.screening import EpistemicStatus
from backend.app.models.recommendation import (
    PreventiveRecommendation,
    RecommendationLifecycleStatus,
    RecommendationCategory,
    RecommendationPriority,
)
from backend.app.models.navigation import CareGapClosureEvidence
from backend.app.services.person_service import PersonService
from backend.app.services.navigation_service import NavigationService


def _setup_clinical_env(db: Session):
    uid = uuid.uuid4().hex[:6]
    org = Organization(name=f"Safety Hospital {uid}", license_number=f"DOH-SAFE-{uid}", organization_type="HOSPITAL", region="Abu Dhabi")
    db.add(org)
    db.commit()

    admin = User(
        email=f"admin.safety.{uid}@wellwatch.ae",
        password_hash="pw",
        full_name="Hospital IT Admin",
        role=UserRole.SYSTEM_ADMIN,
        organization_id=org.id,
        is_active=True,
    )
    doctor = User(
        email=f"dr.safety.{uid}@wellwatch.ae",
        password_hash="pw",
        full_name="Dr. Amina Al Zaabi",
        role=UserRole.HEALTHCARE_PROVIDER,
        organization_id=org.id,
        is_active=True,
    )
    db.add_all([admin, doctor])
    db.commit()

    patient = PersonService.register_person(
        db=db,
        first_name="Mansoor",
        last_name="Al Falasi",
        date_of_birth=date.today() - timedelta(days=365 * 45),
        gender=Gender.MALE,
        organization_id=org.id,
    )
    return org, admin, doctor, patient


def test_clinical_safety_care_gap_closure_requires_licensed_clinician(db_session: Session):
    """Safety Control: Rejects non-clinician administrative roles from closing clinical care gaps."""
    org, admin, doctor, patient = _setup_clinical_env(db_session)

    rec = PreventiveRecommendation(
        organization_id=org.id,
        person_id=patient.id,
        pathway_id="ADPHC_CARDIOVASCULAR",
        category=RecommendationCategory.ADULT_COMPREHENSIVE_SCREENING,
        priority=RecommendationPriority.HIGH,
        lifecycle_status=RecommendationLifecycleStatus.APPROVED,
        recommended_action="Lipid Profile & ECG",
        clinical_rationale="Overdue cardiovascular risk evaluation",
        source_guideline="ADPHC 2026",
        rule_id="RULE-CVD-001",
        rule_version="1.0.0",
    )
    db_session.add(rec)
    db_session.commit()

    # IT Admin must be rejected (403 Forbidden)
    with pytest.raises(HTTPException) as exc_info:
        NavigationService.verify_and_close_care_gap(
            db=db_session,
            person_id=patient.id,
            care_gap_id="GAP-CVD-001",
            evidence_source="CLINICAL_NOTE",
            evidence_id="NOTE-SAFE-001",
            verification_method="CLINICIAN_ATTESTATION",
            current_user=admin,
            recommendation_id=rec.id,
            verification_notes="Administrative closure attempt",
        )
    assert exc_info.value.status_code == 403
    assert "only licensed healthcare providers" in exc_info.value.detail.lower()

    # Licensed doctor must succeed
    res = NavigationService.verify_and_close_care_gap(
        db=db_session,
        person_id=patient.id,
        care_gap_id="GAP-CVD-001",
        evidence_source="CLINICAL_NOTE",
        evidence_id="NOTE-SAFE-001",
        verification_method="CLINICIAN_ATTESTATION",
        current_user=doctor,
        recommendation_id=rec.id,
        verification_notes="Clinical chart review completed and lipid test validated by Dr. Amina.",
    )
    assert res.is_valid is True
    assert res.verifying_user_id == doctor.id
    assert res.care_gap_id == "GAP-CVD-001"


def test_clinical_safety_rejects_closure_without_evidence_id(db_session: Session):
    """Safety Control: Care gaps cannot be closed with empty evidence links."""
    org, admin, doctor, patient = _setup_clinical_env(db_session)

    rec = PreventiveRecommendation(
        organization_id=org.id,
        person_id=patient.id,
        pathway_id="ADPHC_DIABETES",
        category=RecommendationCategory.ADULT_COMPREHENSIVE_SCREENING,
        priority=RecommendationPriority.HIGH,
        lifecycle_status=RecommendationLifecycleStatus.APPROVED,
        recommended_action="Fasting Blood Glucose",
        clinical_rationale="Diabetes screening",
        source_guideline="ADPHC 2026",
        rule_id="RULE-DIAB-001",
        rule_version="1.0.0",
    )
    db_session.add(rec)
    db_session.commit()

    with pytest.raises(HTTPException) as exc_info:
        NavigationService.verify_and_close_care_gap(
            db=db_session,
            person_id=patient.id,
            care_gap_id="GAP-DIAB-001",
            evidence_source="ENCOUNTER",
            evidence_id="",  # Blank evidence ID
            verification_method="CLINICIAN_ATTESTATION",
            current_user=doctor,
            recommendation_id=rec.id,
        )
    assert exc_info.value.status_code == 400
    assert "requires verified evidence_source and evidence_id" in exc_info.value.detail.lower()


def test_clinical_safety_cross_patient_evidence_isolation(db_session: Session):
    """Safety Control: Strictly rejects closing a care gap using evidence from a different patient."""
    org, admin, doctor, patient1 = _setup_clinical_env(db_session)

    patient2 = PersonService.register_person(
        db=db_session,
        first_name="Fatima",
        last_name="Al Mazrouei",
        date_of_birth=date.today() - timedelta(days=365 * 30),
        gender=Gender.FEMALE,
        organization_id=org.id,
    )

    # Encounter belongs to Patient 2
    enc2 = Encounter(
        patient_id=patient2.id,
        facility_id=org.id,
        provider_user_id=doctor.id,
        encounter_type=EncounterType.OUTPATIENT_PREVENTIVE,
        status=EncounterStatus.COMPLETED,
        clinical_notes="Patient 2 encounter notes",
    )
    db_session.add(enc2)

    # Recommendation belongs to Patient 2
    rec2 = PreventiveRecommendation(
        organization_id=org.id,
        person_id=patient2.id,
        pathway_id="ADPHC_HYPERTENSION",
        category=RecommendationCategory.ADULT_COMPREHENSIVE_SCREENING,
        priority=RecommendationPriority.MEDIUM,
        lifecycle_status=RecommendationLifecycleStatus.APPROVED,
        recommended_action="Blood Pressure Check",
        clinical_rationale="Hypertension screening",
        source_guideline="ADPHC 2026",
        rule_id="RULE-HTN-001",
        rule_version="1.0.0",
    )
    db_session.add(rec2)
    db_session.commit()

    # Attempt to close Patient 1's care gap using Patient 2's encounter
    with pytest.raises(HTTPException) as exc_info:
        NavigationService.verify_and_close_care_gap(
            db=db_session,
            person_id=patient1.id,
            care_gap_id="GAP-HTN-P1",
            evidence_source="ENCOUNTER",
            evidence_id=enc2.id,
            verification_method="CLINICIAN_ATTESTATION",
            current_user=doctor,
        )
    assert exc_info.value.status_code == 400
    assert "different patient" in exc_info.value.detail.lower()

    # Attempt to close Patient 1's care gap using Patient 2's recommendation
    with pytest.raises(HTTPException) as exc_info2:
        NavigationService.verify_and_close_care_gap(
            db=db_session,
            person_id=patient1.id,
            care_gap_id="GAP-HTN-P1",
            evidence_source="CLINICAL_NOTE",
            evidence_id="NOTE-SAFE-002",
            verification_method="CLINICIAN_ATTESTATION",
            current_user=doctor,
            recommendation_id=rec2.id,
            verification_notes="Valid chart note text for closure",
        )
    assert exc_info2.value.status_code == 400
    assert "different patient" in exc_info2.value.detail.lower()
