import json
from datetime import date, datetime, timedelta, timezone
import pytest
from sqlalchemy.orm import Session

from backend.app.models.assessment import (
    AssessmentStatus,
    ClinicalRiskTier,
    HumanReviewStatus,
    PreventiveAssessment,
)
from backend.app.models.clinical import ClinicalCondition, ConditionClinicalStatus, ObservationType
from backend.app.models.organization import Organization
from backend.app.models.person import Gender
from backend.app.models.user import User, UserRole
from backend.app.services.person_service import PersonService
from backend.app.services.preventive_assessment_service import PreventiveAssessmentService


def test_elevated_blood_pressure_triggers_elevated_risk(db_session: Session):
    org = Organization(name="Primary Care Clinic", license_number="DOH-030", organization_type="CLINIC", region="Abu Dhabi")
    db_session.add(org)
    db_session.commit()

    adult = PersonService.register_person(
        db=db_session,
        first_name="Hassan",
        last_name="Al Marzooqi",
        date_of_birth=date(1975, 5, 10),
        gender=Gender.MALE,
        organization_id=org.id
    )

    # Record Stage 2 Hypertension: 152/95 mmHg
    PersonService.record_clinical_observation(
        db=db_session,
        person_id=adult.id,
        observation_type=ObservationType.VITAL_SIGN,
        code="8480-6",
        display_name="Systolic BP",
        numeric_value=152.0,
        unit="mmHg"
    )
    PersonService.record_clinical_observation(
        db=db_session,
        person_id=adult.id,
        observation_type=ObservationType.VITAL_SIGN,
        code="8462-4",
        display_name="Diastolic BP",
        numeric_value=95.0,
        unit="mmHg"
    )

    tier, factors, limitations, status = PreventiveAssessmentService.evaluate_clinical_risk_indicators(
        db=db_session,
        person=adult
    )

    assert tier == ClinicalRiskTier.ELEVATED_RISK
    assert len(factors) >= 2
    factor_names = [f["indicator"] for f in factors]
    assert "STAGE_2_HYPERTENSION_SYSTOLIC" in factor_names
    assert "STAGE_2_HYPERTENSION_DIASTOLIC" in factor_names


def test_missing_baseline_vitals_triggers_abstention_needs_data(db_session: Session):
    org = Organization(name="Wellness Center", license_number="DOH-031", organization_type="CLINIC", region="Abu Dhabi")
    db_session.add(org)
    db_session.commit()

    # Adult with zero vitals on file
    adult = PersonService.register_person(
        db=db_session,
        first_name="Mariam",
        last_name="Al Baloushi",
        date_of_birth=date(1990, 8, 20),
        gender=Gender.FEMALE,
        organization_id=org.id
    )

    tier, factors, limitations, status = PreventiveAssessmentService.evaluate_clinical_risk_indicators(
        db=db_session,
        person=adult
    )

    # System must abstain from assigning low or high risk when data is missing
    assert tier == ClinicalRiskTier.INSUFFICIENT_DATA
    assert status == AssessmentStatus.NEEDS_DATA
    assert len(limitations) > 0
    assert limitations[0]["issue"] == "MISSING_BASELINE"


def test_generate_assessment_persists_immutable_snapshot(db_session: Session):
    org = Organization(name="Abu Dhabi Clinic", license_number="DOH-032", organization_type="CLINIC", region="Abu Dhabi")
    db_session.add(org)
    db_session.commit()

    adult = PersonService.register_person(
        db=db_session,
        first_name="Saeed",
        last_name="Al Falasi",
        date_of_birth=date(1985, 2, 14),
        gender=Gender.MALE,
        organization_id=org.id
    )

    # Normal BP: 118/76 mmHg
    PersonService.record_clinical_observation(
        db=db_session,
        person_id=adult.id,
        observation_type=ObservationType.VITAL_SIGN,
        code="8480-6",
        display_name="Systolic BP",
        numeric_value=118.0,
        unit="mmHg"
    )
    PersonService.record_clinical_observation(
        db=db_session,
        person_id=adult.id,
        observation_type=ObservationType.VITAL_SIGN,
        code="8462-4",
        display_name="Diastolic BP",
        numeric_value=76.0,
        unit="mmHg"
    )

    assessment = PreventiveAssessmentService.generate_assessment(
        db=db_session,
        person_id=adult.id,
        pathway_id="ADPHC-ADULT-IFHAS-V1"
    )

    assert assessment.id is not None
    assert assessment.patient_id == adult.id
    assert assessment.pathway_id == "ADPHC-ADULT-IFHAS-V1"
    assert assessment.clinical_risk_tier == ClinicalRiskTier.LOW_RISK
    assert assessment.assessment_status in [AssessmentStatus.COMPLETED, AssessmentStatus.REQUIRES_REVIEW]

    # Verify JSON summaries
    eligibility = json.loads(assessment.eligibility_summary)
    assert eligibility["is_eligible"] is True


def test_human_clinical_review_workflow(db_session: Session):
    org = Organization(name="Hospital Care", license_number="DOH-033", organization_type="HOSPITAL", region="Abu Dhabi")
    doctor_user = User(email="dr.khaled@wellwatch.ae", full_name="Dr. Khaled Al Zaabi", password_hash="pw", role=UserRole.HEALTHCARE_PROVIDER.value, organization_id=org.id)
    db_session.add_all([org, doctor_user])
    db_session.commit()

    person = PersonService.register_person(
        db=db_session,
        first_name="Fatima",
        last_name="Al Shehhi",
        date_of_birth=date(1980, 11, 2),
        gender=Gender.FEMALE,
        organization_id=org.id
    )

    # Add documented Diabetes condition
    cond = ClinicalCondition(
        patient_id=person.id,
        code="E11.9",
        clinical_description="Type 2 Diabetes Mellitus without complications",
        clinical_status=ConditionClinicalStatus.ACTIVE
    )
    db_session.add(cond)
    db_session.commit()

    assessment = PreventiveAssessmentService.generate_assessment(
        db=db_session,
        person_id=person.id,
        pathway_id="ADPHC-ADULT-IFHAS-V1"
    )

    # Elevated condition triggers AWAITING_REVIEW
    assert assessment.clinical_risk_tier == ClinicalRiskTier.ELEVATED_RISK
    assert assessment.human_review_status == HumanReviewStatus.AWAITING_REVIEW

    # Doctor reviews and approves the assessment
    reviewed = PreventiveAssessmentService.submit_clinical_review(
        db=db_session,
        assessment_id=assessment.id,
        reviewer_user=doctor_user,
        new_status=HumanReviewStatus.APPROVED,
        notes="Confirmed elevated risk due to Type 2 diabetes. Enrolling patient in secondary cardiovascular prevention."
    )

    assert reviewed.human_review_status == HumanReviewStatus.APPROVED
    assert reviewed.reviewed_by_user_id == doctor_user.id
    assert reviewed.reviewed_at is not None
    assert "Confirmed elevated risk" in reviewed.reviewer_notes
