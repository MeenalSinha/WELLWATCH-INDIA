from datetime import date, datetime, timedelta, timezone
import pytest
from sqlalchemy.orm import Session

from backend.app.models.organization import Organization
from backend.app.models.user import User, UserRole
from backend.app.models.person import Gender, PopulationClassification
from backend.app.models.recommendation import (
    PreventiveRecommendation,
    RecommendationReview,
    PreventiveCarePlan,
    RecommendationCategory,
    RecommendationPriority,
    RecommendationLifecycleStatus,
)
from backend.app.models.assessment import (
    PreventiveAssessment,
    ClinicalRiskTier,
    HumanReviewStatus,
)
from backend.app.models.screening import EpistemicStatus
from backend.app.services.person_service import PersonService
from backend.app.services.recommendation_service import RecommendationService
from fastapi import HTTPException


def test_generate_pediatric_recommendations_well_child(db_session: Session):
    org = Organization(name="Danat Al Emarat Center", license_number="DOH-HOSP-R01", organization_type="HOSPITAL", region="Abu Dhabi")
    db_session.add(org)
    db_session.commit()

    provider = User(
        email="dr.pediatrics.rec@danatalemarat.ae",
        password_hash="hashed_pw",
        full_name="Dr. Noura Al Kaabi",
        role=UserRole.HEALTHCARE_PROVIDER,
        organization_id=org.id,
        is_active=True,
    )
    db_session.add(provider)
    db_session.commit()

    # 18-month old child
    child_dob = date.today() - timedelta(days=548)
    child = PersonService.register_person(
        db=db_session,
        first_name="Zayed",
        last_name="Al Zaabi",
        date_of_birth=child_dob,
        gender=Gender.MALE,
        organization_id=org.id,
    )

    # Generate recommendations
    recs = RecommendationService.generate_recommendations_for_patient(
        db=db_session,
        person_id=child.id,
        current_user=provider,
    )
    assert len(recs) > 0

    well_child_rec = next((r for r in recs if r.category == RecommendationCategory.WELL_CHILD_VISIT), None)
    assert well_child_rec is not None
    assert well_child_rec.pathway_id == "pediatric_well_child"
    assert "ADPHC Well-Child" in well_child_rec.source_guideline
    assert well_child_rec.lifecycle_status == RecommendationLifecycleStatus.PENDING_CLINICAL_REVIEW
    assert well_child_rec.requires_human_review is True


def test_generate_adult_recommendations_ifhas_and_cardiometabolic(db_session: Session):
    org = Organization(name="Cleveland Clinic Rec", license_number="DOH-HOSP-R02", organization_type="HOSPITAL", region="Abu Dhabi")
    db_session.add(org)
    db_session.commit()

    provider = User(
        email="dr.adult.rec@clevelandclinicabudhabi.ae",
        password_hash="hashed_pw",
        full_name="Dr. Tariq Al Nuaimi",
        role=UserRole.HEALTHCARE_PROVIDER,
        organization_id=org.id,
        is_active=True,
    )
    db_session.add(provider)
    db_session.commit()

    # 48-year old adult
    adult_dob = date.today() - timedelta(days=48 * 365)
    adult = PersonService.register_person(
        db=db_session,
        first_name="Khalid",
        last_name="Al Ameri",
        date_of_birth=adult_dob,
        gender=Gender.MALE,
        organization_id=org.id,
    )

    # Attach assessment with elevated blood pressure indicator
    import json
    assessment = PreventiveAssessment(
        patient_id=adult.id,
        assessment_type="ADPHC_IFHAS_CARDIOVASCULAR",
        population_category="ADULT",
        pathway_id="adphc_ifhas",
        pathway_version="1.0.0",
        clinical_risk_tier=ClinicalRiskTier.ELEVATED_RISK,
        eligibility_summary="{}",
        care_gaps_summary="{}",
        contributing_factors=json.dumps([{"indicator": "ELEVATED_BLOOD_PRESSURE", "measured": "148/92 mmHg"}]),
        data_quality_limitations="{}",
        human_review_status=HumanReviewStatus.AWAITING_REVIEW,
        recommended_action="Follow-up clinical blood pressure check",
    )
    db_session.add(assessment)
    db_session.commit()

    recs = RecommendationService.generate_recommendations_for_patient(
        db=db_session,
        person_id=adult.id,
        current_user=provider,
    )
    assert len(recs) >= 2

    bp_rec = next((r for r in recs if r.care_gap_id == "GAP-BP-OVERDUE"), None)
    assert bp_rec is not None
    assert bp_rec.category == RecommendationCategory.CARDIOMETABOLIC_SCREENING
    assert bp_rec.priority == RecommendationPriority.HIGH


def test_clinical_review_approval_and_rejection(db_session: Session):
    org = Organization(name="SEHA Clinic Rec", license_number="DOH-HOSP-R03", organization_type="PRIMARY_CARE_CLINIC", region="Abu Dhabi")
    db_session.add(org)
    db_session.commit()

    provider = User(
        email="dr.review.rec@seha.ae",
        password_hash="hashed_pw",
        full_name="Dr. Mariam Al Mansoori",
        role=UserRole.HEALTHCARE_PROVIDER,
        organization_id=org.id,
        is_active=True,
    )
    admin = User(
        email="admin.rec@seha.ae",
        password_hash="hashed_pw",
        full_name="Admin User",
        role=UserRole.SYSTEM_ADMIN,
        organization_id=org.id,
        is_active=True,
    )
    db_session.add_all([provider, admin])
    db_session.commit()

    child = PersonService.register_person(
        db=db_session,
        first_name="Meera",
        last_name="Al Shamsi",
        date_of_birth=date.today() - timedelta(days=365),
        gender=Gender.FEMALE,
        organization_id=org.id,
    )

    recs = RecommendationService.generate_recommendations_for_patient(db_session, child.id, provider)
    rec = recs[0]

    # Non-clinical system admin attempts review -> MUST BE REJECTED WITH 403
    with pytest.raises(HTTPException) as excinfo:
        RecommendationService.review_recommendation(
            db=db_session,
            recommendation_id=rec.id,
            reviewer_user=admin,
            review_decision="APPROVED",
            rationale="Admin bypass attempt",
        )
    assert excinfo.value.status_code == 403

    # Licensed healthcare provider reviews and approves
    reviewed = RecommendationService.review_recommendation(
        db=db_session,
        recommendation_id=rec.id,
        reviewer_user=provider,
        review_decision="APPROVED",
        rationale="Pediatric developmental surveillance plan verified against DoH clinical chart.",
    )
    assert reviewed.lifecycle_status == RecommendationLifecycleStatus.APPROVED
    assert reviewed.assigned_provider_id == provider.id
    assert reviewed.reviewer_notes == "Pediatric developmental surveillance plan verified against DoH clinical chart."


def test_recommendation_status_lifecycle_and_declined_reason(db_session: Session):
    org = Organization(name="SEHA Clinic Rec 2", license_number="DOH-HOSP-R04", organization_type="PRIMARY_CARE_CLINIC", region="Abu Dhabi")
    db_session.add(org)
    db_session.commit()

    provider = User(
        email="dr.lifecycle.rec@seha.ae",
        password_hash="hashed_pw",
        full_name="Dr. Mansoor",
        role=UserRole.HEALTHCARE_PROVIDER,
        organization_id=org.id,
        is_active=True,
    )
    db_session.add(provider)
    db_session.commit()

    adult = PersonService.register_person(
        db=db_session,
        first_name="Humaid",
        last_name="Al Dhaheri",
        date_of_birth=date.today() - timedelta(days=40 * 365),
        gender=Gender.MALE,
        organization_id=org.id,
    )

    recs = RecommendationService.generate_recommendations_for_patient(db_session, adult.id, provider)
    rec = recs[0]

    # Offer recommendation
    offered = RecommendationService.update_status(
        db=db_session,
        recommendation_id=rec.id,
        current_user=provider,
        new_status=RecommendationLifecycleStatus.OFFERED,
    )
    assert offered.lifecycle_status == RecommendationLifecycleStatus.OFFERED

    # Decline without reason -> FAILS
    with pytest.raises(HTTPException) as excinfo:
        RecommendationService.update_status(
            db=db_session,
            recommendation_id=rec.id,
            current_user=provider,
            new_status=RecommendationLifecycleStatus.DECLINED,
            declined_reason=None,
        )
    assert excinfo.value.status_code == 400

    # Decline with documented reason -> SUCCEEDS
    declined = RecommendationService.update_status(
        db=db_session,
        recommendation_id=rec.id,
        current_user=provider,
        new_status=RecommendationLifecycleStatus.DECLINED,
        declined_reason="Individual requested postponement due to travel outside the country.",
    )
    assert declined.lifecycle_status == RecommendationLifecycleStatus.DECLINED
    assert declined.declined_reason == "Individual requested postponement due to travel outside the country."


def test_care_plan_creation_and_recommendation_linking(db_session: Session):
    org = Organization(name="SEHA Care Plan Facility", license_number="DOH-HOSP-R05", organization_type="PRIMARY_CARE_CLINIC", region="Abu Dhabi")
    db_session.add(org)
    db_session.commit()

    provider = User(
        email="dr.careplan@seha.ae",
        password_hash="hashed_pw",
        full_name="Dr. Sultan",
        role=UserRole.HEALTHCARE_PROVIDER,
        organization_id=org.id,
        is_active=True,
    )
    db_session.add(provider)
    db_session.commit()

    child = PersonService.register_person(
        db=db_session,
        first_name="Salama",
        last_name="Al Romaithi",
        date_of_birth=date.today() - timedelta(days=2 * 365),
        gender=Gender.FEMALE,
        organization_id=org.id,
    )

    recs = RecommendationService.generate_recommendations_for_patient(db_session, child.id, provider)
    rec_ids = [r.id for r in recs]

    care_plan = RecommendationService.create_care_plan(
        db=db_session,
        person_id=child.id,
        plan_name="Abu Dhabi Toddler Preventive Care Plan",
        pathway_id="pediatric_well_child",
        goals=[{"goal": "Developmental milestone screening", "target": "24 months"}],
        assigned_care_team=[{"role": "Pediatrician", "user_id": provider.id}],
        recommendation_ids=rec_ids,
        current_user=provider,
    )

    assert care_plan.person_id == child.id
    assert care_plan.status == "ACTIVE"
    assert len(care_plan.items) == len(rec_ids)

    # Fetch plans
    plans = RecommendationService.get_care_plans_for_patient(db_session, child.id, provider)
    assert len(plans) == 1
    assert plans[0].plan_name == "Abu Dhabi Toddler Preventive Care Plan"
