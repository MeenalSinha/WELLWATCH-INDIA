import uuid
import pytest
from datetime import date, datetime, timedelta
from sqlalchemy.orm import Session
from fastapi import HTTPException

from backend.app.models.organization import Organization
from backend.app.models.user import User, UserRole
from backend.app.models.person import Person, Gender
from backend.app.models.screening import ScreeningEncounter, EncounterType, EpistemicStatus
from backend.app.models.clinical import ScreeningRecord, ScreeningStatus
from backend.app.services.person_service import PersonService
from backend.app.services.population_analytics_service import PopulationAnalyticsService
from backend.app.api.v1.endpoints.population import generate_population_report
from backend.app.schemas.population import ReportGenerateRequest


def _setup_org_and_admin(db: Session):
    uid = uuid.uuid4().hex[:6]
    org = Organization(name=f"SEHA Al Bateen Health {uid}", license_number=f"DOH-POP-{uid}", organization_type="PRIMARY_CARE_CLINIC", region="Abu Dhabi")
    db.add(org)
    db.commit()

    admin = User(
        email=f"pop.admin.{uid}@wellwatch.ae",
        password_hash="pw",
        full_name="Fatima Al Zaabi",
        role=UserRole.POPULATION_HEALTH_ADMIN,
        organization_id=org.id,
        is_active=True,
    )
    db.add(admin)
    db.commit()
    return org, admin


def test_population_overview_denominators(db_session: Session):
    """Verifies that population metrics maintain distinct, mathematically sound denominators."""
    org, admin = _setup_org_and_admin(db_session)

    # 4-year-old child
    child = PersonService.register_person(
        db=db_session,
        first_name="Zayed",
        last_name="Al Nahyan",
        date_of_birth=date.today() - timedelta(days=365 * 4),
        gender=Gender.MALE,
        organization_id=org.id,
    )

    # 35-year-old adult
    adult = PersonService.register_person(
        db=db_session,
        first_name="Mariam",
        last_name="Al Zaabi",
        date_of_birth=date.today() - timedelta(days=365 * 35),
        gender=Gender.FEMALE,
        organization_id=org.id,
    )

    # Record completed screening for adult
    screening = ScreeningRecord(
        patient_id=adult.id,
        performing_facility_id=org.id,
        screening_type="IFHAS_COMPREHENSIVE",
        screening_date=date.today(),
        screening_status=ScreeningStatus.COMPLETED,
        findings_summary="Annual IFHAS routine screening",
        epistemic_status=EpistemicStatus.VERIFIED,
    )
    db_session.add(screening)
    db_session.commit()

    overview = PopulationAnalyticsService.get_overview(
        db=db_session, organization_id=org.id, time_window_days=90
    )

    assert overview.total_registered >= 2
    assert overview.pediatric_population >= 1
    assert overview.adult_population >= 1
    assert overview.completed_screenings >= 1
    assert 0.0 < overview.screening_coverage_percentage <= 100.0


def test_pediatric_intelligence_developmental_stages(db_session: Session):
    """Verifies pediatric developmental stage classifications and coverage metrics."""
    org, admin = _setup_org_and_admin(db_session)

    infant = PersonService.register_person(
        db=db_session,
        first_name="Salem",
        last_name="Al Mansoori",
        date_of_birth=date.today() - timedelta(days=180),
        gender=Gender.MALE,
        organization_id=org.id,
    )

    pediatric_data = PopulationAnalyticsService.get_pediatric_intelligence(
        db=db_session, organization_id=org.id
    )

    assert pediatric_data.total_pediatric >= 1
    assert pediatric_data.developmental_stages["INFANT_0_12M"] >= 1
    assert pediatric_data.eligible_for_well_child == pediatric_data.total_pediatric


def test_adult_intelligence_ifhas_coverage(db_session: Session):
    """Verifies adult IFHAS screening coverage and risk tier aggregation."""
    org, admin = _setup_org_and_admin(db_session)

    adult = PersonService.register_person(
        db=db_session,
        first_name="Ahmed",
        last_name="Al Dhaheri",
        date_of_birth=date.today() - timedelta(days=365 * 45),
        gender=Gender.MALE,
        organization_id=org.id,
    )

    adult_data = PopulationAnalyticsService.get_adult_intelligence(
        db=db_session, organization_id=org.id
    )

    assert adult_data.total_adult >= 1
    assert adult_data.eligible_for_ifhas >= 1
    assert "LOW" in adult_data.cardiometabolic_risk_distribution
    assert "UNASSESSED" in adult_data.cardiometabolic_risk_distribution


def test_geographic_analytics_small_cell_suppression(db_session: Session):
    """Verifies that geographic cohorts with count < 5 are properly masked to protect patient privacy."""
    org, admin = _setup_org_and_admin(db_session)

    geo_data = PopulationAnalyticsService.get_geographic_analytics(
        db=db_session, organization_id=org.id
    )

    assert len(geo_data.regions) == 3
    # Check if suppression was applied to any small cell (< 5)
    suppressed_regions = [r for r in geo_data.regions if r.is_suppressed]
    for r in suppressed_regions:
        assert r.registered_population == "< 5"
        assert r.screening_coverage_percentage is None


def test_population_equity_and_ml_fairness_guardrail(db_session: Session):
    """Verifies disparity ratio computation and ML fairness epistemic abstention."""
    org, admin = _setup_org_and_admin(db_session)

    equity_data = PopulationAnalyticsService.get_equity_analytics(
        db=db_session, organization_id=org.id
    )

    assert equity_data.benchmark_region == "AD_CENTRAL"
    assert len(equity_data.subgroup_metrics) > 0
    # Guardrail: Abstains from calculating synthetic fairness metrics without longitudinal ground truth
    assert equity_data.ml_fairness_evaluation["status"] == "UNAVAILABLE"
    assert equity_data.ml_fairness_evaluation["reason"] == "NO_CONFIRMED_GROUND_TRUTH_LABELS"


def test_population_report_generation_rbac(db_session: Session):
    """Verifies that only authorized roles can generate population health aggregate reports."""
    org, admin = _setup_org_and_admin(db_session)
    uid = uuid.uuid4().hex[:6]

    indiv = User(
        email=f"indiv.pop.rpt.{uid}@wellwatch.ae",
        password_hash="pw",
        full_name="Patient Individual",
        role=UserRole.INDIVIDUAL,
        is_active=True,
    )
    db_session.add(indiv)
    db_session.commit()

    gov_officer = User(
        email=f"dgo.pop.rpt.{uid}@wellwatch.ae",
        password_hash="pw",
        full_name="Data Governance Officer",
        role=UserRole.DATA_GOVERNANCE_OFFICER,
        organization_id=org.id,
        is_active=True,
    )
    db_session.add(gov_officer)
    db_session.commit()

    req = ReportGenerateRequest(report_type="POPULATION_SUMMARY", time_window_days=90)

    # Individual should be forbidden
    with pytest.raises(HTTPException) as exc_info:
        generate_population_report(payload=req, db=db_session, current_user=indiv)
    assert exc_info.value.status_code == 403

    # Data Governance Officer should succeed
    res = generate_population_report(payload=req, db=db_session, current_user=gov_officer)
    assert res.report_type == "POPULATION_SUMMARY"
    assert "metrics_summary" in res.model_dump()


def test_population_trends_with_verified_care_gap_closure(db_session: Session):
    """Verifies that longitudinal trends calculate closed care gaps using genuine verified_at timestamps."""
    from backend.app.models.navigation import CareGapClosureEvidence
    from backend.app.models.recommendation import PreventiveRecommendation, RecommendationCategory, RecommendationPriority, RecommendationLifecycleStatus

    org, admin = _setup_org_and_admin(db_session)
    patient = PersonService.register_person(
        db=db_session,
        first_name="Sultan",
        last_name="Al Nuaimi",
        date_of_birth=date.today() - timedelta(days=365 * 42),
        gender=Gender.MALE,
        organization_id=org.id,
    )

    rec = PreventiveRecommendation(
        organization_id=org.id,
        person_id=patient.id,
        pathway_id="ADPHC_DIABETES",
        category=RecommendationCategory.ADULT_COMPREHENSIVE_SCREENING,
        priority=RecommendationPriority.MEDIUM,
        lifecycle_status=RecommendationLifecycleStatus.COMPLETED,
        recommended_action="HbA1c Blood Test",
        clinical_rationale="Overdue diabetes screening",
        source_guideline="ADPHC 2026",
        rule_id="RULE-DIAB-001",
        rule_version="1.0.0",
    )
    db_session.add(rec)
    db_session.commit()

    closure = CareGapClosureEvidence(
        organization_id=org.id,
        person_id=patient.id,
        care_gap_id="GAP-DIAB-001",
        recommendation_id=rec.id,
        evidence_source="LAB_RESULT",
        evidence_id="LAB-HBA1C-991",
        verification_method="AUTOMATED_OBSERVATION_MATCH",
        verifying_user_id=admin.id,
        verified_at=datetime.utcnow() - timedelta(days=5),
        is_valid=True,
    )
    db_session.add(closure)
    db_session.commit()

    trends = PopulationAnalyticsService.get_trends(db=db_session, organization_id=org.id)
    assert len(trends.trend_points) == 6
    latest_month = trends.trend_points[-1]
    assert latest_month.closed_care_gaps >= 1


def test_geographic_analytics_realistic_mapping_and_data_completeness(db_session: Session):
    """Verifies that geographic analysis uses genuine patient districts and reflects real profile completeness."""
    org, admin = _setup_org_and_admin(db_session)

    # Patient in Al Ain
    p_ain = PersonService.register_person(
        db=db_session,
        first_name="Hamdan",
        last_name="Al Ameri",
        date_of_birth=date.today() - timedelta(days=365 * 30),
        gender=Gender.MALE,
        organization_id=org.id,
    )
    p_ain.district = "Al Jimi, Al Ain"
    p_ain.patient_profile.completeness_score = 0.85

    # Patient in Al Dhafra
    p_dhafra = PersonService.register_person(
        db=db_session,
        first_name="Mohamed",
        last_name="Al Mazrouei",
        date_of_birth=date.today() - timedelta(days=365 * 50),
        gender=Gender.MALE,
        organization_id=org.id,
    )
    p_dhafra.district = "Madinat Zayed, Al Dhafra"
    p_dhafra.patient_profile.completeness_score = 0.70

    # Patient in Abu Dhabi Central
    p_central = PersonService.register_person(
        db=db_session,
        first_name="Shaikha",
        last_name="Al Mansoori",
        date_of_birth=date.today() - timedelta(days=365 * 25),
        gender=Gender.FEMALE,
        organization_id=org.id,
    )
    p_central.district = "Al Bateen, Abu Dhabi"
    p_central.patient_profile.completeness_score = 0.90

    db_session.commit()

    # Verify region classification
    assert PopulationAnalyticsService.classify_person_region(p_ain) == "AL_AIN"
    assert PopulationAnalyticsService.classify_person_region(p_dhafra) == "AL_DHAFRA"
    assert PopulationAnalyticsService.classify_person_region(p_central) == "AD_CENTRAL"

    # Patient with no district and unclassified org is NOT arbitrarily dumped into AD_CENTRAL
    p_unclass = Person(
        first_name="Rashid",
        last_name="Unknown",
        date_of_birth=date.today() - timedelta(days=365 * 20),
        gender=Gender.MALE,
        district=None,
    )
    db_session.add(p_unclass)
    db_session.commit()
    assert PopulationAnalyticsService.classify_person_region(p_unclass) is None

    # Verify Equity Analytics data completeness calculation & explicit methodology
    equity = PopulationAnalyticsService.get_equity_analytics(db=db_session, organization_id=org.id)
    assert equity.benchmark_region == "AD_CENTRAL"
    assert equity.benchmark_methodology is not None
    assert "ADPHC" in equity.benchmark_methodology
    assert "AD_CENTRAL" in equity.data_completeness_by_subgroup
    assert "AL_AIN" in equity.data_completeness_by_subgroup
    assert "AL_DHAFRA" in equity.data_completeness_by_subgroup

    # Verify that completeness is truthful (not hardcoded 95.0)
    assert equity.data_completeness_by_subgroup["AL_AIN"] == 85.0
    assert equity.data_completeness_by_subgroup["AL_DHAFRA"] == 70.0
    assert equity.data_completeness_by_subgroup["AD_CENTRAL"] >= 70.0

