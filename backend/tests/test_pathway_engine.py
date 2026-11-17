from datetime import date, timedelta
import pytest
from sqlalchemy.orm import Session

from backend.app.models.clinical import ScreeningRecord, ScreeningStatus
from backend.app.models.organization import Organization
from backend.app.models.person import Gender
from backend.app.models.screening import EpistemicStatus
from backend.app.services.pathway_engine import CareGapStatus, EligibilityStatus, PathwayEngine
from backend.app.services.person_service import PersonService


def test_list_available_pathways():
    pathways = PathwayEngine.list_available_pathways()
    assert len(pathways) >= 2
    ids = [p["pathway_id"] for p in pathways]
    assert "DOH-PEDIATRIC-WELLCHILD-V1" in ids
    assert "ADPHC-ADULT-IFHAS-V1" in ids


def test_pediatric_eligibility_and_milestone_evaluation(db_session: Session):
    org = Organization(name="Al Rahba Clinic", license_number="DOH-020", organization_type="CLINIC", region="Abu Dhabi")
    db_session.add(org)
    db_session.commit()

    # 12-month-old child
    dob = date(2025, 1, 15)
    eval_date = date(2026, 1, 15)  # Exactly 12 months old
    child = PersonService.register_person(
        db=db_session,
        first_name="Salem",
        last_name="Al Ameri",
        date_of_birth=dob,
        gender=Gender.MALE,
        organization_id=org.id
    )

    pathway = PathwayEngine.load_pathway("DOH-PEDIATRIC-WELLCHILD-V1")
    assert pathway is not None

    # Eligibility Check
    res = PathwayEngine.evaluate_eligibility(pathway=pathway, person=child, evaluation_date=eval_date)
    assert res["status"] == EligibilityStatus.ELIGIBLE.value
    assert res["is_eligible"] is True
    assert res["age_evaluated"]["months"] == 12

    # Care Gap Check: 12M visit should be DUE
    gaps = PathwayEngine.evaluate_care_gaps(db=db_session, pathway=pathway, person=child, evaluation_date=eval_date)
    m12 = next(g for g in gaps if g["milestone_id"] == "WC-12M")
    assert m12["status"] == CareGapStatus.DUE.value

    # Future 36M visit should be NOT_APPLICABLE
    m36 = next(g for g in gaps if g["milestone_id"] == "WC-36M")
    assert m36["status"] == CareGapStatus.NOT_APPLICABLE.value


def test_pediatric_ineligible_for_adult_pathway(db_session: Session):
    org = Organization(name="Clinic Alpha", license_number="DOH-021", organization_type="CLINIC", region="Abu Dhabi")
    db_session.add(org)
    db_session.commit()

    child = PersonService.register_person(
        db=db_session,
        first_name="Amna",
        last_name="Al Marri",
        date_of_birth=date(2023, 6, 1),
        gender=Gender.FEMALE,
        organization_id=org.id
    )

    adult_pathway = PathwayEngine.load_pathway("ADPHC-ADULT-IFHAS-V1")
    res = PathwayEngine.evaluate_eligibility(pathway=adult_pathway, person=child)
    assert res["status"] == EligibilityStatus.NOT_ELIGIBLE.value
    assert res["is_eligible"] is False
    assert "below adult boundary" in res["explanation"]


def test_adult_ifhas_completed_screening_within_cycle(db_session: Session):
    org = Organization(name="Mubadala Clinic", license_number="DOH-022", organization_type="CLINIC", region="Abu Dhabi")
    db_session.add(org)
    db_session.commit()

    adult = PersonService.register_person(
        db=db_session,
        first_name="Abdullah",
        last_name="Al Kaabi",
        date_of_birth=date(1982, 3, 10),
        gender=Gender.MALE,
        organization_id=org.id
    )

    eval_date = date(2026, 6, 1)

    # Completed Diabetes IFHAS screening 6 months prior
    scr = ScreeningRecord(
        patient_id=adult.id,
        screening_type="IFHAS-DIABETES",
        screening_date=date(2025, 12, 1),
        screening_status=ScreeningStatus.COMPLETED,
        findings_summary="Fasting glucose 92 mg/dL, HbA1c 5.3%",
        epistemic_status=EpistemicStatus.VERIFIED
    )
    db_session.add(scr)
    db_session.commit()

    pathway = PathwayEngine.load_pathway("ADPHC-ADULT-IFHAS-V1")
    gaps = PathwayEngine.evaluate_care_gaps(db=db_session, pathway=pathway, person=adult, evaluation_date=eval_date)
    diabetes_gap = next(g for g in gaps if g["milestone_id"] == "IFHAS-DIABETES")
    
    assert diabetes_gap["status"] == CareGapStatus.COMPLETED.value
    assert diabetes_gap["months_since"] == 6


def test_adult_screening_repeat_due_after_3_year_cycle(db_session: Session):
    org = Organization(name="Zayed Military Hospital", license_number="DOH-023", organization_type="HOSPITAL", region="Abu Dhabi")
    db_session.add(org)
    db_session.commit()

    adult = PersonService.register_person(
        db=db_session,
        first_name="Omar",
        last_name="Al Blooshi",
        date_of_birth=date(1978, 1, 1),
        gender=Gender.MALE,
        organization_id=org.id
    )

    eval_date = date(2026, 6, 1)

    # Previous screening was 42 months ago (> 36 months IFHAS interval)
    scr = ScreeningRecord(
        patient_id=adult.id,
        screening_type="IFHAS-CARDIOVASCULAR",
        screening_date=date(2022, 12, 1),
        screening_status=ScreeningStatus.COMPLETED,
        findings_summary="BP 125/80 mmHg",
        epistemic_status=EpistemicStatus.VERIFIED
    )
    db_session.add(scr)
    db_session.commit()

    pathway = PathwayEngine.load_pathway("ADPHC-ADULT-IFHAS-V1")
    gaps = PathwayEngine.evaluate_care_gaps(db=db_session, pathway=pathway, person=adult, evaluation_date=eval_date)
    cvd_gap = next(g for g in gaps if g["milestone_id"] == "IFHAS-CARDIOVASCULAR")
    
    # Exceeded 36 months -> DUE for repeat screening
    assert cvd_gap["status"] == CareGapStatus.DUE.value
    assert cvd_gap["months_since"] == 42


def test_gender_specific_screening_exclusion(db_session: Session):
    org = Organization(name="Reem Island Medical", license_number="DOH-024", organization_type="CLINIC", region="Abu Dhabi")
    db_session.add(org)
    db_session.commit()

    male_adult = PersonService.register_person(
        db=db_session,
        first_name="Sultan",
        last_name="Al Mehairi",
        date_of_birth=date(1980, 5, 20),
        gender=Gender.MALE,
        organization_id=org.id
    )

    pathway = PathwayEngine.load_pathway("ADPHC-ADULT-IFHAS-V1")
    gaps = PathwayEngine.evaluate_care_gaps(db=db_session, pathway=pathway, person=male_adult)
    
    # Breast & Cervical screening must be NOT_APPLICABLE for Male
    breast = next(g for g in gaps if g["milestone_id"] == "IFHAS-BREAST-CANCER")
    cervical = next(g for g in gaps if g["milestone_id"] == "IFHAS-CERVICAL-CANCER")
    
    assert breast["status"] == CareGapStatus.NOT_APPLICABLE.value
    assert cervical["status"] == CareGapStatus.NOT_APPLICABLE.value


def test_reproducibility_and_determinism(db_session: Session):
    org = Organization(name="Testing Facility", license_number="DOH-025", organization_type="CLINIC", region="Abu Dhabi")
    db_session.add(org)
    db_session.commit()

    person = PersonService.register_person(
        db=db_session,
        first_name="Huda",
        last_name="Al Shamsi",
        date_of_birth=date(1995, 7, 14),
        gender=Gender.FEMALE,
        organization_id=org.id
    )

    pathway = PathwayEngine.load_pathway("ADPHC-ADULT-IFHAS-V1")
    eval_date = date(2026, 11, 1)

    # 10 evaluations on the same input must produce identical results
    results = [PathwayEngine.evaluate_eligibility(pathway=pathway, person=person, evaluation_date=eval_date) for _ in range(10)]
    for r in results:
        assert r == results[0]
