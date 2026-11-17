from datetime import date, datetime, timedelta, timezone
import pytest
from sqlalchemy.orm import Session

from backend.app.models.clinical import Encounter, EncounterStatus, EncounterType, ObservationType, ScreeningRecord, ScreeningStatus
from backend.app.models.organization import Organization
from backend.app.models.person import Gender, IdentifierNamespace, PopulationClassification
from backend.app.models.screening import EpistemicStatus
from backend.app.models.user import User, UserRole
from backend.app.services.person_service import PersonService


def test_patient_registration_pediatric_and_adult(db_session: Session):
    org = Organization(name="Cleveland Clinic Abu Dhabi", license_number="DOH-HOSP-001", organization_type="HOSPITAL", region="Abu Dhabi")
    db_session.add(org)
    db_session.commit()

    # 1. Register Child (3 years old)
    dob_child = date(date.today().year - 3, 2, 10)
    child = PersonService.register_person(
        db=db_session,
        first_name="Zayed",
        last_name="Al Nahyan",
        date_of_birth=dob_child,
        gender=Gender.MALE,
        organization_id=org.id,
        primary_identifier=(IdentifierNamespace.EMIRATES_ID, "784-2023-9998881-1")
    )
    assert child.patient_profile.population_classification == PopulationClassification.PEDIATRIC
    assert child.age_years == 3

    # 2. Register Adult (45 years old)
    dob_adult = date(date.today().year - 45, 8, 20)
    adult = PersonService.register_person(
        db=db_session,
        first_name="Mansoor",
        last_name="Al Mazrouei",
        date_of_birth=dob_adult,
        gender=Gender.MALE,
        organization_id=org.id,
        primary_identifier=(IdentifierNamespace.EMIRATES_ID, "784-1981-5554443-2")
    )
    assert adult.patient_profile.population_classification == PopulationClassification.ADULT
    assert adult.age_years == 45


def test_duplicate_emirates_id_rejected(db_session: Session):
    org = Organization(name="Sheikh Shakhbout Medical City", license_number="DOH-HOSP-002", organization_type="HOSPITAL", region="Abu Dhabi")
    db_session.add(org)
    db_session.commit()

    PersonService.register_person(
        db=db_session,
        first_name="Hamad",
        last_name="Al Rumaithi",
        date_of_birth=date(1992, 5, 1),
        gender=Gender.MALE,
        organization_id=org.id,
        primary_identifier=(IdentifierNamespace.EMIRATES_ID, "784-1992-1112223-1")
    )

    # Attempting to register another person with the identical Emirates ID must raise ValueError
    with pytest.raises(ValueError, match="Duplicate registration rejected"):
        PersonService.register_person(
            db=db_session,
            first_name="Duplicate",
            last_name="Person",
            date_of_birth=date(1994, 2, 2),
            gender=Gender.MALE,
            organization_id=org.id,
            primary_identifier=(IdentifierNamespace.EMIRATES_ID, "784-1992-1112223-1")
        )


def test_patient_directory_search_and_pagination(db_session: Session):
    org = Organization(name="Al Dhafra Clinic", license_number="DOH-CLINIC-010", organization_type="CLINIC", region="Al Dhafra")
    db_session.add(org)
    db_session.commit()

    # Create 3 patients
    PersonService.register_person(db=db_session, first_name="Khalid", last_name="Al Qasimi", date_of_birth=date(1980, 1, 1), gender=Gender.MALE, organization_id=org.id, district="Madinat Zayed")
    PersonService.register_person(db=db_session, first_name="Noura", last_name="Al Suwaidi", date_of_birth=date(2020, 3, 1), gender=Gender.FEMALE, organization_id=org.id, district="Ghayathi")
    PersonService.register_person(db=db_session, first_name="Saeed", last_name="Al Marzouqi", date_of_birth=date(1955, 6, 1), gender=Gender.MALE, organization_id=org.id, district="Liwa")

    # 1. Search by name 'Noura'
    items, total = PersonService.list_persons(
        db=db_session,
        requesting_role=UserRole.HEALTHCARE_PROVIDER.value,
        requesting_org_id=org.id,
        search_query="Noura"
    )
    assert total == 1
    assert items[0].first_name == "Noura"

    # 2. Filter by PEDIATRIC classification
    ped_items, ped_total = PersonService.list_persons(
        db=db_session,
        requesting_role=UserRole.HEALTHCARE_PROVIDER.value,
        requesting_org_id=org.id,
        classification_filter="PEDIATRIC"
    )
    assert ped_total == 1
    assert ped_items[0].first_name == "Noura"


def test_pediatric_and_adult_profile_data(db_session: Session):
    org = Organization(name="Danat Al Emarat Hospital", license_number="DOH-HOSP-003", organization_type="HOSPITAL", region="Abu Dhabi")
    db_session.add(org)
    db_session.commit()

    # Create Child
    child = PersonService.register_person(
        db=db_session,
        first_name="Salama",
        last_name="Al Ketbi",
        date_of_birth=date(date.today().year - 2, 4, 10),
        gender=Gender.FEMALE,
        organization_id=org.id
    )

    # Add Well-Child screening
    scr = ScreeningRecord(
        patient_id=child.id,
        screening_type="WELL_CHILD_18M",
        screening_date=date.today() - timedelta(days=60),
        screening_status=ScreeningStatus.COMPLETED,
        findings_summary="Normal developmental milestones observed",
        epistemic_status=EpistemicStatus.VERIFIED
    )
    db_session.add(scr)
    db_session.commit()

    ped_profile = PersonService.get_pediatric_profile(db=db_session, person=child)
    assert len(ped_profile["well_child_visits"]) == 1
    assert ped_profile["well_child_visits"][0]["screening_type"] == "WELL_CHILD_18M"


def test_longitudinal_timeline_aggregation(db_session: Session):
    org = Organization(name="Mubadala Health Hub", license_number="DOH-HOSP-004", organization_type="HOSPITAL", region="Abu Dhabi")
    db_session.add(org)
    db_session.commit()

    person = PersonService.register_person(
        db=db_session,
        first_name="Tariq",
        last_name="Al Hashimi",
        date_of_birth=date(1975, 10, 5),
        gender=Gender.MALE,
        organization_id=org.id
    )

    now = datetime.now(timezone.utc)

    # 1. Add Encounter
    enc = Encounter(
        patient_id=person.id,
        facility_id=org.id,
        encounter_type=EncounterType.OUTPATIENT_PREVENTIVE,
        encounter_date=now - timedelta(days=30),
        status=EncounterStatus.COMPLETED,
        clinical_notes="Annual checkup"
    )
    db_session.add(enc)

    # 2. Add Observation
    obs = PersonService.record_clinical_observation(
        db=db_session,
        person_id=person.id,
        observation_type=ObservationType.VITAL_SIGN,
        code="8480-6",
        display_name="Systolic BP",
        numeric_value=122.0,
        unit="mmHg",
        collection_time=now - timedelta(days=10)
    )

    timeline = PersonService.get_longitudinal_timeline(db=db_session, person_id=person.id)
    assert len(timeline) >= 2
    # Verify descending chronological order
    assert timeline[0].timestamp >= timeline[1].timestamp
