from datetime import date, datetime, timezone
import pytest
from sqlalchemy.orm import Session
from backend.app.models.person import (
    Person,
    PersonIdentifier,
    PatientProfile,
    Gender,
    RecordStatus,
    PopulationClassification,
    IdentifierNamespace,
)
from backend.app.models.clinical import ObservationType
from backend.app.models.organization import Organization
from backend.app.models.user import User, UserRole
from backend.app.models.guardian import Guardianship, GuardianshipStatus, RelationshipType
from backend.app.models.provenance import DataSource, SourceType, TrustLevel
from backend.app.models.screening import EpistemicStatus
from backend.app.services.person_service import PersonService


def test_person_created_without_user_account(db_session: Session):
    # Setup clinic organization
    org = Organization(name="Abu Dhabi Family Health Center", license_number="DOH-CLINIC-001", organization_type="CLINIC", region="Abu Dhabi")
    db_session.add(org)
    db_session.commit()

    # Create a 4-year-old child Person without any platform login account
    dob = date(date.today().year - 4, 5, 10)
    person = PersonService.register_person(
        db=db_session,
        first_name="Rashid",
        last_name="Al Mansoori",
        date_of_birth=dob,
        gender=Gender.MALE,
        organization_id=org.id,
        user_id=None,  # No login account!
        primary_identifier=(IdentifierNamespace.EMIRATES_ID, "784-2022-1234567-1")
    )

    assert person.id is not None
    assert person.user_id is None
    assert person.full_name == "Rashid Al Mansoori"
    assert person.age_years == 4
    assert person.patient_profile is not None
    assert person.patient_profile.population_classification == PopulationClassification.PEDIATRIC
    assert len(person.identifiers) == 1
    assert person.identifiers[0].namespace == IdentifierNamespace.EMIRATES_ID
    assert person.identifiers[0].identifier_hash == PersonService.hash_identifier("784-2022-1234567-1")


def test_clinical_observation_with_provenance_and_unit_conversion(db_session: Session):
    org = Organization(name="Al Ain Medical Clinic", license_number="DOH-CLINIC-002", organization_type="CLINIC", region="Al Ain")
    db_session.add(org)
    db_session.commit()

    # Data Source (e.g. Clinic EMR)
    data_source = DataSource(
        source_type=SourceType.CLINIC_EMR,
        organization_id=org.id,
        system_name="SEHA Epic EHR",
        trust_level=TrustLevel.ACCREDITED_CLINICAL,
    )
    db_session.add(data_source)
    db_session.commit()

    # Register adult person
    dob = date(1985, 4, 15)
    person = PersonService.register_person(
        db=db_session,
        first_name="Fatima",
        last_name="Al Zaabi",
        date_of_birth=dob,
        gender=Gender.FEMALE,
        organization_id=org.id,
    )
    assert person.patient_profile.population_classification == PopulationClassification.ADULT

    # Record blood glucose in mmol/L (e.g. 6.0 mmol/L should be converted to ~108.1 mg/dL)
    obs = PersonService.record_clinical_observation(
        db=db_session,
        person_id=person.id,
        observation_type=ObservationType.LAB_RESULT,
        code="2339-0",
        display_name="Fasting Blood Glucose",
        numeric_value=6.0,
        unit="mmol/L",
        collection_time=datetime.now(timezone.utc),
        epistemic_status=EpistemicStatus.VERIFIED,
        data_source_id=data_source.id
    )

    assert obs.id is not None
    assert obs.numeric_value == 6.0
    assert obs.unit == "mmol/L"
    assert obs.canonical_unit == "mg/dL"
    assert obs.canonical_value == pytest.approx(108.1, abs=0.2)
    assert obs.provenance_id is not None
    assert obs.provenance.data_source_id == data_source.id
    assert "Normalized" in (obs.provenance.transformation_history or "")


def test_cross_organization_boundary_blocked(db_session: Session):
    org_a = Organization(name="Clinic A", license_number="DOH-001", organization_type="CLINIC", region="Abu Dhabi")
    org_b = Organization(name="Clinic B", license_number="DOH-002", organization_type="CLINIC", region="Abu Dhabi")
    db_session.add_all([org_a, org_b])
    db_session.commit()

    person = PersonService.register_person(
        db=db_session,
        first_name="Ali",
        last_name="Al Nuaimi",
        date_of_birth=date(1990, 1, 1),
        gender=Gender.MALE,
        organization_id=org_a.id
    )

    # Provider from Clinic B attempts to read Clinic A's patient
    with pytest.raises(PermissionError, match="Cross-organization boundary violation"):
        PersonService.get_person_by_id(
            db=db_session,
            person_id=person.id,
            requesting_user_id="prov-user-b",
            requesting_role=UserRole.HEALTHCARE_PROVIDER.value,
            requesting_org_id=org_b.id
        )

    # Provider from Clinic A should succeed
    retrieved = PersonService.get_person_by_id(
        db=db_session,
        person_id=person.id,
        requesting_user_id="prov-user-a",
        requesting_role=UserRole.HEALTHCARE_PROVIDER.value,
        requesting_org_id=org_a.id
    )
    assert retrieved.id == person.id


def test_system_admin_blocked_from_patient_phi(db_session: Session):
    org = Organization(name="General Clinic", license_number="DOH-003", organization_type="CLINIC", region="Abu Dhabi")
    db_session.add(org)
    db_session.commit()

    person = PersonService.register_person(
        db=db_session,
        first_name="Sultan",
        last_name="Al Shamsi",
        date_of_birth=date(1995, 6, 20),
        gender=Gender.MALE,
        organization_id=org.id
    )

    # System administrator must be blocked from clinical PHI
    with pytest.raises(PermissionError, match="do not have access to individual patient clinical records"):
        PersonService.get_person_by_id(
            db=db_session,
            person_id=person.id,
            requesting_user_id="admin-user-1",
            requesting_role=UserRole.SYSTEM_ADMIN.value,
            requesting_org_id=org.id
        )


def test_guardian_access_to_dependent_person(db_session: Session):
    org = Organization(name="Pediatric Health Hub", license_number="DOH-004", organization_type="CLINIC", region="Abu Dhabi")
    parent_user = User(email="parent@wellwatch.ae", full_name="Amina Al Dhaheri", password_hash="pw", role=UserRole.PARENT_GUARDIAN.value, organization_id=org.id)
    other_user = User(email="stranger@wellwatch.ae", full_name="Other User", password_hash="pw", role=UserRole.PARENT_GUARDIAN.value, organization_id=org.id)
    db_session.add_all([org, parent_user, other_user])
    db_session.commit()

    child_person = PersonService.register_person(
        db=db_session,
        first_name="Maryam",
        last_name="Al Dhaheri",
        date_of_birth=date(date.today().year - 2, 8, 1),
        gender=Gender.FEMALE,
        organization_id=org.id
    )

    # Establish verified guardianship linking parent_user to child_person
    guardianship = Guardianship(
        guardian_user_id=parent_user.id,
        dependent_user_id=parent_user.id,  # Fallback user id
        guardian_person_id=None,
        dependent_person_id=child_person.id,
        relationship_type=RelationshipType.MOTHER.value,
        verification_status=GuardianshipStatus.VERIFIED.value,
    )
    db_session.add(guardianship)
    db_session.commit()

    # Verified parent can access
    child = PersonService.get_person_by_id(
        db=db_session,
        person_id=child_person.id,
        requesting_user_id=parent_user.id,
        requesting_role=UserRole.PARENT_GUARDIAN.value,
        requesting_org_id=org.id
    )
    assert child.id == child_person.id

    # Unrelated user cannot access child
    with pytest.raises(PermissionError, match="You are not a verified guardian"):
        PersonService.get_person_by_id(
            db=db_session,
            person_id=child_person.id,
            requesting_user_id=other_user.id,
            requesting_role=UserRole.PARENT_GUARDIAN.value,
            requesting_org_id=org.id
        )
