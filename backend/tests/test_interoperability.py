import uuid
import pytest
from datetime import date, datetime, timedelta
from sqlalchemy.orm import Session
from fastapi import HTTPException

from backend.app.models.organization import Organization
from backend.app.models.user import User, UserRole
from backend.app.models.person import Person, PersonIdentifier, IdentifierNamespace, Gender
from backend.app.models.clinical import ClinicalObservation, ObservationType
from backend.app.models.screening import ScreeningEncounter, EncounterType, EpistemicStatus
from backend.app.models.integration import ProvenanceRecord
from backend.app.services.person_service import PersonService
from backend.app.integrations.fhir.serializer import FhirSerializer
from backend.app.integrations.terminology.mapper import TerminologyMapper
from backend.app.integrations.adapters.malaffi import MalaffiAdapter
from backend.app.services.interoperability_service import InteroperabilityService


def _setup_org_admin_patient(db: Session):
    uid = uuid.uuid4().hex[:6]
    org = Organization(name=f"SEHA Al Mushrif HIE {uid}", license_number=f"DOH-HIE-{uid}", organization_type="PRIMARY_CARE_CLINIC", region="Abu Dhabi")
    db.add(org)
    db.commit()

    admin = User(
        email=f"hie.admin.{uid}@wellwatch.ae",
        password_hash="pw",
        full_name="Interoperability Lead",
        role=UserRole.SYSTEM_ADMIN,
        organization_id=org.id,
        is_active=True,
    )
    db.add(admin)
    db.commit()

    patient = PersonService.register_person(
        db=db,
        first_name="Sheikha",
        last_name="Al Rumaithi",
        date_of_birth=date.today() - timedelta(days=365 * 30),
        gender=Gender.FEMALE,
        organization_id=org.id,
    )
    return org, admin, patient


def test_fhir_patient_serialization(db_session: Session):
    """Verifies FHIR R4 Patient resource serialization compliant with DoH standards."""
    org, admin, patient = _setup_org_admin_patient(db_session)

    ident = PersonIdentifier(
        person_id=patient.id,
        namespace=IdentifierNamespace.EMIRATES_ID,
        identifier_value="784-1988-1234567-1",
    )
    fhir_pat = FhirSerializer.serialize_patient(patient, [ident])

    assert fhir_pat["resourceType"] == "Patient"
    assert fhir_pat["id"] == patient.id
    assert fhir_pat["identifier"][0]["system"] == "http://fhir.doh.gov.ae/identifier/emirates-id"
    assert fhir_pat["identifier"][0]["value"] == "784-1988-1234567-1"
    assert fhir_pat["name"][0]["family"] == patient.last_name


def test_fhir_observation_serialization_with_loinc():
    """Verifies FHIR R4 Observation resource serialization with canonical LOINC codes."""
    obs = ClinicalObservation(
        patient_id="pat-test-123",
        observation_type=ObservationType.VITAL_SIGN,
        code_system="LOINC",
        code="8480-6",
        display_name="Systolic Blood Pressure",
        numeric_value=125.0,
        unit="mm[Hg]",
        collection_time=datetime.utcnow(),
    )
    fhir_obs = FhirSerializer.serialize_observation(obs)

    assert fhir_obs["resourceType"] == "Observation"
    assert fhir_obs["status"] == "final"
    assert fhir_obs["code"]["coding"][0]["system"] == "http://loinc.org"
    assert fhir_obs["code"]["coding"][0]["code"] == "8480-6"
    assert fhir_obs["valueQuantity"]["value"] == 125.0
    assert fhir_obs["valueQuantity"]["unit"] == "mm[Hg]"


def test_governed_terminology_mapper():
    """Verifies governed bidirectional terminology mappings for LOINC, SNOMED, and ICD-10."""
    # Internal to LOINC
    bp_map = TerminologyMapper.lookup_internal("SYSTOLIC_BP")
    assert bp_map is not None
    assert bp_map["target_code"] == "8480-6"
    assert bp_map["ucum_unit"] == "mm[Hg]"

    # Reverse LOINC to Internal
    rev_map = TerminologyMapper.reverse_lookup("http://loinc.org", "8480-6")
    assert rev_map is not None
    assert rev_map["internal_code"] == "SYSTOLIC_BP"

    # Unmapped code returns None (strict honesty)
    assert TerminologyMapper.lookup_internal("NON_EXISTENT_CODE") is None


def test_malaffi_adapter_honest_not_configured_boundary():
    """Verifies Malaffi adapter reports honest NOT_CONFIGURED status when credentials are unset."""
    adapter = MalaffiAdapter()
    assert adapter.get_status() == "NOT_CONFIGURED"

    success, message = adapter.test_connection()
    assert success is False
    assert "NOT_CONFIGURED" in message
    assert len(adapter.get_prerequisites()) >= 4


def test_fhir_patient_bundle_export(db_session: Session):
    """Verifies outbound FHIR R4 Bundle export with authorized patient and clinical vitals."""
    org, admin, patient = _setup_org_admin_patient(db_session)

    res = InteroperabilityService.export_patient_bundle(
        db=db_session, patient_id=patient.id, user=admin
    )
    assert res.patient_id == patient.id
    bundle = res.bundle
    assert bundle["resourceType"] == "Bundle"
    assert bundle["total"] >= 1


def test_fhir_bundle_ingestion_with_identity_and_provenance(db_session: Session):
    """Verifies external FHIR R4 Bundle ingestion, Emirates ID matching, and provenance recording."""
    org, admin, patient = _setup_org_admin_patient(db_session)

    # Link Emirates ID to patient
    eid_val = "784-1985-7654321-1"
    ident = PersonIdentifier(
        person_id=patient.id,
        namespace=IdentifierNamespace.EMIRATES_ID,
        identifier_value=eid_val,
    )
    db_session.add(ident)
    db_session.commit()

    # Construct incoming FHIR bundle from external hospital
    bundle = {
        "resourceType": "Bundle",
        "type": "collection",
        "entry": [
            {
                "resource": {
                    "resourceType": "Patient",
                    "identifier": [
                        {
                            "system": "http://fhir.doh.gov.ae/identifier/emirates-id",
                            "value": eid_val,
                        }
                    ],
                }
            },
            {
                "resource": {
                    "resourceType": "Observation",
                    "id": "ext-obs-999",
                    "status": "final",
                    "code": {
                        "coding": [
                            {"system": "http://loinc.org", "code": "8480-6", "display": "Systolic BP"}
                        ]
                    },
                    "valueQuantity": {"value": 138.0, "unit": "mm[Hg]"},
                }
            },
        ],
    }

    ingest_res = InteroperabilityService.ingest_fhir_bundle(
        db=db_session,
        bundle=bundle,
        source_system="SEHA_HOSPITAL_EMR",
        user=admin,
    )

    assert ingest_res.status == "SUCCESS"
    assert ingest_res.patient_id == patient.id
    assert ingest_res.observations_created == 1

    # Verify Provenance Record was persisted
    prov = (
        db_session.query(ProvenanceRecord)
        .filter(ProvenanceRecord.source_system == "SEHA_HOSPITAL_EMR")
        .first()
    )
    assert prov is not None
    assert prov.target_resource_type == "ClinicalObservation"


def test_integration_monitoring_endpoint(db_session: Session):
    """Verifies integration monitoring overview reporting provider health."""
    org, admin, patient = _setup_org_admin_patient(db_session)

    data = InteroperabilityService.get_monitoring_overview(db=db_session)
    assert data.total_providers >= 4
    assert len(data.providers) >= 4
