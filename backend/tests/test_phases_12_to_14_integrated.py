import pytest
import uuid
from datetime import date, datetime, timedelta
from sqlalchemy.orm import Session

from backend.app.models.organization import Organization
from backend.app.models.user import User, UserRole
from backend.app.models.person import Person, PersonIdentifier, IdentifierNamespace, Gender
from backend.app.models.clinical import ScreeningRecord, ScreeningStatus
from backend.app.models.screening import EpistemicStatus
from backend.app.models.recommendation import PreventiveRecommendation, RecommendationLifecycleStatus
from backend.app.models.navigation import Appointment, CareGapClosureEvidence, AppointmentStatus
from backend.app.models.offline import SyncConflict, ConflictResolutionStatus
from backend.app.models.integration import ProvenanceRecord
from backend.app.services.person_service import PersonService
from backend.app.services.population_analytics_service import PopulationAnalyticsService
from backend.app.services.offline_sync_service import OfflineSyncService
from backend.app.services.interoperability_service import InteroperabilityService
from backend.app.schemas.offline import OfflineSyncBatchRequest, OfflineSyncOperationRequest, ResolveConflictRequest


def _setup_full_environment(db: Session):
    uid = uuid.uuid4().hex[:6]
    org = Organization(name=f"Integrated Health {uid}", license_number=f"DOH-INT-{uid}", organization_type="HOSPITAL", region="Abu Dhabi")
    db.add(org)
    db.commit()

    admin = User(
        email=f"int.admin.{uid}@wellwatch.ae",
        password_hash="pw",
        full_name="Dr. Hessa Al Dhaheri",
        role=UserRole.SYSTEM_ADMIN,
        organization_id=org.id,
        is_active=True,
    )
    chw = User(
        email=f"int.chw.{uid}@wellwatch.ae",
        password_hash="pw",
        full_name="Khadija Al Mazrouei (CHW)",
        role=UserRole.COMMUNITY_HEALTH_WORKER,
        organization_id=org.id,
        is_active=True,
    )
    doctor = User(
        email=f"int.doc.{uid}@wellwatch.ae",
        password_hash="pw",
        full_name="Dr. Sultan Al Nuaimi",
        role=UserRole.HEALTHCARE_PROVIDER,
        organization_id=org.id,
        is_active=True,
    )
    db.add_all([admin, chw, doctor])
    db.commit()

    patient = PersonService.register_person(
        db=db,
        first_name="Salem",
        last_name="Al Mansoori",
        date_of_birth=date.today() - timedelta(days=365 * 38),
        gender=Gender.MALE,
        organization_id=org.id,
    )
    return org, admin, chw, doctor, patient


def test_cross_phase_workflow_a_population_intelligence(db_session: Session):
    """Workflow A: Database records -> Pathway -> Recommendations -> Navigation -> Verified Population Metrics."""
    org, admin, chw, doctor, patient = _setup_full_environment(db_session)

    # 1. Add pediatric child
    child = PersonService.register_person(
        db=db_session,
        first_name="Hamad",
        last_name="Al Mazrouei",
        date_of_birth=date.today() - timedelta(days=365 * 3),
        gender=Gender.MALE,
        organization_id=org.id,
    )

    # 2. Add an approved recommendation and appointment for adult
    rec = PreventiveRecommendation(
        organization_id=org.id,
        person_id=patient.id,
        rule_id="RULE-ADULT-IFHAS-001",
        rule_version="1.0.0",
        recommendation_category="SCREENING",
        title="ADPHC IFHAS Screening",
        clinical_justification="Adult preventive baseline overdue",
        priority="HIGH",
        lifecycle_status=RecommendationLifecycleStatus.APPROVED,
    )
    db_session.add(rec)
    db_session.commit()

    appt = Appointment(
        organization_id=org.id,
        person_id=patient.id,
        recommendation_id=rec.id,
        scheduled_start=datetime.utcnow() - timedelta(days=2),
        scheduled_end=datetime.utcnow() - timedelta(days=2) + timedelta(minutes=30),
        status=AppointmentStatus.ATTENDED,
    )
    db_session.add(appt)
    db_session.commit()

    # 3. Add care gap closure evidence
    evidence = CareGapClosureEvidence(
        organization_id=org.id,
        recommendation_id=rec.id,
        person_id=patient.id,
        closure_method="ENCOUNTER_COMPLETION",
        evidence_reference_type="ENCOUNTER",
        evidence_reference_id="ENC-SAMPLE-001",
        recorded_at=datetime.utcnow(),
    )
    db_session.add(evidence)
    db_session.commit()

    # 4. Compute Population Metrics
    overview = PopulationAnalyticsService.get_overview(
        db=db_session, organization_id=org.id, time_window_days=90
    )

    assert overview.total_registered >= 2
    assert overview.pediatric_population >= 1
    assert overview.recommendations_approved >= 1
    assert overview.appointments_attended >= 1
    assert overview.care_gaps_closed >= 1


def test_cross_phase_workflow_b_offline_community_screening(db_session: Session):
    """Workflow B: CHW assignment -> Offline field capture -> Revalidation -> Sync -> Observation persisted."""
    org, admin, chw, doctor, patient = _setup_full_environment(db_session)

    # 1. Download offline package
    pkg = OfflineSyncService.get_assignment_package(db=db_session, user=chw)
    assert len(pkg.assigned_patients) > 0

    # 2. Record offline screening
    op_id = f"OP-INT-WF-B-{uuid.uuid4().hex[:6]}"
    batch = OfflineSyncBatchRequest(
        device_id="DEV-TABLET-CHW-01",
        operations=[
            OfflineSyncOperationRequest(
                operation_id=op_id,
                idempotency_key=f"IDEMP-{op_id}",
                resource_type="SCREENING",
                operation_type="CREATE",
                client_timestamp=datetime.utcnow(),
                payload={
                    "patient_id": patient.id,
                    "encounter_type": "WELL_CHILD_VISIT",
                    "blood_pressure_systolic": 110,
                    "blood_pressure_diastolic": 70,
                    "clinical_notes": "Well child visit field measurement",
                },
            )
        ],
    )

    # 3. Process Sync
    sync_res = OfflineSyncService.process_sync_batch(db=db_session, user=chw, batch=batch)
    assert sync_res.synced_count == 1
    screening_id = sync_res.results[0].resource_id

    # 4. Verify Database Persistence
    screening = db_session.query(ScreeningRecord).filter(ScreeningRecord.id == screening_id).first()
    assert screening is not None
    assert screening.screening_type == "WELL_CHILD_VISIT"


def test_cross_phase_workflow_c_external_laboratory_ingestion(db_session: Session):
    """Workflow C: External FHIR Lab Observation Ingestion -> Terminology Mapped -> Provenance Recorded."""
    org, admin, chw, doctor, patient = _setup_full_environment(db_session)

    # 1. Link Emirates ID
    eid_val = "784-1992-9988776-1"
    ident = PersonIdentifier(
        person_id=patient.id,
        identifier_namespace=IdentifierNamespace.EMIRATES_ID,
        identifier_value=eid_val,
    )
    db_session.add(ident)
    db_session.commit()

    # 2. External Lab FHIR Bundle
    lab_bundle = {
        "resourceType": "Bundle",
        "type": "transaction",
        "entry": [
            {
                "resource": {
                    "resourceType": "Patient",
                    "identifier": [{"system": "http://fhir.doh.gov.ae/identifier/emirates-id", "value": eid_val}],
                }
            },
            {
                "resource": {
                    "resourceType": "Observation",
                    "id": "LAB-FASTING-GLUCOSE-01",
                    "status": "final",
                    "code": {
                        "coding": [
                            {"system": "http://loinc.org", "code": "15074-8", "display": "Fasting Blood Glucose"}
                        ]
                    },
                    "valueQuantity": {"value": 105.0, "unit": "mg/dL"},
                }
            },
        ],
    }

    # 3. Ingest Lab Result
    res = InteroperabilityService.ingest_fhir_bundle(
        db=db_session,
        bundle=lab_bundle,
        source_system="ABU_DHABI_NATIONAL_REFERENCE_LAB",
        user=admin,
    )

    assert res.status == "SUCCESS"
    assert res.observations_created == 1

    # 4. Verify Provenance
    prov = (
        db_session.query(ProvenanceRecord)
        .filter(ProvenanceRecord.source_system == "ABU_DHABI_NATIONAL_REFERENCE_LAB")
        .first()
    )
    assert prov is not None
    assert prov.target_resource_type == "ClinicalObservation"


def test_cross_phase_workflow_d_external_care_completion(db_session: Session):
    """Workflow D: External Screening Encounter Ingested -> Screening record updated -> Reflected in analytics."""
    org, admin, chw, doctor, patient = _setup_full_environment(db_session)

    eid_val = "784-1995-1122334-1"
    ident = PersonIdentifier(
        person_id=patient.id,
        identifier_namespace=IdentifierNamespace.EMIRATES_ID,
        identifier_value=eid_val,
    )
    db_session.add(ident)
    db_session.commit()

    encounter_bundle = {
        "resourceType": "Bundle",
        "type": "transaction",
        "entry": [
            {
                "resource": {
                    "resourceType": "Patient",
                    "identifier": [{"system": "http://fhir.doh.gov.ae/identifier/emirates-id", "value": eid_val}],
                }
            },
            {
                "resource": {
                    "resourceType": "Encounter",
                    "id": "EXT-ENC-SEHA-001",
                    "status": "finished",
                    "class": {"code": "AMB"},
                }
            },
        ],
    }

    res = InteroperabilityService.ingest_fhir_bundle(
        db=db_session,
        bundle=encounter_bundle,
        source_system="SEHA_AMBULATORY_CLINIC",
        user=admin,
    )

    assert res.status == "SUCCESS"
    assert res.encounters_created == 1

    # Ingested encounter appears in ScreeningRecord table
    sc = (
        db_session.query(ScreeningRecord)
        .filter(ScreeningRecord.patient_id == patient.id)
        .order_by(ScreeningRecord.screening_date.desc())
        .first()
    )
    assert sc is not None
    assert "SEHA_AMBULATORY_CLINIC" in sc.findings_summary


def test_cross_phase_workflow_e_offline_conflict_and_resolution(db_session: Session):
    """Workflow E: Offline Sync vs Newer Server Record -> Conflict Created -> Provider Adjudication."""
    org, admin, chw, doctor, patient = _setup_full_environment(db_session)

    # 1. Clinic records screening at T-10m
    clinic_sc = ScreeningRecord(
        patient_id=patient.id,
        performing_facility_id=org.id,
        screening_type="IFHAS_COMPREHENSIVE",
        screening_date=date.today(),
        screening_status=ScreeningStatus.COMPLETED,
        findings_summary="Clinic baseline examination",
        epistemic_status=EpistemicStatus.VERIFIED,
    )
    db_session.add(clinic_sc)
    db_session.commit()

    # 2. CHW submits offline screening captured at T-60m
    op_id = f"OP-CONFLICT-E-{uuid.uuid4().hex[:6]}"
    batch = OfflineSyncBatchRequest(
        device_id="DEV-FIELD-01",
        operations=[
            OfflineSyncOperationRequest(
                operation_id=op_id,
                idempotency_key=f"IDEMP-{op_id}",
                resource_type="SCREENING",
                operation_type="CREATE",
                client_timestamp=datetime.utcnow() - timedelta(minutes=60),
                payload={
                    "patient_id": patient.id,
                    "blood_pressure_systolic": 120,
                    "blood_pressure_diastolic": 78,
                },
            )
        ],
    )

    sync_res = OfflineSyncService.process_sync_batch(db=db_session, user=chw, batch=batch)
    assert sync_res.conflict_count == 1
    assert sync_res.results[0].status == "CONFLICT"

    # 3. Doctor adjudicates conflict
    conflict = db_session.query(SyncConflict).filter(SyncConflict.operation_id == op_id).first()
    assert conflict is not None

    resolve_req = ResolveConflictRequest(
        resolution="KEEP_SERVER",
        notes="Clinic automated sphygmomanometer confirmed 135/85 mmHg; retaining clinic observation as primary.",
    )
    resolved = OfflineSyncService.resolve_conflict(
        db=db_session, user=doctor, conflict_id=conflict.id, payload=resolve_req
    )

    assert resolved.resolution_status == "RESOLVED_REJECTED"
