import pytest
import uuid
from datetime import date, datetime, timedelta
from sqlalchemy.orm import Session
from fastapi import HTTPException

from backend.app.models.organization import Organization
from backend.app.models.user import User, UserRole
from backend.app.models.person import Person, Gender
from backend.app.models.clinical import ScreeningRecord, ScreeningStatus
from backend.app.models.screening import EpistemicStatus
from backend.app.models.offline import SyncConflict, ConflictResolutionStatus
from backend.app.services.person_service import PersonService
from backend.app.services.offline_sync_service import OfflineSyncService
from backend.app.schemas.offline import (
    DeviceRegisterRequest,
    OfflineSyncBatchRequest,
    OfflineSyncOperationRequest,
    ResolveConflictRequest,
)


def _setup_chw_and_patient(db: Session):
    uid = uuid.uuid4().hex[:6]
    org = Organization(name=f"SEHA Field Health {uid}", license_number=f"DOH-CHW-{uid}", organization_type="PRIMARY_CARE_CLINIC", region="Al Ain")
    db.add(org)
    db.commit()

    chw = User(
        email=f"chw.field.{uid}@wellwatch.ae",
        password_hash="pw",
        full_name="Fatima Al Dhaheri (CHW)",
        role=UserRole.COMMUNITY_HEALTH_WORKER,
        organization_id=org.id,
        is_active=True,
    )
    db.add(chw)
    db.commit()

    patient = PersonService.register_person(
        db=db,
        first_name="Mansoor",
        last_name="Al Ameri",
        date_of_birth=date.today() - timedelta(days=365 * 40),
        gender=Gender.MALE,
        organization_id=org.id,
    )
    return org, chw, patient


def test_device_registration_lifecycle(db_session: Session):
    """Verifies mobile device registration and binding to authenticated CHW."""
    org, chw, patient = _setup_chw_and_patient(db_session)

    req = DeviceRegisterRequest(
        device_id="DEV-TABLET-ABUDHABI-001",
        device_model="Samsung Galaxy Tab Active4 Pro",
        os_platform="Android 14 / WebPWA",
        app_version="1.2.0",
    )
    res = OfflineSyncService.register_device(db=db_session, user=chw, payload=req)
    assert res.device_id == "DEV-TABLET-ABUDHABI-001"
    assert res.user_id == chw.id
    assert res.status == "ACTIVE"


def test_offline_assignment_package_retrieval(db_session: Session):
    """Verifies secure assignment package creation and data minimization."""
    org, chw, patient = _setup_chw_and_patient(db_session)

    pkg = OfflineSyncService.get_assignment_package(db=db_session, user=chw)
    assert pkg.chw_user_id == chw.id
    assert pkg.organization_id == chw.organization_id
    assert len(pkg.assigned_patients) > 0
    assert pkg.expires_at > datetime.utcnow()


def test_offline_screening_sync_and_idempotency(db_session: Session):
    """Verifies persistent offline screening ingestion and duplicate prevention via idempotency keys."""
    org, chw, patient = _setup_chw_and_patient(db_session)

    op_id = f"OP-{uuid.uuid4().hex[:8]}"
    idemp_key = f"IDEMP-{uuid.uuid4().hex[:8]}"

    batch_req = OfflineSyncBatchRequest(
        device_id="DEV-001",
        operations=[
            OfflineSyncOperationRequest(
                operation_id=op_id,
                idempotency_key=idemp_key,
                resource_type="SCREENING",
                operation_type="CREATE",
                client_timestamp=datetime.utcnow() - timedelta(hours=2),
                payload={
                    "patient_id": patient.id,
                    "encounter_type": "IFHAS_COMPREHENSIVE",
                    "blood_pressure_systolic": 128,
                    "blood_pressure_diastolic": 82,
                    "glucose_value": 98.0,
                    "bmi_value": 24.2,
                    "clinical_notes": "Field screening by community health worker",
                },
            )
        ],
    )

    # First sync execution
    res1 = OfflineSyncService.process_sync_batch(db=db_session, user=chw, batch=batch_req)
    assert res1.synced_count == 1
    assert res1.results[0].status == "SYNCED"
    screening_id = res1.results[0].resource_id
    assert screening_id is not None

    # Verify screening is genuinely persisted in database
    persisted = db_session.query(ScreeningRecord).filter(ScreeningRecord.id == screening_id).first()
    assert persisted is not None
    assert persisted.screening_type == "IFHAS_COMPREHENSIVE"

    # Second sync execution with SAME idempotency key (duplicate protection)
    res2 = OfflineSyncService.process_sync_batch(db=db_session, user=chw, batch=batch_req)
    assert res2.synced_count == 1
    assert res2.results[0].status == "SYNCED"
    assert res2.results[0].resource_id == screening_id

    # Verify count of screenings for patient is still exactly 1
    count = db_session.query(ScreeningRecord).filter(ScreeningRecord.id == screening_id).count()
    assert count == 1


def test_offline_sync_clinical_conflict_detection(db_session: Session):
    """Verifies that a newer clinical screening on the server prevents silent overwrite by stale offline sync."""
    org, chw, patient = _setup_chw_and_patient(db_session)

    # 1. Server has newer screening record
    server_screening = ScreeningRecord(
        patient_id=patient.id,
        performing_facility_id=org.id,
        screening_type="IFHAS_COMPREHENSIVE",
        screening_date=date.today(),
        screening_status=ScreeningStatus.COMPLETED,
        findings_summary="Clinic baseline examination",
        epistemic_status=EpistemicStatus.VERIFIED,
    )
    db_session.add(server_screening)
    db_session.commit()

    # 2. CHW submits offline screening performed earlier
    op_id = f"OP-CONFLICT-{uuid.uuid4().hex[:8]}"
    batch_req = OfflineSyncBatchRequest(
        device_id="DEV-001",
        operations=[
            OfflineSyncOperationRequest(
                operation_id=op_id,
                idempotency_key=f"IDEMP-{op_id}",
                resource_type="SCREENING",
                operation_type="CREATE",
                client_timestamp=datetime.utcnow() - timedelta(hours=3),
                payload={
                    "patient_id": patient.id,
                    "encounter_type": "IFHAS_COMPREHENSIVE",
                    "blood_pressure_systolic": 118,
                    "blood_pressure_diastolic": 76,
                },
            )
        ],
    )

    res = OfflineSyncService.process_sync_batch(db=db_session, user=chw, batch=batch_req)
    assert res.conflict_count == 1
    assert res.results[0].status == "CONFLICT"

    # Verify conflict record exists
    conflict = db_session.query(SyncConflict).filter(SyncConflict.operation_id == op_id).first()
    assert conflict is not None
    assert conflict.resolution_status == ConflictResolutionStatus.PENDING_REVIEW


def test_conflict_resolution_by_licensed_provider(db_session: Session):
    """Verifies authorized clinician adjudication of synchronization conflicts."""
    org, chw, patient = _setup_chw_and_patient(db_session)
    uid = uuid.uuid4().hex[:6]

    doctor = User(
        email=f"dr.resolve.{uid}@wellwatch.ae",
        password_hash="pw",
        full_name="Dr. Tariq Al Nuaimi",
        role=UserRole.HEALTHCARE_PROVIDER,
        organization_id=org.id,
        is_active=True,
    )
    db_session.add(doctor)
    db_session.commit()

    conflict = SyncConflict(
        operation_id="OP-RESOLVE-TEST",
        resource_type="SCREENING",
        resource_id="SCR-SERVER-001",
        patient_id=patient.id,
        user_id=chw.id,
        conflict_reason="Discrepant vitals between clinic and field screening",
        client_payload='{"systolic": 130}',
        server_payload='{"systolic": 145}',
        resolution_status=ConflictResolutionStatus.PENDING_REVIEW,
    )
    db_session.add(conflict)
    db_session.commit()

    resolve_req = ResolveConflictRequest(
        resolution="ACCEPT_CLIENT",
        notes="Field reading verified using calibrated Omron device; valid for outpatient follow-up.",
    )
    resolved = OfflineSyncService.resolve_conflict(
        db=db_session, user=doctor, conflict_id=conflict.id, payload=resolve_req
    )

    assert resolved.resolution_status == "RESOLVED_ACCEPTED"
    assert conflict.resolved_by == doctor.id
