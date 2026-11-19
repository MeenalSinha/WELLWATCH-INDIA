import time
import uuid
import pytest
from datetime import date, datetime, timedelta
from sqlalchemy.orm import Session
from fastapi import HTTPException

from backend.app.models.organization import Organization
from backend.app.models.user import User, UserRole
from backend.app.models.person import Person, Gender
from backend.app.services.person_service import PersonService
from backend.app.services.offline_sync_service import OfflineSyncService
from backend.app.services.population_analytics_service import PopulationAnalyticsService
from backend.app.schemas.offline import OfflineSyncBatchRequest, OfflineSyncOperationRequest


def test_batch_sync_size_limit_enforcement(db_session: Session):
    """Reliability Control: Rejects batches exceeding maximum operational limit of 25 operations."""
    uid = uuid.uuid4().hex[:6]
    org = Organization(name=f"Perf Clinic {uid}", license_number=f"DOH-PERF-{uid}", organization_type="CLINIC", region="Abu Dhabi")
    db_session.add(org)
    db_session.commit()

    chw = User(
        email=f"chw.perf.{uid}@wellwatch.ae",
        password_hash="pw",
        full_name="Fatima Perf",
        role=UserRole.COMMUNITY_HEALTH_WORKER,
        organization_id=org.id,
        is_active=True,
    )
    db_session.add(chw)
    db_session.commit()

    # Create 30 operations (exceeds limit of 25)
    operations = [
        OfflineSyncOperationRequest(
            operation_id=f"OP-OVERSIZE-{i}",
            idempotency_key=f"IDEMP-OVERSIZE-{i}",
            resource_type="SCREENING",
            operation_type="CREATE",
            client_timestamp=datetime.utcnow(),
            payload={"patient_id": f"pat-{i}"},
        )
        for i in range(30)
    ]

    oversize_batch = OfflineSyncBatchRequest(
        device_id="DEV-TABLET-OVERSIZE",
        operations=operations,
    )

    with pytest.raises(HTTPException) as exc_info:
        OfflineSyncService.process_sync_batch(db=db_session, user=chw, batch=oversize_batch)
    assert exc_info.value.status_code == 400
    assert "exceeds maximum allowed limit" in exc_info.value.detail


def test_population_analytics_query_execution_latency(db_session: Session):
    """Performance Benchmark: Population overview queries must execute within 250ms under normal load."""
    uid = uuid.uuid4().hex[:6]
    org = Organization(name=f"Benchmark Health {uid}", license_number=f"DOH-BENCH-{uid}", organization_type="HOSPITAL", region="Abu Dhabi")
    db_session.add(org)
    db_session.commit()

    # Pre-seed 10 patient records
    for i in range(10):
        PersonService.register_person(
            db=db_session,
            first_name=f"Patient{i}",
            last_name="Al Zaabi",
            date_of_birth=date.today() - timedelta(days=365 * (20 + i)),
            gender=Gender.MALE if i % 2 == 0 else Gender.FEMALE,
            organization_id=org.id,
        )

    start_time = time.time()
    overview = PopulationAnalyticsService.get_overview(
        db=db_session, organization_id=org.id, time_window_days=90
    )
    elapsed_ms = (time.time() - start_time) * 1000.0

    assert overview.total_registered >= 10
    # Query execution latency must be sub-500ms
    assert elapsed_ms < 500.0, f"Query took {elapsed_ms}ms, exceeding 500ms benchmark"
