import json
import os
import uuid
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from sqlalchemy.orm import Session
from sqlalchemy import or_, and_, desc
from fastapi import HTTPException, status

from backend.app.models.user import User, UserRole
from backend.app.models.person import Person, PatientProfile, PersonIdentifier
from backend.app.models.screening import ScreeningEncounter, EncounterType, EpistemicStatus
from backend.app.models.clinical import Encounter, ClinicalObservation, ScreeningRecord, ScreeningStatus, ObservationType, EncounterStatus
from backend.app.models.recommendation import PreventiveRecommendation, RecommendationLifecycleStatus
from backend.app.models.offline import (
    DeviceRegistration,
    OfflineAssignment,
    SyncOperationAudit,
    SyncConflict,
    DeviceStatus,
    OfflineAssignmentStatus,
    SyncOperationStatus,
    ConflictResolutionStatus,
)
from backend.app.schemas.offline import (
    DeviceRegisterRequest,
    DeviceRegisterResponse,
    OfflinePatientSummary,
    OfflineAssignmentPackageResponse,
    OfflineSyncBatchRequest,
    OfflineSyncBatchResponse,
    SyncOperationResult,
    SyncConflictResponse,
    ResolveConflictRequest,
)

CONFIG_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "..",
    "configs",
    "offline_policy.json",
)


class OfflineSyncService:
    """Enterprise Offline-First Synchronization & Conflict Management Engine for Community Health Workers."""

    @staticmethod
    def _load_policy() -> Dict[str, Any]:
        normalized_path = os.path.abspath(CONFIG_PATH)
        if os.path.exists(normalized_path):
            with open(normalized_path, "r", encoding="utf-8") as f:
                return json.load(f)
        return {
            "offline_assignment": {"max_duration_hours": 72, "max_patients_per_chw": 50},
            "synchronization": {"batch_size_limit": 25, "max_retries": 3},
            "conflict_resolution": {"silent_overwrites_allowed": False},
        }

    @staticmethod
    def register_device(
        db: Session,
        user: User,
        payload: DeviceRegisterRequest,
    ) -> DeviceRegisterResponse:
        """Registers and binds a field device/session to an authenticated Community Health Worker."""
        existing = (
            db.query(DeviceRegistration)
            .filter(
                DeviceRegistration.device_id == payload.device_id,
                DeviceRegistration.user_id == user.id,
            )
            .first()
        )
        if existing:
            existing.status = DeviceStatus.ACTIVE
            existing.last_sync_at = datetime.utcnow()
            existing.app_version = payload.app_version
            db.commit()
            db.refresh(existing)
            return DeviceRegisterResponse(
                registration_id=existing.id,
                device_id=existing.device_id,
                user_id=existing.user_id,
                status=existing.status.value if hasattr(existing.status, "value") else str(existing.status),
                registered_at=existing.registered_at,
            )

        reg = DeviceRegistration(
            device_id=payload.device_id,
            user_id=user.id,
            organization_id=user.organization_id or "org-default",
            device_model=payload.device_model,
            os_platform=payload.os_platform,
            app_version=payload.app_version,
            status=DeviceStatus.ACTIVE,
            registered_at=datetime.utcnow(),
        )
        db.add(reg)
        db.commit()
        db.refresh(reg)

        return DeviceRegisterResponse(
            registration_id=reg.id,
            device_id=reg.device_id,
            user_id=reg.user_id,
            status=reg.status.value if hasattr(reg.status, "value") else str(reg.status),
            registered_at=reg.registered_at,
        )

    @staticmethod
    def get_assignment_package(
        db: Session,
        user: User,
    ) -> OfflineAssignmentPackageResponse:
        """Retrieves authorized patient cohort assignment for field CHW offline visits."""
        policy = OfflineSyncService._load_policy()
        duration_hours = policy.get("offline_assignment", {}).get("max_duration_hours", 72)
        max_patients = policy.get("offline_assignment", {}).get("max_patients_per_chw", 50)
        expires_at = datetime.utcnow() + timedelta(hours=duration_hours)

        # 1. Fetch existing active assignments
        now = datetime.utcnow()
        active_assignments = (
            db.query(OfflineAssignment)
            .filter(
                OfflineAssignment.user_id == user.id,
                OfflineAssignment.status == OfflineAssignmentStatus.ACTIVE,
                OfflineAssignment.expires_at > now,
            )
            .all()
        )

        assigned_patient_ids = [a.patient_id for a in active_assignments]

        # 2. If no active assignments, allocate up to 5 patients from the organization
        if not assigned_patient_ids:
            candidates_query = db.query(Person)
            if user.organization_id:
                candidates_query = candidates_query.join(PatientProfile).filter(PatientProfile.organization_id == user.organization_id)
            candidates = candidates_query.limit(5).all()

            for p in candidates:
                org_id = p.patient_profile.organization_id if p.patient_profile else (user.organization_id or "org-default")
                new_assign = OfflineAssignment(
                    user_id=user.id,
                    patient_id=p.id,
                    organization_id=org_id,
                    assignment_reason="PREVENTIVE_COMMUNITY_OUTREACH",
                    assigned_at=now,
                    expires_at=expires_at,
                    status=OfflineAssignmentStatus.ACTIVE,
                )
                db.add(new_assign)
                assigned_patient_ids.append(p.id)
            db.commit()

        # 3. Assemble patient summaries
        assigned_patients: List[OfflinePatientSummary] = []
        persons = db.query(Person).filter(Person.id.in_(assigned_patient_ids)).all()

        for p in persons:
            ident = db.query(PersonIdentifier).filter(PersonIdentifier.person_id == p.id).first()
            eid = ident.identifier_value if ident else None

            last_screening = (
                db.query(ScreeningRecord)
                .filter(ScreeningRecord.patient_id == p.id)
                .order_by(ScreeningRecord.screening_date.desc())
                .first()
            )
            last_date = last_screening.screening_date.strftime("%Y-%m-%d") if last_screening else None

            open_gaps = (
                db.query(PreventiveRecommendation)
                .filter(
                    PreventiveRecommendation.person_id == p.id,
                    PreventiveRecommendation.lifecycle_status.in_([
                        RecommendationLifecycleStatus.APPROVED,
                        RecommendationLifecycleStatus.SCHEDULED,
                    ]),
                )
                .count()
            )

            dob_str = p.date_of_birth.strftime("%Y-%m-%d") if p.date_of_birth else None

            assigned_patients.append(
                OfflinePatientSummary(
                    patient_id=p.id,
                    full_name=f"{p.first_name} {p.last_name}",
                    date_of_birth=dob_str,
                    gender=p.gender.value if hasattr(p.gender, "value") else str(p.gender),
                    emirates_id=eid,
                    last_screening_date=last_date,
                    assigned_until=expires_at,
                    active_care_gaps_count=open_gaps,
                )
            )

        return OfflineAssignmentPackageResponse(
            chw_user_id=user.id,
            organization_id=user.organization_id or "org-default",
            assigned_patients=assigned_patients,
            package_timestamp=now,
            expires_at=expires_at,
        )

    @staticmethod
    def process_sync_batch(
        db: Session,
        user: User,
        batch: OfflineSyncBatchRequest,
    ) -> OfflineSyncBatchResponse:
        """Processes a batch of offline synchronization operations with idempotency and conflict detection."""
        # Step 1: Revalidate authorization
        if not user.is_active:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="User account is deactivated; offline sync rejected",
            )
        user_role_str = user.role.value if hasattr(user.role, "value") else str(user.role)
        allowed_roles = [
            UserRole.COMMUNITY_HEALTH_WORKER.value,
            UserRole.HEALTHCARE_PROVIDER.value,
            UserRole.SYSTEM_ADMIN.value,
        ]
        if user_role_str not in allowed_roles:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Role not authorized for offline data synchronization",
            )

        policy = OfflineSyncService._load_policy()
        limit = policy.get("synchronization", {}).get("batch_size_limit", 25)
        if len(batch.operations) > limit:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Batch size {len(batch.operations)} exceeds maximum allowed limit of {limit}",
            )

        results: List[SyncOperationResult] = []
        synced_count = 0
        conflict_count = 0
        rejected_count = 0

        for op in batch.operations:
            now = datetime.utcnow()

            # Step 2: Idempotency Check (Check operation_id and idempotency_key)
            existing_audit = (
                db.query(SyncOperationAudit)
                .filter(
                    or_(
                        SyncOperationAudit.operation_id == op.operation_id,
                        SyncOperationAudit.idempotency_key == op.idempotency_key,
                    )
                )
                .first()
            )
            if existing_audit:
                results.append(
                    SyncOperationResult(
                        operation_id=op.operation_id,
                        status=existing_audit.status.value if hasattr(existing_audit.status, "value") else str(existing_audit.status),
                        resource_id=existing_audit.resource_id,
                        server_timestamp=existing_audit.server_timestamp,
                        error_message="Duplicate operation detected: returning previously processed acknowledgment",
                    )
                )
                synced_count += 1
                continue

            # Step 3: Validate Resource & Patient Target
            payload = op.payload
            patient_id = payload.get("patient_id")
            if not patient_id:
                results.append(
                    SyncOperationResult(
                        operation_id=op.operation_id,
                        status="REJECTED",
                        resource_id=None,
                        server_timestamp=now,
                        error_message="Operation rejected: missing required patient_id",
                    )
                )
                rejected_count += 1
                continue

            patient = db.query(Person).filter(Person.id == patient_id).first()
            if not patient:
                results.append(
                    SyncOperationResult(
                        operation_id=op.operation_id,
                        status="REJECTED",
                        resource_id=None,
                        server_timestamp=now,
                        error_message=f"Operation rejected: patient {patient_id} not found in database",
                    )
                )
                rejected_count += 1
                continue

            pat_org_id = patient.patient_profile.organization_id if patient.patient_profile else user.organization_id
            # Check organization boundary
            if user.role != UserRole.SYSTEM_ADMIN and pat_org_id != user.organization_id:
                results.append(
                    SyncOperationResult(
                        operation_id=op.operation_id,
                        status="REJECTED",
                        resource_id=None,
                        server_timestamp=now,
                        error_message="Operation rejected: tenant boundary violation",
                    )
                )
                rejected_count += 1
                continue

            # Step 4: Handle Resource Types & Detect Conflicts
            if op.resource_type == "SCREENING":
                # Check for Conflict: Has a newer screening record been created for this patient?
                newer_screening = (
                    db.query(ScreeningRecord)
                    .filter(
                        ScreeningRecord.patient_id == patient_id,
                        ScreeningRecord.created_at > op.client_timestamp,
                    )
                    .first()
                )

                if newer_screening:
                    conflict = SyncConflict(
                        operation_id=op.operation_id,
                        resource_type="SCREENING",
                        resource_id=newer_screening.id,
                        patient_id=patient_id,
                        user_id=user.id,
                        conflict_reason=(
                            f"Newer screening record {newer_screening.id} recorded at {newer_screening.created_at} "
                            f"conflicts with offline submission from {op.client_timestamp}."
                        ),
                        client_payload=json.dumps(payload),
                        server_payload=json.dumps({
                            "screening_id": newer_screening.id,
                            "screening_type": newer_screening.screening_type,
                            "screening_date": str(newer_screening.screening_date),
                        }),
                        resolution_status=ConflictResolutionStatus.PENDING_REVIEW,
                    )
                    db.add(conflict)

                    audit = SyncOperationAudit(
                        operation_id=op.operation_id,
                        idempotency_key=op.idempotency_key,
                        user_id=user.id,
                        device_id=batch.device_id,
                        resource_type="SCREENING",
                        resource_id=newer_screening.id,
                        operation_type=op.operation_type,
                        status=SyncOperationStatus.CONFLICT,
                        payload_snapshot=json.dumps(payload),
                        client_timestamp=op.client_timestamp,
                        server_timestamp=now,
                    )
                    db.add(audit)
                    db.commit()

                    results.append(
                        SyncOperationResult(
                            operation_id=op.operation_id,
                            status="CONFLICT",
                            resource_id=newer_screening.id,
                            server_timestamp=now,
                            error_message="Clinical conflict: newer verified screening exists. Preserved for clinical review.",
                        )
                    )
                    conflict_count += 1
                    continue

                # No conflict: Apply Screening Persistently
                enc_type_str = str(payload.get("encounter_type", "COMMUNITY_VITALS_CHECK")).upper()
                screening_type = (
                    "WELL_CHILD_VISIT"
                    if "WELL_CHILD" in enc_type_str
                    else "IFHAS_COMPREHENSIVE"
                    if "IFHAS" in enc_type_str
                    else "COMMUNITY_SCREENING"
                )

                screening_record = ScreeningRecord(
                    patient_id=patient_id,
                    performing_facility_id=pat_org_id,
                    screening_type=screening_type,
                    screening_date=op.client_timestamp.date() if isinstance(op.client_timestamp, datetime) else now.date(),
                    screening_status=ScreeningStatus.COMPLETED,
                    findings_summary=payload.get("clinical_notes", "Recorded via CHW Mobile Offline Sync"),
                    epistemic_status=EpistemicStatus.VERIFIED,
                )
                db.add(screening_record)
                db.flush()

                # Audit Successful Sync
                audit = SyncOperationAudit(
                    operation_id=op.operation_id,
                    idempotency_key=op.idempotency_key,
                    user_id=user.id,
                    device_id=batch.device_id,
                    resource_type="SCREENING",
                    resource_id=screening_record.id,
                    operation_type=op.operation_type,
                    status=SyncOperationStatus.SYNCED,
                    payload_snapshot=json.dumps(payload),
                    client_timestamp=op.client_timestamp,
                    server_timestamp=now,
                )
                db.add(audit)
                db.commit()

                results.append(
                    SyncOperationResult(
                        operation_id=op.operation_id,
                        status="SYNCED",
                        resource_id=screening_record.id,
                        server_timestamp=now,
                    )
                )
                synced_count += 1

            else:
                # Other resource types (e.g. CLINICAL_OBSERVATION, FOLLOW_UP)
                audit = SyncOperationAudit(
                    operation_id=op.operation_id,
                    idempotency_key=op.idempotency_key,
                    user_id=user.id,
                    device_id=batch.device_id,
                    resource_type=op.resource_type,
                    resource_id=f"RES-{uuid.uuid4().hex[:8]}",
                    operation_type=op.operation_type,
                    status=SyncOperationStatus.SYNCED,
                    payload_snapshot=json.dumps(payload),
                    client_timestamp=op.client_timestamp,
                    server_timestamp=now,
                )
                db.add(audit)
                db.commit()

                results.append(
                    SyncOperationResult(
                        operation_id=op.operation_id,
                        status="SYNCED",
                        resource_id=audit.resource_id,
                        server_timestamp=now,
                    )
                )
                synced_count += 1

        batch_id = f"SYNC-BATCH-{uuid.uuid4().hex[:8].upper()}"
        return OfflineSyncBatchResponse(
            batch_id=batch_id,
            total_processed=len(batch.operations),
            synced_count=synced_count,
            conflict_count=conflict_count,
            rejected_count=rejected_count,
            results=results,
            server_time=datetime.utcnow(),
        )

    @staticmethod
    def get_conflicts(
        db: Session,
        user: User,
        status_filter: Optional[str] = None,
    ) -> List[SyncConflictResponse]:
        """Lists synchronization conflicts for clinician/supervisor reconciliation."""
        query = db.query(SyncConflict)
        if status_filter:
            query = query.filter(SyncConflict.resolution_status == status_filter)
        conflicts = query.order_by(SyncConflict.created_at.desc()).all()

        results = []
        for c in conflicts:
            client_dict = json.loads(c.client_payload) if c.client_payload else {}
            server_dict = json.loads(c.server_payload) if c.server_payload else {}
            results.append(
                SyncConflictResponse(
                    conflict_id=c.id,
                    operation_id=c.operation_id,
                    resource_type=c.resource_type,
                    resource_id=c.resource_id,
                    patient_id=c.patient_id,
                    user_id=c.user_id,
                    conflict_reason=c.conflict_reason,
                    client_payload=client_dict,
                    server_payload=server_dict,
                    resolution_status=c.resolution_status.value if hasattr(c.resolution_status, "value") else str(c.resolution_status),
                    created_at=c.created_at,
                )
            )
        return results

    @staticmethod
    def resolve_conflict(
        db: Session,
        user: User,
        conflict_id: str,
        payload: ResolveConflictRequest,
    ) -> SyncConflictResponse:
        """Resolves a clinical synchronization conflict through authorized clinician review."""
        user_role_str = user.role.value if hasattr(user.role, "value") else str(user.role)
        if user_role_str not in [UserRole.HEALTHCARE_PROVIDER.value, UserRole.SYSTEM_ADMIN.value]:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Only healthcare providers or system administrators can resolve clinical sync conflicts",
            )

        conflict = db.query(SyncConflict).filter(SyncConflict.id == conflict_id).first()
        if not conflict:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Conflict {conflict_id} not found",
            )

        if payload.resolution == "ACCEPT_CLIENT":
            conflict.resolution_status = ConflictResolutionStatus.RESOLVED_ACCEPTED
        else:
            conflict.resolution_status = ConflictResolutionStatus.RESOLVED_REJECTED

        conflict.resolved_by = user.id
        conflict.resolution_notes = payload.notes
        conflict.resolved_at = datetime.utcnow()
        db.commit()
        db.refresh(conflict)

        return SyncConflictResponse(
            conflict_id=conflict.id,
            operation_id=conflict.operation_id,
            resource_type=conflict.resource_type,
            resource_id=conflict.resource_id,
            patient_id=conflict.patient_id,
            user_id=conflict.user_id,
            conflict_reason=conflict.conflict_reason,
            client_payload=json.loads(conflict.client_payload) if conflict.client_payload else {},
            server_payload=json.loads(conflict.server_payload) if conflict.server_payload else {},
            resolution_status=conflict.resolution_status.value if hasattr(conflict.resolution_status, "value") else str(conflict.resolution_status),
            created_at=conflict.created_at,
        )
