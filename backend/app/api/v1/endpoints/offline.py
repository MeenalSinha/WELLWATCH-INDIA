from typing import Optional, List
from fastapi import APIRouter, Depends, Query, HTTPException, status
from sqlalchemy.orm import Session

from backend.app.core.database import get_db
from backend.app.security.rbac import get_current_user
from backend.app.models.user import User, UserRole
from backend.app.schemas.offline import (
    DeviceRegisterRequest,
    DeviceRegisterResponse,
    OfflineAssignmentPackageResponse,
    OfflineSyncBatchRequest,
    OfflineSyncBatchResponse,
    SyncConflictResponse,
    ResolveConflictRequest,
)
from backend.app.services.offline_sync_service import OfflineSyncService

router = APIRouter()


@router.post("/devices/register", response_model=DeviceRegisterResponse)
def register_device(
    payload: DeviceRegisterRequest,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """Registers a mobile or tablet field device for authorized Community Health Worker use."""
    return OfflineSyncService.register_device(db=db, user=current_user, payload=payload)


@router.get("/assignments", response_model=OfflineAssignmentPackageResponse)
@router.get("/assignments/current", response_model=OfflineAssignmentPackageResponse)
def get_offline_assignments(
    duration_hours: int = Query(72),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """Downloads time-bounded authorized patient cohort for offline community screening visits."""
    return OfflineSyncService.get_assignment_package(db=db, user=current_user)


@router.post("/sync", response_model=OfflineSyncBatchResponse)
def process_sync_batch(
    payload: OfflineSyncBatchRequest,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """Uploads offline change queue with idempotency, duplicate suppression, and clinical conflict detection."""
    return OfflineSyncService.process_sync_batch(db=db, user=current_user, batch=payload)


@router.get("/conflicts", response_model=List[SyncConflictResponse])
def get_conflicts(
    status: Optional[str] = Query(None),
    status_filter: Optional[str] = Query(None),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """Lists clinical synchronization conflicts for clinician/supervisor reconciliation."""
    eff_status = status or status_filter
    return OfflineSyncService.get_conflicts(db=db, user=current_user, status_filter=eff_status)


@router.post("/conflicts/{conflict_id}/resolve", response_model=SyncConflictResponse)
def resolve_conflict(
    conflict_id: str,
    payload: ResolveConflictRequest,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """Resolves a clinical synchronization conflict via licensed provider adjudication."""
    return OfflineSyncService.resolve_conflict(
        db=db, user=current_user, conflict_id=conflict_id, payload=payload
    )
