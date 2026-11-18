from datetime import datetime
from typing import Dict, List, Optional, Any
from pydantic import BaseModel, ConfigDict, Field


class DeviceRegisterRequest(BaseModel):
    device_id: str = Field(..., min_length=4, max_length=128)
    device_model: str = Field("Generic Tablet/Mobile", max_length=100)
    os_platform: str = Field("WebPWA", max_length=50)
    app_version: str = Field("1.0.0", max_length=50)


class DeviceRegisterResponse(BaseModel):
    registration_id: str
    device_id: str
    user_id: str
    status: str
    registered_at: datetime

    model_config = ConfigDict(from_attributes=True)


class OfflinePatientSummary(BaseModel):
    patient_id: str
    full_name: str
    date_of_birth: Optional[str]
    gender: str
    emirates_id: Optional[str]
    last_screening_date: Optional[str]
    assigned_until: datetime
    active_care_gaps_count: int


class OfflineAssignmentPackageResponse(BaseModel):
    chw_user_id: str
    organization_id: str
    assigned_patients: List[OfflinePatientSummary]
    package_timestamp: datetime
    expires_at: datetime


class OfflineSyncOperationRequest(BaseModel):
    operation_id: str = Field(..., description="Unique client-generated operation UUID")
    idempotency_key: str = Field(..., description="Idempotency key for duplicate prevention")
    resource_type: str = Field(..., description="SCREENING, CLINICAL_OBSERVATION, or FOLLOW_UP")
    operation_type: str = Field("CREATE", description="CREATE or UPDATE")
    client_timestamp: datetime
    payload: Dict[str, Any]


class OfflineSyncBatchRequest(BaseModel):
    device_id: str
    operations: List[OfflineSyncOperationRequest]


class SyncOperationResult(BaseModel):
    operation_id: str
    status: str  # SYNCED, CONFLICT, REJECTED, FAILED
    resource_id: Optional[str] = None
    server_timestamp: datetime
    error_message: Optional[str] = None


class OfflineSyncBatchResponse(BaseModel):
    batch_id: str
    total_processed: int
    synced_count: int
    conflict_count: int
    rejected_count: int
    results: List[SyncOperationResult]
    server_time: datetime


class SyncConflictResponse(BaseModel):
    conflict_id: str
    operation_id: str
    resource_type: str
    resource_id: str
    patient_id: str
    user_id: str
    conflict_reason: str
    client_payload: Dict[str, Any]
    server_payload: Dict[str, Any]
    resolution_status: str
    created_at: datetime


class ResolveConflictRequest(BaseModel):
    resolution: str = Field(..., description="ACCEPT_CLIENT or KEEP_SERVER")
    notes: Optional[str] = None
