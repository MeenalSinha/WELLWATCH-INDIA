from datetime import datetime
from typing import Dict, List, Optional, Any
from pydantic import BaseModel, ConfigDict, Field


class IntegrationProviderResponse(BaseModel):
    provider_name: str
    display_name: str
    provider_type: str
    status: str
    endpoint_url: Optional[str] = None
    is_enabled: bool
    prerequisites: List[str]
    last_heartbeat_at: Optional[datetime] = None


class HealthCheckResult(BaseModel):
    provider_name: str
    success: bool
    status: str
    message: str
    timestamp: datetime


class FhirIngestRequest(BaseModel):
    bundle: Dict[str, Any] = Field(..., description="Valid HL7 FHIR R4 Bundle")
    source_system: str = Field(..., description="Name of external system e.g. MALAFFI, SEHA_EMR, LAB")


class FhirIngestResponse(BaseModel):
    status: str
    patient_id: Optional[str]
    resources_ingested: int
    observations_created: int
    encounters_created: int
    provenance_id: str
    timestamp: datetime


class FhirExportResponse(BaseModel):
    patient_id: str
    bundle: Dict[str, Any]
    exported_at: datetime


class TerminologyLookupResponse(BaseModel):
    internal_code: str
    display_name: str
    target_system: str
    target_code: str
    target_display: str
    ucum_unit: Optional[str] = None
    status: str


class IntegrationMonitoringResponse(BaseModel):
    total_providers: int
    configured_providers: int
    not_configured_providers: int
    providers: List[IntegrationProviderResponse]
    recent_jobs: List[Dict[str, Any]]
    generated_at: datetime
