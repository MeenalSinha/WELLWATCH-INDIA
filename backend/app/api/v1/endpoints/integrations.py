from typing import List, Optional
from fastapi import APIRouter, Depends, Query, HTTPException, status
from sqlalchemy.orm import Session

from backend.app.core.database import get_db
from backend.app.security.rbac import get_current_user
from backend.app.models.user import User, UserRole
from backend.app.schemas.integrations import (
    IntegrationProviderResponse,
    HealthCheckResult,
    FhirIngestRequest,
    FhirIngestResponse,
    FhirExportResponse,
    TerminologyLookupResponse,
    IntegrationMonitoringResponse,
)
from backend.app.services.interoperability_service import InteroperabilityService
from backend.app.integrations.terminology.mapper import TerminologyMapper

router = APIRouter()


@router.get("/providers", response_model=List[IntegrationProviderResponse])
def list_providers(
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """Lists external healthcare adapters and their honest operational configuration status."""
    return InteroperabilityService.get_providers(db=db)


@router.get("/providers/{provider_name}/health", response_model=HealthCheckResult)
def test_provider_health(
    provider_name: str,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """Executes a live connectivity and authentication check against the specified provider."""
    return InteroperabilityService.test_provider_health(
        db=db, provider_name=provider_name, user=current_user
    )


@router.post("/fhir/export/patient/{patient_id}", response_model=FhirExportResponse)
@router.get("/fhir/export/patient/{patient_id}", response_model=FhirExportResponse)
@router.get("/export/fhir/patient/{patient_id}", response_model=FhirExportResponse)
@router.post("/export/fhir/patient/{patient_id}", response_model=FhirExportResponse)
def export_patient_fhir(
    patient_id: str,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """Exports patient records, vitals, and encounters formatted as a validated FHIR R4 Bundle."""
    return InteroperabilityService.export_patient_bundle(
        db=db, patient_id=patient_id, user=current_user
    )


@router.post("/fhir/ingest", response_model=FhirIngestResponse)
@router.post("/ingest/fhir", response_model=FhirIngestResponse)
def ingest_fhir_bundle(
    payload: FhirIngestRequest,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """Ingests external FHIR R4 bundle with Emirates ID matching, LOINC mapping, and provenance."""
    user_role_str = current_user.role.value if hasattr(current_user.role, "value") else str(current_user.role)
    allowed_roles = [
        UserRole.SYSTEM_ADMIN.value,
        UserRole.HEALTHCARE_PROVIDER.value,
        UserRole.DATA_GOVERNANCE_OFFICER.value,
    ]
    if user_role_str not in allowed_roles:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Role not authorized to ingest external clinical data",
        )
    return InteroperabilityService.ingest_fhir_bundle(
        db=db,
        bundle=payload.bundle,
        source_system=payload.source_system,
        user=current_user,
    )


@router.get("/terminology", response_model=List[TerminologyLookupResponse])
def get_terminology_mappings(
    current_user: User = Depends(get_current_user),
):
    """Retrieves governed LOINC, SNOMED CT, ICD-10, and UCUM clinical terminology mappings."""
    mappings = TerminologyMapper.get_all_mappings()
    return [
        TerminologyLookupResponse(
            internal_code=m["internal_code"],
            display_name=m["display_name"],
            target_system=m["target_system"],
            target_code=m["target_code"],
            target_display=m["target_display"],
            ucum_unit=m.get("ucum_unit"),
            status=m.get("status", "APPROVED"),
        )
        for m in mappings
    ]


@router.get("/monitoring", response_model=IntegrationMonitoringResponse)
def get_monitoring_overview(
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """Retrieves central monitoring status across all external adapters and recent exchange jobs."""
    return InteroperabilityService.get_monitoring_overview(db=db)
