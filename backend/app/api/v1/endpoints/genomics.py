"""
Genomic Intelligence Endpoints for WELLWATCH PREVENT.
Provides controlled clinical access to genomic findings, ensuring strict separation of duties,
RBAC enforcement ('genomic:read' / 'genomic:write'), and genetic counselor oversight.
"""
from datetime import datetime
from typing import List, Optional
from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, ConfigDict
from sqlalchemy.orm import Session

from backend.app.core.database import get_db
from backend.app.models.user import User
from backend.app.models.genomics import VariantClassification, GenomicReviewStatus
from backend.app.security.rbac import get_current_user, require_permission
from backend.app.security.permissions import Permission
from backend.app.services.genomic_service import GenomicService

router = APIRouter()


class GenomicFindingCreate(BaseModel):
    gene_symbol: str
    variant_id: str
    variant_classification: VariantClassification
    clinical_interpretation: str
    source_laboratory: str
    genomic_coordinates: Optional[str] = None
    report_reference: Optional[str] = None


class GenomicFindingResponse(BaseModel):
    id: str
    patient_id: str
    gene_symbol: str
    variant_id: str
    genomic_coordinates: Optional[str]
    variant_classification: str
    clinical_interpretation: str
    actionable_guidance: Optional[str]
    source_laboratory: str
    report_reference: Optional[str]
    review_status: str
    reviewed_by_user_id: Optional[str]
    reviewed_at: Optional[datetime]
    reviewer_notes: Optional[str]
    created_at: datetime

    model_config = ConfigDict(from_attributes=True)


class GenomicReviewRequest(BaseModel):
    new_status: GenomicReviewStatus
    notes: str


@router.post("/patients/{patient_id}/genomics", response_model=GenomicFindingResponse, status_code=status.HTTP_201_CREATED)
def record_genomic_finding(
    patient_id: str,
    payload: GenomicFindingCreate,
    db: Session = Depends(get_db),
    current_user: User = Depends(require_permission(Permission.GENOMIC_DATA_WRITE))
):
    """
    Ingests an authorized genomic variant report from an accredited laboratory (e.g. Emirati Genome Program).
    Requires 'genomic:write' permission.
    """
    try:
        finding = GenomicService.record_genomic_finding(
            db=db,
            patient_id=patient_id,
            gene_symbol=payload.gene_symbol,
            variant_id=payload.variant_id,
            variant_classification=payload.variant_classification,
            clinical_interpretation=payload.clinical_interpretation,
            source_laboratory=payload.source_laboratory,
            genomic_coordinates=payload.genomic_coordinates,
            report_reference=payload.report_reference,
            recording_user=current_user
        )
        return finding
    except ValueError as ve:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(ve))
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))


@router.get("/patients/{patient_id}/genomics", response_model=List[GenomicFindingResponse])
def get_patient_genomic_findings(
    patient_id: str,
    db: Session = Depends(get_db),
    current_user: User = Depends(require_permission(Permission.GENOMIC_DATA_READ))
):
    """
    Retrieves authorized genomic findings for a patient.
    Requires 'genomic:read' permission and enforces organization boundaries.
    """
    try:
        findings = GenomicService.get_patient_genomic_findings(
            db=db,
            patient_id=patient_id,
            requesting_user=current_user
        )
        return findings
    except PermissionError as pe:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail=str(pe))
    except ValueError as ve:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(ve))


@router.post("/genomics/{finding_id}/review", response_model=GenomicFindingResponse)
def review_genomic_finding(
    finding_id: str,
    payload: GenomicReviewRequest,
    db: Session = Depends(get_db),
    current_user: User = Depends(require_permission(Permission.GENOMIC_DATA_WRITE))
):
    """
    Enables a licensed clinical geneticist or counselor to review and verify a genomic finding.
    """
    try:
        reviewed = GenomicService.submit_genomic_counselor_review(
            db=db,
            finding_id=finding_id,
            reviewer_user=current_user,
            new_status=payload.new_status,
            notes=payload.notes
        )
        return reviewed
    except PermissionError as pe:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail=str(pe))
    except ValueError as ve:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(ve))
