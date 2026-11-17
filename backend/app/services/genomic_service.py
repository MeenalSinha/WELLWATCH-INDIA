"""
Genomic Intelligence Service for WELLWATCH PREVENT.
Implements clinical interpretation of genomic variants aligned with the Emirati Genome Program (EGP)
and enforces strict safety invariants: VUS (Variants of Uncertain Significance) must never trigger
automated clinical risk escalation or intervention orders.
"""
from datetime import datetime, timezone
from typing import List, Dict, Any, Optional
from sqlalchemy.orm import Session
from sqlalchemy import desc

from backend.app.models.genomics import GenomicFinding, VariantClassification, GenomicReviewStatus
from backend.app.models.person import Person
from backend.app.models.provenance import (
    DataSource,
    DataProvenance,
    SourceType,
    TrustLevel,
    ConnectionStatus,
    ProvenanceVerificationStatus,
)
from backend.app.models.user import User
from backend.app.security.permissions import Permission
from backend.app.security.rbac import has_permission


class GenomicService:
    """
    Governs ingestion, interpretation, safety invariant enforcement, and review of genomic findings.
    """

    # Gene to High-Confidence Preventive Indication Mapping (Authoritative EGP / ACMG Secondary Findings)
    ACTIONABLE_GENE_INDICATIONS = {
        "LDLR": {
            "condition": "Familial Hypercholesterolemia (FH)",
            "preventive_pathway": "EARLY_CARDIOVASCULAR_LIPID_SCREENING",
            "clinical_guidance": "Initiate early intensive lipid panel screening (annual fasting lipid profile, ApoB, Lp(a)) and lifestyle cardiometabolic monitoring."
        },
        "APOB": {
            "condition": "Familial Hypercholesterolemia (FH)",
            "preventive_pathway": "EARLY_CARDIOVASCULAR_LIPID_SCREENING",
            "clinical_guidance": "Initiate annual lipid profile evaluation; evaluate candidate for early lipid-lowering therapy."
        },
        "BRCA1": {
            "condition": "Hereditary Breast and Ovarian Cancer (HBOC)",
            "preventive_pathway": "HIGH_RISK_BREAST_SCREENING",
            "clinical_guidance": "Accelerate breast cancer surveillance: annual contrast-enhanced breast MRI beginning at age 25-30, combined with mammography."
        },
        "BRCA2": {
            "condition": "Hereditary Breast and Ovarian Cancer (HBOC)",
            "preventive_pathway": "HIGH_RISK_BREAST_SCREENING",
            "clinical_guidance": "Initiate enhanced surveillance: annual breast MRI and mammography starting age 25-30."
        },
        "MLH1": {
            "condition": "Lynch Syndrome (Hereditary Colorectal Cancer)",
            "preventive_pathway": "EARLY_COLONOSCOPY_SURVEILLANCE",
            "clinical_guidance": "Initiate high-resolution colonoscopy surveillance every 1-2 years beginning at age 20-25."
        },
        "MSH2": {
            "condition": "Lynch Syndrome (Hereditary Colorectal Cancer)",
            "preventive_pathway": "EARLY_COLONOSCOPY_SURVEILLANCE",
            "clinical_guidance": "Initiate high-resolution colonoscopy surveillance every 1-2 years beginning at age 20-25."
        }
    }

    @classmethod
    def record_genomic_finding(
        cls,
        db: Session,
        patient_id: str,
        gene_symbol: str,
        variant_id: str,
        variant_classification: VariantClassification,
        clinical_interpretation: str,
        source_laboratory: str,
        genomic_coordinates: Optional[str] = None,
        report_reference: Optional[str] = None,
        recording_user: Optional[User] = None
    ) -> GenomicFinding:
        """
        Records a validated genomic variant report.
        Automatically attaches provenance and sets review status based on pathogenicity.
        """
        person = db.query(Person).filter(Person.id == patient_id).first()
        if not person:
            raise ValueError(f"Patient ID '{patient_id}' not found.")

        # Determine actionable guidance if gene has recognized preventive indication
        actionable_guidance = None
        gene_upper = gene_symbol.upper()
        if variant_classification in [VariantClassification.PATHOGENIC, VariantClassification.LIKELY_PATHOGENIC]:
            if gene_upper in cls.ACTIONABLE_GENE_INDICATIONS:
                actionable_guidance = cls.ACTIONABLE_GENE_INDICATIONS[gene_upper]["clinical_guidance"]

        # Register data source and provenance
        data_source = db.query(DataSource).filter(DataSource.system_name == source_laboratory).first()
        if not data_source:
            data_source = DataSource(
                source_type=SourceType.GENOMIC_CORE,
                system_name=source_laboratory,
                system_version="1.0",
                trust_level=TrustLevel.ACCREDITED_CLINICAL,
                connection_status=ConnectionStatus.OPERATIONAL,
            )
            db.add(data_source)
            db.flush()

        provenance = DataProvenance(
            data_source_id=data_source.id,
            collection_method="GENOMIC_SEQUENCING_REPORT",
            external_record_id=report_reference or f"EGP-{variant_id}",
            verification_status=ProvenanceVerificationStatus.VERIFIED,
            responsible_processor_id=recording_user.id if recording_user else None,
        )
        db.add(provenance)
        db.flush()

        finding = GenomicFinding(
            patient_id=patient_id,
            gene_symbol=gene_upper,
            variant_id=variant_id,
            genomic_coordinates=genomic_coordinates,
            variant_classification=variant_classification,
            clinical_interpretation=clinical_interpretation,
            actionable_guidance=actionable_guidance,
            source_laboratory=source_laboratory,
            report_reference=report_reference,
            provenance_id=provenance.id,
            review_status=GenomicReviewStatus.PENDING_REVIEW
        )
        db.add(finding)
        db.commit()
        db.refresh(finding)
        return finding

    @classmethod
    def get_patient_genomic_findings(
        cls,
        db: Session,
        patient_id: str,
        requesting_user: User
    ) -> List[GenomicFinding]:
        """
        Lists genomic findings for a patient with strict RBAC permission enforcement.
        """
        # RBAC Check: User must possess GENOMIC_DATA_READ permission
        if not has_permission(requesting_user.role, Permission.GENOMIC_DATA_READ):
            raise PermissionError("Access denied: User lacks required 'genomic:read' permission.")

        person = db.query(Person).filter(Person.id == patient_id).first()
        if not person:
            raise ValueError(f"Patient ID '{patient_id}' not found.")

        # Organization boundary check
        patient_org_id = person.patient_profile.organization_id if person.patient_profile else None
        if requesting_user.organization_id and patient_org_id:
            if requesting_user.organization_id != patient_org_id:
                raise PermissionError("Access denied: Cross-organization patient data access prohibited.")

        return db.query(GenomicFinding).filter(
            GenomicFinding.patient_id == patient_id
        ).order_by(desc(GenomicFinding.created_at)).all()

    @classmethod
    def evaluate_genomic_risk_factors(
        cls,
        db: Session,
        patient_id: str
    ) -> Dict[str, Any]:
        """
        Synthesizes genomic findings for preventive risk assessment.

        SAFETY INVARIANT:
        Variants of Uncertain Significance (VUS) MUST NEVER elevate clinical risk tiers
        or trigger automated interventions.
        """
        findings = db.query(GenomicFinding).filter(
            GenomicFinding.patient_id == patient_id
        ).all()

        actionable_factors = []
        uncertain_findings = []
        benign_count = 0
        has_pathogenic_risk = False

        for f in findings:
            if f.variant_classification in [VariantClassification.PATHOGENIC, VariantClassification.LIKELY_PATHOGENIC]:
                # Actionable pathogenic variant
                has_pathogenic_risk = True
                indication = cls.ACTIONABLE_GENE_INDICATIONS.get(f.gene_symbol, {})
                actionable_factors.append({
                    "indicator": f"GENOMIC_PATHOGENIC_{f.gene_symbol}",
                    "gene": f.gene_symbol,
                    "variant_id": f.variant_id,
                    "classification": f.variant_classification.value,
                    "condition": indication.get("condition", "Hereditary Disease Predisposition"),
                    "preventive_pathway": indication.get("preventive_pathway", "GENETIC_HIGH_RISK_SURVEILLANCE"),
                    "actionable_guidance": f.actionable_guidance or f.clinical_interpretation,
                    "clinical_source": f"Emirati Genome Program / {f.source_laboratory}",
                    "severity": "ELEVATED"
                })
            elif f.variant_classification == VariantClassification.VUS:
                # VUS: MUST NOT elevate risk. Kept strictly as informational / counseling required.
                uncertain_findings.append({
                    "gene": f.gene_symbol,
                    "variant_id": f.variant_id,
                    "classification": "VUS",
                    "note": "Variant of Uncertain Significance. Non-actionable. Does NOT modify clinical risk tier. Specialized genetic counseling required if family history positive."
                })
            else:
                benign_count += 1

        return {
            "has_pathogenic_risk": has_pathogenic_risk,
            "actionable_factors": actionable_factors,
            "uncertain_findings": uncertain_findings,
            "benign_findings_count": benign_count
        }

    @classmethod
    def submit_genomic_counselor_review(
        cls,
        db: Session,
        finding_id: str,
        reviewer_user: User,
        new_status: GenomicReviewStatus,
        notes: str
    ) -> GenomicFinding:
        """
        Allows a qualified genetic counselor or physician to verify or modify a genomic finding.
        """
        if not has_permission(reviewer_user.role, Permission.GENOMIC_DATA_WRITE):
            raise PermissionError("User lacks 'genomic:write' permission for genetic review.")

        finding = db.query(GenomicFinding).filter(GenomicFinding.id == finding_id).first()
        if not finding:
            raise ValueError(f"Genomic finding ID '{finding_id}' not found.")

        finding.review_status = new_status
        finding.reviewed_by_user_id = reviewer_user.id
        finding.reviewed_at = datetime.now(timezone.utc)
        finding.reviewer_notes = notes

        db.commit()
        db.refresh(finding)
        return finding
