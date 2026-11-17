"""
Unit and Integration Test Suite for Phase 8 Genomic Intelligence Subsystem.
Verifies VUS safety invariants, pathogenic risk evaluation, RBAC permissions,
and genetic counselor review workflows.
"""
from datetime import date
import pytest
from sqlalchemy.orm import Session

from backend.app.models.organization import Organization
from backend.app.models.user import User, UserRole
from backend.app.models.person import Gender, PopulationClassification
from backend.app.models.genomics import (
    GenomicFinding,
    VariantClassification,
    GenomicReviewStatus,
)
from backend.app.models.assessment import ClinicalRiskTier, AssessmentStatus
from backend.app.services.person_service import PersonService
from backend.app.services.genomic_service import GenomicService
from backend.app.services.preventive_assessment_service import PreventiveAssessmentService


def test_vus_safety_rule_does_not_elevate_clinical_risk(db_session: Session):
    """
    CRITICAL SAFETY INVARIANT:
    Variants of Uncertain Significance (VUS) must NEVER trigger automated risk elevation
    or automated clinical orders.
    """
    org = Organization(name="SSMC Genomic Institute", license_number="DOH-GEN-051", organization_type="HOSPITAL", region="Abu Dhabi")
    db_session.add(org)
    db_session.commit()

    person = PersonService.register_person(
        db=db_session,
        first_name="Mariam",
        last_name="Al Mansoori",
        date_of_birth=date(1992, 5, 10),
        gender=Gender.FEMALE,
        organization_id=org.id
    )

    # Ingest a VUS in BRCA1
    GenomicService.record_genomic_finding(
        db=db_session,
        patient_id=person.id,
        gene_symbol="BRCA1",
        variant_id="c.5095C>T (p.Arg1699Gln)",
        variant_classification=VariantClassification.VUS,
        clinical_interpretation="Variant of uncertain clinical significance. Conflicting evidence in literature.",
        source_laboratory="Emirati Genome Program / G42 Healthcare",
        report_reference="EGP-BRCA-9921"
    )

    # Evaluate genomic risk summary directly
    summary = GenomicService.evaluate_genomic_risk_factors(db=db_session, patient_id=person.id)
    assert summary["has_pathogenic_risk"] is False
    assert len(summary["actionable_factors"]) == 0
    assert len(summary["uncertain_findings"]) == 1
    assert "Non-actionable" in summary["uncertain_findings"][0]["note"]

    # Generate assessment: Must abstain from elevated tier and capture VUS limitation
    assessment = PreventiveAssessmentService.generate_assessment(
        db=db_session,
        person_id=person.id,
        pathway_id="ADPHC-ADULT-IFHAS-V1"
    )

    # Risk tier must NOT be elevated; without vitals, it should abstain with INSUFFICIENT_DATA
    assert assessment.clinical_risk_tier != ClinicalRiskTier.ELEVATED_RISK
    assert assessment.clinical_risk_tier == ClinicalRiskTier.INSUFFICIENT_DATA
    assert "VARIANT_OF_UNCERTAIN_SIGNIFICANCE" in assessment.data_quality_limitations


def test_pathogenic_variant_elevates_risk_and_provides_actionable_guidance(db_session: Session):
    """
    Verifies that a verified PATHOGENIC variant (e.g. in LDLR for Familial Hypercholesterolemia)
    elevates clinical risk and attaches targeted preventive surveillance guidance.
    """
    org = Organization(name="CCAD Precision Health", license_number="DOH-GEN-052", organization_type="HOSPITAL", region="Abu Dhabi")
    db_session.add(org)
    db_session.commit()

    person = PersonService.register_person(
        db=db_session,
        first_name="Ahmed",
        last_name="Al Nuaimi",
        date_of_birth=date(1988, 3, 20),
        gender=Gender.MALE,
        organization_id=org.id
    )

    # Record Pathogenic LDLR variant
    GenomicService.record_genomic_finding(
        db=db_session,
        patient_id=person.id,
        gene_symbol="LDLR",
        variant_id="rs121908028",
        variant_classification=VariantClassification.PATHOGENIC,
        clinical_interpretation="Pathogenic missense variant causing Familial Hypercholesterolemia.",
        source_laboratory="Emirati Genome Program / G42 Healthcare",
        report_reference="EGP-LIPID-8842"
    )

    summary = GenomicService.evaluate_genomic_risk_factors(db=db_session, patient_id=person.id)
    assert summary["has_pathogenic_risk"] is True
    assert len(summary["actionable_factors"]) == 1
    factor = summary["actionable_factors"][0]
    assert factor["gene"] == "LDLR"
    assert factor["condition"] == "Familial Hypercholesterolemia (FH)"
    assert "EARLY_CARDIOVASCULAR_LIPID_SCREENING" in factor["preventive_pathway"]

    # Generate assessment: Must incorporate pathogenic finding into elevated risk
    assessment = PreventiveAssessmentService.generate_assessment(
        db=db_session,
        person_id=person.id,
        pathway_id="ADPHC-ADULT-IFHAS-V1"
    )

    assert assessment.clinical_risk_tier == ClinicalRiskTier.ELEVATED_RISK
    assert "Genomic Pathogenic Variant: LDLR" in assessment.contributing_factors


def test_genomic_data_rbac_and_organization_isolation(db_session: Session):
    """
    Verifies strict RBAC permissions (genomic:read required) and organization boundary enforcement.
    """
    org_a = Organization(name="Hospital A", license_number="DOH-061", organization_type="HOSPITAL", region="Abu Dhabi")
    org_b = Organization(name="Hospital B", license_number="DOH-062", organization_type="HOSPITAL", region="Abu Dhabi")
    db_session.add_all([org_a, org_b])
    db_session.commit()

    provider_a = User(email="dr.a@hospital-a.ae", full_name="Dr. Provider A", password_hash="hash", role=UserRole.HEALTHCARE_PROVIDER.value, organization_id=org_a.id)
    provider_b = User(email="dr.b@hospital-b.ae", full_name="Dr. Provider B", password_hash="hash", role=UserRole.HEALTHCARE_PROVIDER.value, organization_id=org_b.id)
    chw_a = User(email="chw.a@hospital-a.ae", full_name="CHW A", password_hash="hash", role=UserRole.COMMUNITY_HEALTH_WORKER.value, organization_id=org_a.id)

    db_session.add_all([provider_a, provider_b, chw_a])
    db_session.commit()

    patient_a = PersonService.register_person(
        db=db_session,
        first_name="Hessa",
        last_name="Al Mazrouei",
        date_of_birth=date(1995, 8, 14),
        gender=Gender.FEMALE,
        organization_id=org_a.id
    )

    GenomicService.record_genomic_finding(
        db=db_session,
        patient_id=patient_a.id,
        gene_symbol="BRCA2",
        variant_id="rs80359550",
        variant_classification=VariantClassification.PATHOGENIC,
        clinical_interpretation="Pathogenic frameshift variant.",
        source_laboratory="Emirati Genome Program",
        recording_user=provider_a
    )

    # 1. Provider from Org A can access
    findings = GenomicService.get_patient_genomic_findings(
        db=db_session,
        patient_id=patient_a.id,
        requesting_user=provider_a
    )
    assert len(findings) == 1
    assert findings[0].gene_symbol == "BRCA2"

    # 2. CHW from Org A lacks genomic:read permission -> Blocked
    with pytest.raises(PermissionError) as exc_chw:
        GenomicService.get_patient_genomic_findings(
            db=db_session,
            patient_id=patient_a.id,
            requesting_user=chw_a
        )
    assert "lacks required 'genomic:read' permission" in str(exc_chw.value)

    # 3. Provider from Org B is in a different organization -> Blocked
    with pytest.raises(PermissionError) as exc_org:
        GenomicService.get_patient_genomic_findings(
            db=db_session,
            patient_id=patient_a.id,
            requesting_user=provider_b
        )
    assert "Cross-organization patient data access prohibited" in str(exc_org.value)


def test_genomic_counselor_review_workflow(db_session: Session):
    """
    Verifies that a clinical geneticist / provider can verify or update a genomic finding.
    """
    org = Organization(name="Genomics Center", license_number="DOH-GEN-071", organization_type="HOSPITAL", region="Abu Dhabi")
    db_session.add(org)
    db_session.commit()
    counselor = User(email="counselor@genomics.ae", full_name="Dr. Genetic Counselor", password_hash="hash", role=UserRole.HEALTHCARE_PROVIDER.value, organization_id=org.id)
    db_session.add(counselor)
    db_session.commit()

    person = PersonService.register_person(
        db=db_session,
        first_name="Sultan",
        last_name="Al Qasimi",
        date_of_birth=date(1990, 1, 1),
        gender=Gender.MALE,
        organization_id=org.id
    )

    finding = GenomicService.record_genomic_finding(
        db=db_session,
        patient_id=person.id,
        gene_symbol="MLH1",
        variant_id="rs63750447",
        variant_classification=VariantClassification.LIKELY_PATHOGENIC,
        clinical_interpretation="Likely pathogenic variant associated with Lynch syndrome.",
        source_laboratory="Emirati Genome Program"
    )

    assert finding.review_status == GenomicReviewStatus.PENDING_REVIEW

    # Counselor review
    reviewed = GenomicService.submit_genomic_counselor_review(
        db=db_session,
        finding_id=finding.id,
        reviewer_user=counselor,
        new_status=GenomicReviewStatus.GENETIC_COUNSELOR_VERIFIED,
        notes="Confirmed pathogenicity through segregation analysis. Referred to gastroenterology for early colonoscopy."
    )

    assert reviewed.review_status == GenomicReviewStatus.GENETIC_COUNSELOR_VERIFIED
    assert reviewed.reviewed_by_user_id == counselor.id
    assert reviewed.reviewed_at is not None
    assert "Confirmed pathogenicity" in reviewed.reviewer_notes
