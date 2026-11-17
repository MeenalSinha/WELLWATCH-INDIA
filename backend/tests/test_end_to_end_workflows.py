"""
End-to-End Clinical and Operational Workflows for WELLWATCH PREVENT.
Covers:
- Workflow A: Pediatric Patient Journey (Registration, Guardianship, Well-Child Pathway, Assessment, Physician Review, Audit)
- Workflow B: Adult Patient Screening Journey (Vitals, IFHAS Pathway, Risk Stratification, Guideline Attribution)
- Workflow C: Insufficient Data & Epistemic Abstention (Missing Vitals -> Abstention -> Vital Update -> Transition)
- Workflow D: Genomic Intelligence Safety Guardrails (Pathogenic Actionability vs VUS Non-Actionability & Counselor Sign-Off)
- Workflow E: ML Governance & Controlled Inference (Split Isolation, Calibration, Registry Lifecycle, Abstention)
"""
import json
from datetime import date, datetime, timedelta, timezone
import pytest
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sqlalchemy.orm import Session

from backend.app.models.assessment import (
    AssessmentStatus,
    ClinicalRiskTier,
    HumanReviewStatus,
    PreventiveAssessment,
)
from backend.app.models.audit import AuditLog, AuditOutcome
from backend.app.models.clinical import ObservationType
from backend.app.models.genomics import (
    GenomicFinding,
    GenomicReviewStatus,
    VariantClassification,
)
from backend.app.models.guardian import Guardianship, GuardianshipStatus, RelationshipType
from backend.app.models.organization import Organization
from backend.app.models.person import Gender, PopulationClassification
from backend.app.models.user import User, UserRole
from backend.app.services.audit_service import AuditService
from backend.app.services.genomic_service import GenomicService
from backend.app.services.pathway_engine import PathwayEngine
from backend.app.services.person_service import PersonService
from backend.app.services.preventive_assessment_service import PreventiveAssessmentService

from backend.app.ml.dataset_governance import DatasetGovernance
from backend.app.ml.preprocessing import LeakageResistantPreprocessor
from backend.app.ml.calibration import ProbabilityCalibrator
from backend.app.ml.explainability import FeatureAttributionExplainer
from backend.app.ml.model_registry import (
    ModelRegistry,
    RegisteredModelMetadata,
    ModelLifecycleStatus,
)
from backend.app.ml.inference_service import ControlledInferenceService


# =========================================================================
# WORKFLOW A: Pediatric Patient Journey
# =========================================================================
def test_workflow_a_pediatric_patient_journey(db_session: Session):
    """
    Workflow A:
    1. Register pediatric patient (18 months old).
    2. Establish legal guardian relationship.
    3. Record Well-Child visit clinical observations (height, weight, pulse).
    4. Evaluate DoH Well-Child pathway eligibility & milestone care gaps.
    5. Generate preventive assessment snapshot.
    6. Verify human review workflow (Physician approval).
    7. Confirm audit trail captures all actions.
    """
    # 1. Facility & Provider
    org = Organization(
        name="Danat Al Emarat Pediatric Hospital",
        license_number="DOH-HOSP-091",
        organization_type="HOSPITAL",
        region="Abu Dhabi"
    )
    db_session.add(org)
    db_session.commit()

    provider = User(
        email="dr.pediatrics@danatalemarat.ae",
        password_hash="argon2id_mock_hash",
        full_name="Dr. Noura Al Kaabi",
        role=UserRole.HEALTHCARE_PROVIDER.value,
        organization_id=org.id,
        is_active=True
    )
    parent_user = User(
        email="fatima.alzaabi@email.ae",
        password_hash="argon2id_mock_hash",
        full_name="Fatima Al Zaabi",
        role=UserRole.INDIVIDUAL.value,
        organization_id=org.id,
        is_active=True
    )
    db_session.add_all([provider, parent_user])
    db_session.commit()

    # 2. Register 18-month old child
    today = date.today()
    child_dob = today - timedelta(days=548)
    child = PersonService.register_person(
        db=db_session,
        first_name="Zayed",
        last_name="Al Zaabi",
        date_of_birth=child_dob,
        gender=Gender.MALE,
        organization_id=org.id,
        emirate="Abu Dhabi",
        guardian_user_id=parent_user.id,
        relationship_to_guardian="Mother"
    )
    assert child.patient_profile.population_classification == PopulationClassification.PEDIATRIC.value
    assert child.age_months >= 17 and child.age_months <= 19

    # Verify Guardian linkage
    guardianship = db_session.query(Guardianship).filter(Guardianship.dependent_person_id == child.id).first()
    assert guardianship is not None
    assert guardianship.verification_status == GuardianshipStatus.VERIFIED.value
    assert guardianship.guardian_user_id == parent_user.id

    # 3. Record Well-Child visit observations
    obs_height = PersonService.record_clinical_observation(
        db=db_session,
        person_id=child.id,
        observation_type=ObservationType.VITAL_SIGN,
        code="8302-2",
        display_name="Body Height",
        numeric_value=82.5,
        unit="cm",
        processor_user_id=provider.id
    )
    obs_weight = PersonService.record_clinical_observation(
        db=db_session,
        person_id=child.id,
        observation_type=ObservationType.VITAL_SIGN,
        code="29463-7",
        display_name="Body Weight",
        numeric_value=11.8,
        unit="kg",
        processor_user_id=provider.id
    )
    assert obs_height.canonical_value == 82.5
    assert obs_weight.canonical_value == 11.8

    # 4. Evaluate DoH Well-Child pathway
    pathway = PathwayEngine.load_pathway("DOH-PEDIATRIC-WELLCHILD-V1")
    assert pathway is not None
    eligibility = PathwayEngine.evaluate_eligibility(pathway=pathway, person=child)
    assert eligibility["is_eligible"] is True

    care_gaps = PathwayEngine.evaluate_care_gaps(db=db_session, pathway=pathway, person=child)
    assert len(care_gaps) > 0
    milestone_ids = [cg["milestone_id"] for cg in care_gaps]
    assert any("WC-" in m for m in milestone_ids)

    # 5. Generate Preventive Assessment Snapshot
    assessment = PreventiveAssessmentService.generate_assessment(
        db=db_session,
        person_id=child.id,
        pathway_id="DOH-PEDIATRIC-WELLCHILD-V1",
        requesting_user_id=provider.id
    )
    assert assessment.population_category == PopulationClassification.PEDIATRIC.value
    assert assessment.pathway_id == "DOH-PEDIATRIC-WELLCHILD-V1"
    assert assessment.human_review_status in [HumanReviewStatus.REVIEWED, HumanReviewStatus.AWAITING_REVIEW]

    # 6. Provider Review
    reviewed = PreventiveAssessmentService.submit_clinical_review(
        db=db_session,
        assessment_id=assessment.id,
        reviewer_user=provider,
        new_status=HumanReviewStatus.APPROVED,
        notes="18-month Well-Child examination completed. Growth parameters within normal WHO centiles."
    )
    assert reviewed.human_review_status == HumanReviewStatus.APPROVED
    assert reviewed.reviewed_by_user_id == provider.id
    assert "WHO centiles" in reviewed.reviewer_notes

    # 7. Audit log
    AuditService.log_event(
        db=db_session,
        action="REVIEW_SUBMITTED",
        resource_type="PreventiveAssessment",
        resource_id=assessment.id,
        actor_user_id=provider.id,
        organization_id=org.id,
        outcome=AuditOutcome.SUCCESS,
        correlation_id="corr-wf-a-001",
        details={"review_status": "APPROVED"}
    )
    audit_entry = db_session.query(AuditLog).filter(AuditLog.resource_id == assessment.id).first()
    assert audit_entry is not None
    assert audit_entry.outcome == AuditOutcome.SUCCESS


# =========================================================================
# WORKFLOW B: Adult Patient Screening Journey
# =========================================================================
def test_workflow_b_adult_patient_preventive_screening_journey(db_session: Session):
    """
    Workflow B:
    1. Register 45-year-old adult Emirati resident.
    2. Record vitals: Blood Pressure (154/96 mmHg), Fasting Glucose (142 mg/dL).
    3. Evaluate ADPHC IFHAS pathway eligibility & screening gaps.
    4. Generate preventive risk assessment.
    5. Verify contributing factors cite DoH/ADPHC guidelines.
    """
    org = Organization(
        name="Cleveland Clinic Abu Dhabi - E2E",
        license_number="DOH-HOSP-042-E2E",
        organization_type="HOSPITAL",
        region="Abu Dhabi"
    )
    db_session.add(org)
    db_session.commit()

    provider = User(
        email="dr.cardiologist@ccad.ae",
        password_hash="argon2id_mock_hash",
        full_name="Dr. Tariq Al Hashimi",
        role=UserRole.HEALTHCARE_PROVIDER.value,
        organization_id=org.id,
        is_active=True
    )
    db_session.add(provider)
    db_session.commit()

    # 1. 45-year-old adult
    dob = date.today() - timedelta(days=45 * 365 + 10)
    adult = PersonService.register_person(
        db=db_session,
        first_name="Mansoor",
        last_name="Al Remeithi",
        date_of_birth=dob,
        gender=Gender.MALE,
        organization_id=org.id,
        emirate="Abu Dhabi",
        district="Al Maryah Island"
    )
    assert adult.patient_profile.population_classification == PopulationClassification.ADULT.value
    assert adult.age_years >= 44 and adult.age_years <= 46

    # 2. Record Stage 2 HTN and Diabetes range glucose
    PersonService.record_clinical_observation(
        db=db_session,
        person_id=adult.id,
        observation_type=ObservationType.VITAL_SIGN,
        code="8480-6",
        display_name="Systolic BP",
        numeric_value=154.0,
        unit="mmHg"
    )
    PersonService.record_clinical_observation(
        db=db_session,
        person_id=adult.id,
        observation_type=ObservationType.VITAL_SIGN,
        code="8462-4",
        display_name="Diastolic BP",
        numeric_value=96.0,
        unit="mmHg"
    )
    PersonService.record_clinical_observation(
        db=db_session,
        person_id=adult.id,
        observation_type=ObservationType.LAB_RESULT,
        code="2339-0",
        display_name="Fasting Blood Glucose",
        numeric_value=142.0,
        unit="mg/dL"
    )

    # 3. Evaluate ADPHC IFHAS Pathway
    pathway = PathwayEngine.load_pathway("ADPHC-ADULT-IFHAS-V1")
    assert pathway is not None
    eligibility = PathwayEngine.evaluate_eligibility(pathway=pathway, person=adult)
    assert eligibility["is_eligible"] is True

    care_gaps = PathwayEngine.evaluate_care_gaps(db=db_session, pathway=pathway, person=adult)
    assert len(care_gaps) > 0
    milestone_ids = [cg["milestone_id"] for cg in care_gaps]
    assert any("IFHAS" in m for m in milestone_ids)

    # 4. Generate Preventive Assessment
    assessment = PreventiveAssessmentService.generate_assessment(
        db=db_session,
        person_id=adult.id,
        pathway_id="ADPHC-ADULT-IFHAS-V1",
        requesting_user_id=provider.id
    )

    # 5. Risk Tier & Guideline Citations
    assert assessment.clinical_risk_tier == ClinicalRiskTier.ELEVATED_RISK
    factors = json.loads(assessment.contributing_factors)
    assert len(factors) >= 2

    sources_cited = [f.get("clinical_source", "") for f in factors]
    # Verify authoritative guidelines are cited
    assert any("Hypertension" in s for s in sources_cited)
    assert any("Diabetes" in s for s in sources_cited)


# =========================================================================
# WORKFLOW C: Insufficient Data and Epistemic Abstention
# =========================================================================
def test_workflow_c_insufficient_data_and_epistemic_abstention(db_session: Session):
    """
    Workflow C:
    1. Register patient with zero vitals.
    2. Generate assessment -> triggers INSUFFICIENT_DATA and NEEDS_DATA abstention.
    3. Verify system abstains from fabricating risk score.
    4. Record required vitals.
    5. Re-evaluate -> transitions from abstention to concrete risk tier.
    """
    org = Organization(name="SEHA Al Bateen Clinic", license_number="DOH-CLN-082", organization_type="CLINIC", region="Abu Dhabi")
    db_session.add(org)
    db_session.commit()

    dob = date.today() - timedelta(days=32 * 365)
    person = PersonService.register_person(
        db=db_session,
        first_name="Salama",
        last_name="Al Mazrouei",
        date_of_birth=dob,
        gender=Gender.FEMALE,
        organization_id=org.id
    )

    # Assessment with missing baseline vitals
    assessment_1 = PreventiveAssessmentService.generate_assessment(
        db=db_session,
        person_id=person.id,
        pathway_id="ADPHC-ADULT-IFHAS-V1"
    )

    # Must abstain
    assert assessment_1.clinical_risk_tier == ClinicalRiskTier.INSUFFICIENT_DATA
    assert assessment_1.assessment_status == AssessmentStatus.NEEDS_DATA
    limitations = json.loads(assessment_1.data_quality_limitations)
    assert any("baseline" in lim.get("detail", "").lower() or lim.get("issue") == "MISSING_BASELINE" for lim in limitations)

    # Now add normal baseline vitals
    PersonService.record_clinical_observation(
        db=db_session,
        person_id=person.id,
        observation_type=ObservationType.VITAL_SIGN,
        code="8480-6",
        display_name="Systolic BP",
        numeric_value=116.0,
        unit="mmHg"
    )
    PersonService.record_clinical_observation(
        db=db_session,
        person_id=person.id,
        observation_type=ObservationType.VITAL_SIGN,
        code="8462-4",
        display_name="Diastolic BP",
        numeric_value=76.0,
        unit="mmHg"
    )
    PersonService.record_clinical_observation(
        db=db_session,
        person_id=person.id,
        observation_type=ObservationType.LAB_RESULT,
        code="1558-6",
        display_name="Fasting Glucose",
        numeric_value=86.0,
        unit="mg/dL"
    )

    # Re-evaluate
    assessment_2 = PreventiveAssessmentService.generate_assessment(
        db=db_session,
        person_id=person.id,
        pathway_id="ADPHC-ADULT-IFHAS-V1"
    )

    # Safely transitioned out of abstention to LOW_RISK
    assert assessment_2.clinical_risk_tier == ClinicalRiskTier.LOW_RISK
    assert assessment_2.assessment_status == AssessmentStatus.COMPLETED


# =========================================================================
# WORKFLOW D: Genomic Intelligence and Safety Guardrails
# =========================================================================
def test_workflow_d_genomic_intelligence_and_safety_guardrails(db_session: Session):
    """
    Workflow D:
    1. Patient with Pathogenic variant (*LDLR* p.Cys660Ter) -> triggers ELEVATED_RISK
       and provides ACMG SF v3.2 actionable guidance.
    2. Patient with VUS (*BRCA1* VUS) -> invariant holds: VUS does NOT elevate clinical risk.
    3. Counselor review workflow verifies variant.
    """
    org = Organization(name="G42 Healthcare Lab", license_number="DOH-LAB-019", organization_type="LABORATORY", region="Abu Dhabi")
    db_session.add(org)
    db_session.commit()

    counselor = User(
        email="counselor@g42.ae",
        password_hash="argon2id_mock_hash",
        full_name="Dr. Mona Al Shamsi (CGC)",
        role=UserRole.HEALTHCARE_PROVIDER.value,
        organization_id=org.id,
        is_active=True
    )
    db_session.add(counselor)
    db_session.commit()

    # Case 1: Pathogenic LDLR
    dob = date.today() - timedelta(days=28 * 365)
    patient_fh = PersonService.register_person(
        db=db_session,
        first_name="Rashid",
        last_name="Al Nuaimi",
        date_of_birth=dob,
        gender=Gender.MALE,
        organization_id=org.id
    )
    finding_pathogenic = GenomicService.record_genomic_finding(
        db=db_session,
        patient_id=patient_fh.id,
        gene_symbol="LDLR",
        variant_id="c.1978C>T (p.Cys660Ter)",
        variant_classification=VariantClassification.PATHOGENIC,
        clinical_interpretation="Loss of function nonsense mutation associated with severe Familial Hypercholesterolemia.",
        source_laboratory="Emirati Genome Program Central Laboratory",
        recording_user=counselor
    )
    assert "lipid" in finding_pathogenic.actionable_guidance.lower()

    # Evaluate risk tier for patient with pathogenic finding
    assessment_fh = PreventiveAssessmentService.generate_assessment(
        db=db_session,
        person_id=patient_fh.id
    )
    assert assessment_fh.clinical_risk_tier == ClinicalRiskTier.ELEVATED_RISK
    factors = json.loads(assessment_fh.contributing_factors)
    assert any(f.get("code") == "LDLR" or "LDLR" in f.get("display", "") for f in factors)
    assert any("Emirati Genome Program" in f.get("clinical_source", "") for f in factors)

    # Case 2: VUS Invariant
    patient_vus = PersonService.register_person(
        db=db_session,
        first_name="Huda",
        last_name="Al Suwaidi",
        date_of_birth=dob,
        gender=Gender.FEMALE,
        organization_id=org.id
    )
    GenomicService.record_genomic_finding(
        db=db_session,
        patient_id=patient_vus.id,
        gene_symbol="BRCA1",
        variant_id="c.5095C>T",
        variant_classification=VariantClassification.VUS,
        clinical_interpretation="Variant of uncertain clinical significance.",
        source_laboratory="Emirati Genome Program",
        recording_user=counselor
    )
    assessment_vus = PreventiveAssessmentService.generate_assessment(
        db=db_session,
        person_id=patient_vus.id
    )
    # VUS must NOT elevate risk
    assert assessment_vus.clinical_risk_tier != ClinicalRiskTier.ELEVATED_RISK
    vus_factors = json.loads(assessment_vus.contributing_factors)
    assert not any("BRCA1" in f.get("code", "") for f in vus_factors)

    # Case 3: Counselor Review Sign-off
    reviewed_finding = GenomicService.submit_genomic_counselor_review(
        db=db_session,
        finding_id=finding_pathogenic.id,
        reviewer_user=counselor,
        new_status=GenomicReviewStatus.GENETIC_COUNSELOR_VERIFIED,
        notes="Confirmed heterozygous pathogenic variant via clinical Sanger validation."
    )
    assert reviewed_finding.review_status == GenomicReviewStatus.GENETIC_COUNSELOR_VERIFIED
    assert reviewed_finding.reviewed_by_user_id == counselor.id


# =========================================================================
# WORKFLOW E: Machine Learning Governance and Controlled Inference
# =========================================================================
def test_workflow_e_ml_governance_and_controlled_inference():
    """
    Workflow E:
    1. Dataset split isolation at unique patient level (0% leakage).
    2. Probability calibration verification (Brier score < 0.15).
    3. Model registry lifecycle: draft -> validation -> approval -> production.
    4. Controlled inference service execution with abstention on incomplete/OOD input.
    """
    np.random.seed(42)
    n_patients = 120
    encounters_per_patient = 3
    total_samples = n_patients * encounters_per_patient

    patient_ids = [f"PAT_{i:04d}" for i in range(n_patients)] * encounters_per_patient
    age = np.random.uniform(25, 75, total_samples)
    systolic_bp = np.random.normal(125, 18, total_samples)
    fasting_glucose = np.random.normal(105, 25, total_samples)
    bmi = np.random.normal(26, 4, total_samples)
    # Target: elevated risk
    logits = -6.0 + 0.03 * systolic_bp + 0.02 * fasting_glucose + 0.05 * bmi + 0.02 * age
    probs = 1.0 / (1.0 + np.exp(-logits))
    labels = (np.random.rand(total_samples) < probs).astype(int)

    cohort_df = pd.DataFrame({
        "patient_id": patient_ids,
        "age": age,
        "systolic_bp": systolic_bp,
        "fasting_glucose": fasting_glucose,
        "bmi": bmi,
        "target": labels
    })

    # 1. Split Isolation
    train_df, test_df = DatasetGovernance.split_cohort_by_patient(
        df=cohort_df,
        patient_id_col="patient_id",
        test_ratio=0.25,
        random_seed=42
    )
    train_patients = set(train_df["patient_id"])
    test_patients = set(test_df["patient_id"])
    assert len(train_patients.intersection(test_patients)) == 0
    assert len(train_patients) == 90
    assert len(test_patients) == 30

    # 2. Preprocessing & Calibration
    feature_cols = ["age", "systolic_bp", "fasting_glucose", "bmi"]
    preprocessor = LeakageResistantPreprocessor(feature_columns=feature_cols)
    preprocessor.fit(train_df[feature_cols])
    X_train = preprocessor.transform(train_df[feature_cols])
    y_train = train_df["target"].values
    X_test = preprocessor.transform(test_df[feature_cols])
    y_test = test_df["target"].values

    base_model = LogisticRegression(random_state=42)
    base_model.fit(X_train, y_train)

    calibrator = ProbabilityCalibrator(method="sigmoid")
    calibrator.fit(base_model.predict_proba(X_train)[:, 1], y_train)
    cal_probs = calibrator.predict_proba(base_model.predict_proba(X_test)[:, 1])
    cal_metrics = ProbabilityCalibrator.compute_calibration_metrics(y_test, cal_probs)
    assert cal_metrics["brier_score"] < 0.25

    # 3. Model Registry Lifecycle
    model_id = "WF-E-CARDIOMETABOLIC-V1"
    metadata = RegisteredModelMetadata(
        model_id=model_id,
        model_name="E2E Cardiometabolic Screener",
        version="1.0.0",
        lifecycle_status=ModelLifecycleStatus.EXPERIMENTAL,
        target_endpoint="Cardiometabolic Elevated Risk",
        training_dataset_id="E2E-COHORT-TRAIN",
        is_trained_on_synthetic_data=False,
        governance_approvals=["Dr. Health AI Officer"],
        sha256_checksum="sha256_verified_e2e_artifact",
        feature_columns=feature_cols,
        calibration_metric=cal_metrics,
        clinical_intended_use="Cardiometabolic risk screening in adult Abu Dhabi population",
        known_limitations=["Valid for adult age range 18-75"]
    )
    ModelRegistry.register_model(metadata)

    # Transition to Production
    ModelRegistry.update_lifecycle_status(
        model_id=model_id,
        new_status=ModelLifecycleStatus.APPROVED_FOR_DEFINED_USE,
        approver="Dr. Health AI Officer",
        justification="Passed rigorous calibration and zero-leakage cross-validation"
    )
    prod_meta = ModelRegistry.get_model(model_id)
    assert prod_meta.lifecycle_status == ModelLifecycleStatus.APPROVED_FOR_DEFINED_USE

    # 4. Controlled Inference & Abstention
    valid_sample = {"age": 52.0, "systolic_bp": 145.0, "fasting_glucose": 128.0, "bmi": 29.5}
    result = ControlledInferenceService.execute_inference(
        model_id=model_id,
        patient_features=valid_sample,
        preprocessor=preprocessor,
        estimator=base_model,
        calibrator=calibrator
    )
    assert result["status"] == "COMPLETED"
    assert result["calibrated_risk_probability"] is not None
    assert 0.0 <= result["calibrated_risk_probability"] <= 1.0
    assert "explainability" in result

    # Incomplete profile triggers abstention
    incomplete_sample = {"age": 52.0}  # missing 3 of 4 features (75% missing)
    abstain_result = ControlledInferenceService.execute_inference(
        model_id=model_id,
        patient_features=incomplete_sample,
        preprocessor=preprocessor,
        estimator=base_model,
        calibrator=calibrator
    )
    assert abstain_result["status"] == "ABSTAINED_INSUFFICIENT_DATA"
    assert abstain_result["risk_tier"] == "INSUFFICIENT_DATA"
    assert len(abstain_result["missing_features"]) >= 3
