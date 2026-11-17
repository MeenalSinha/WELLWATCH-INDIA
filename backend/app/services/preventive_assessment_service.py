import json
from datetime import date, datetime, timezone
from typing import Any, Dict, List, Optional, Tuple
from sqlalchemy.orm import Session
from sqlalchemy import desc

from backend.app.models.assessment import (
    AssessmentStatus,
    ClinicalRiskTier,
    HumanReviewStatus,
    PreventiveAssessment,
)
from backend.app.models.clinical import (
    ClinicalCondition,
    ClinicalObservation,
    FamilyHistory,
    ObservationType,
    ScreeningRecord,
)
from backend.app.models.person import Person, PopulationClassification
from backend.app.models.screening import EpistemicStatus
from backend.app.services.pathway_engine import CareGapStatus, PathwayEngine


class PreventiveAssessmentService:
    """
    Orchestrates the Explainable Preventive Risk Intelligence framework.
    Strictly decouples:
      1. Clinical Risk Tier (Grounded clinical indicators)
      2. Preventive Program Eligibility (from PathwayEngine)
      3. Confirmed Care Gaps (from PathwayEngine)
      4. Data Confidence & Epistemic Uncertainty (Abstention)
      5. Human Clinical Review
    """

    @classmethod
    def evaluate_clinical_risk_indicators(
        cls,
        db: Session,
        person: Person
    ) -> Tuple[ClinicalRiskTier, List[Dict[str, Any]], List[Dict[str, Any]], AssessmentStatus]:
        """
        Evaluates evidence-based clinical risk indicators from verified measurements.
        Returns: (risk_tier, contributing_factors, data_limitations, assessment_status)
        """
        factors: List[Dict[str, Any]] = []
        limitations: List[Dict[str, Any]] = []
        status = AssessmentStatus.COMPLETED

        # 1. Fetch latest clinical observations
        observations = db.query(ClinicalObservation).filter(
            ClinicalObservation.patient_id == person.id
        ).order_by(desc(ClinicalObservation.collection_time)).all()

        obs_by_code: Dict[str, ClinicalObservation] = {}
        for o in observations:
            if o.code not in obs_by_code:
                obs_by_code[o.code] = o

        # Check for epistemic conflicts or outdated data
        for code, o in obs_by_code.items():
            if o.epistemic_status == EpistemicStatus.CONFLICTING:
                limitations.append({
                    "code": code,
                    "name": o.display_name,
                    "issue": "CONFLICTING_MEASUREMENTS",
                    "detail": "Recent measurements from different sources disagree beyond clinical threshold."
                })
                status = AssessmentStatus.REQUIRES_REVIEW
            elif o.epistemic_status == EpistemicStatus.OUTDATED:
                limitations.append({
                    "code": code,
                    "name": o.display_name,
                    "issue": "OUTDATED_MEASUREMENT",
                    "detail": f"Observation collected on {o.collection_time.date().isoformat()} exceeds 12-month recency window."
                })

        # Check conditions
        conditions = db.query(ClinicalCondition).filter(
            ClinicalCondition.patient_id == person.id
        ).all()
        from backend.app.services.genomic_service import GenomicService
        genomic_eval = GenomicService.evaluate_genomic_risk_factors(db=db, patient_id=person.id)

        # Insufficient data check: If person is Adult and has zero vitals, zero documented conditions, and zero pathogenic genomics
        if person.patient_profile.population_classification != PopulationClassification.PEDIATRIC:
            if not obs_by_code and not conditions and not genomic_eval["has_pathogenic_risk"]:
                limitations.append({
                    "code": "CORE_VITALS",
                    "issue": "MISSING_BASELINE",
                    "detail": "No baseline blood pressure, glucose, or BMI measurements on file."
                })
                for uf in genomic_eval["uncertain_findings"]:
                    limitations.append({
                        "code": f"GENOMIC_VUS_{uf['gene']}",
                        "issue": "VARIANT_OF_UNCERTAIN_SIGNIFICANCE",
                        "detail": uf["note"]
                    })
                return ClinicalRiskTier.INSUFFICIENT_DATA, factors, limitations, AssessmentStatus.NEEDS_DATA
            elif not obs_by_code:
                limitations.append({
                    "code": "CORE_VITALS",
                    "issue": "MISSING_VITALS",
                    "detail": "Patient has documented clinical diagnoses or genetic findings but lacks recent physiological measurements."
                })

        elevated_count = 0
        moderate_count = 0

        # Indicator A: Blood Pressure (Systolic 8480-6, Diastolic 8462-4)
        sys_obs = obs_by_code.get("8480-6")
        dia_obs = obs_by_code.get("8462-4")
        if sys_obs and sys_obs.canonical_value is not None:
            val = sys_obs.canonical_value
            if val >= 140.0:
                elevated_count += 1
                factors.append({
                    "indicator": "STAGE_2_HYPERTENSION_SYSTOLIC",
                    "code": "8480-6",
                    "display": "Elevated Systolic Blood Pressure",
                    "value": f"{val} mmHg",
                    "clinical_source": "DoH Primary Care Hypertension Guidelines",
                    "severity": "ELEVATED",
                    "provenance_ref": sys_obs.provenance_id
                })
            elif val >= 130.0:
                moderate_count += 1
                factors.append({
                    "indicator": "STAGE_1_HYPERTENSION_SYSTOLIC",
                    "code": "8480-6",
                    "display": "Pre-Hypertension / Stage 1 Systolic",
                    "value": f"{val} mmHg",
                    "clinical_source": "DoH Primary Care Hypertension Guidelines",
                    "severity": "MODERATE",
                    "provenance_ref": sys_obs.provenance_id
                })

        if dia_obs and dia_obs.canonical_value is not None:
            val = dia_obs.canonical_value
            if val >= 90.0:
                elevated_count += 1
                factors.append({
                    "indicator": "STAGE_2_HYPERTENSION_DIASTOLIC",
                    "code": "8462-4",
                    "display": "Elevated Diastolic Blood Pressure",
                    "value": f"{val} mmHg",
                    "clinical_source": "DoH Primary Care Hypertension Guidelines",
                    "severity": "ELEVATED",
                    "provenance_ref": dia_obs.provenance_id
                })
            elif val >= 80.0:
                moderate_count += 1
                factors.append({
                    "indicator": "STAGE_1_HYPERTENSION_DIASTOLIC",
                    "code": "8462-4",
                    "display": "Pre-Hypertension / Stage 1 Diastolic",
                    "value": f"{val} mmHg",
                    "clinical_source": "DoH Primary Care Hypertension Guidelines",
                    "severity": "MODERATE",
                    "provenance_ref": dia_obs.provenance_id
                })

        # Indicator B: Fasting Blood Glucose (2339-0)
        glu_obs = obs_by_code.get("2339-0")
        if glu_obs and glu_obs.canonical_value is not None:
            val = glu_obs.canonical_value
            if val >= 126.0:
                elevated_count += 1
                factors.append({
                    "indicator": "DIABETES_RANGE_FASTING_GLUCOSE",
                    "code": "2339-0",
                    "display": "Fasting Blood Glucose (Diabetes Range)",
                    "value": f"{val} mg/dL",
                    "clinical_source": "DoH Diabetes Clinical Practice Guidelines",
                    "severity": "ELEVATED",
                    "provenance_ref": glu_obs.provenance_id
                })
            elif val >= 100.0:
                moderate_count += 1
                factors.append({
                    "indicator": "IMPAIRED_FASTING_GLUCOSE_PREDIABETES",
                    "code": "2339-0",
                    "display": "Impaired Fasting Glucose (Prediabetes Range)",
                    "value": f"{val} mg/dL",
                    "clinical_source": "DoH Diabetes Clinical Practice Guidelines",
                    "severity": "MODERATE",
                    "provenance_ref": glu_obs.provenance_id
                })

        # Indicator C: Body Mass Index (39156-5)
        bmi_obs = obs_by_code.get("39156-5")
        if bmi_obs and bmi_obs.canonical_value is not None:
            val = bmi_obs.canonical_value
            if val >= 35.0:
                elevated_count += 1
                factors.append({
                    "indicator": "OBESITY_CLASS_2",
                    "code": "39156-5",
                    "display": "Obesity Class II+",
                    "value": f"{val} kg/m2",
                    "clinical_source": "ADPHC Adult Health Standards",
                    "severity": "ELEVATED",
                    "provenance_ref": bmi_obs.provenance_id
                })
            elif val >= 30.0:
                moderate_count += 1
                factors.append({
                    "indicator": "OBESITY_CLASS_1",
                    "code": "39156-5",
                    "display": "Obesity Class I",
                    "value": f"{val} kg/m2",
                    "clinical_source": "ADPHC Adult Health Standards",
                    "severity": "MODERATE",
                    "provenance_ref": bmi_obs.provenance_id
                })

        # Indicator D: Documented Chronic Conditions (ICD-10)
        for c in conditions:
            elevated_count += 1
            factors.append({
                "indicator": f"CHRONIC_CONDITION_{c.code}",
                "code": c.code,
                "display": f"Documented Condition: {c.clinical_description}",
                "value": c.clinical_status.value if hasattr(c.clinical_status, 'value') else c.clinical_status,
                "clinical_source": "Verified Clinical Diagnosis",
                "severity": "ELEVATED"
            })

        # Indicator E: Family History
        fam = db.query(FamilyHistory).filter(FamilyHistory.patient_id == person.id).all()
        for f in fam:
            moderate_count += 1
            factors.append({
                "indicator": "HEREDITARY_RISK",
                "code": f.condition_code or "FAM-HIST",
                "display": f"Family History ({f.relationship_to_patient.value if hasattr(f.relationship_to_patient, 'value') else f.relationship_to_patient}): {f.condition_name}",
                "value": f"Onset age {f.approximate_age_of_onset}" if f.approximate_age_of_onset else "Positive history",
                "clinical_source": "Patient Reported Family History",
                "severity": "MODERATE"
            })

        # Indicator F: Genomic Intelligence (Phase 8)
        if genomic_eval["has_pathogenic_risk"]:
            for gf in genomic_eval["actionable_factors"]:
                elevated_count += 1
                factors.append({
                    "indicator": gf["indicator"],
                    "code": gf["gene"],
                    "display": f"Genomic Pathogenic Variant: {gf['gene']} ({gf['condition']})",
                    "value": gf["variant_id"],
                    "clinical_source": gf["clinical_source"],
                    "severity": "ELEVATED"
                })

        for uf in genomic_eval["uncertain_findings"]:
            limitations.append({
                "code": f"GENOMIC_VUS_{uf['gene']}",
                "issue": "VARIANT_OF_UNCERTAIN_SIGNIFICANCE",
                "detail": uf["note"]
            })

        # Determine Tier
        if elevated_count >= 1 or moderate_count >= 3:
            tier = ClinicalRiskTier.ELEVATED_RISK
        elif moderate_count >= 1:
            tier = ClinicalRiskTier.MODERATE_RISK
        else:
            tier = ClinicalRiskTier.LOW_RISK

        return tier, factors, limitations, status

    @classmethod
    def generate_assessment(
        cls,
        db: Session,
        person_id: str,
        pathway_id: Optional[str] = None,
        evaluation_date: Optional[date] = None,
        requesting_user_id: Optional[str] = None
    ) -> PreventiveAssessment:
        """
        Generates and persists a comprehensive, explainable preventive assessment.
        Creates an immutable historical snapshot.
        """
        person = db.query(Person).filter(Person.id == person_id).first()
        if not person:
            raise ValueError("Patient not found.")

        eval_date = evaluation_date or date.today()

        # Select appropriate default pathway based on population classification
        pop_class = person.patient_profile.population_classification
        if not pathway_id:
            pathway_id = "DOH-PEDIATRIC-WELLCHILD-V1" if pop_class == PopulationClassification.PEDIATRIC else "ADPHC-ADULT-IFHAS-V1"

        pathway = PathwayEngine.load_pathway(pathway_id)
        if not pathway:
            raise ValueError(f"Pathway '{pathway_id}' not found.")

        # 1. Clinical Risk Indicator Evaluation
        risk_tier, factors, limitations, risk_status = cls.evaluate_clinical_risk_indicators(db=db, person=person)

        # 2. Pathway Eligibility Evaluation
        eligibility = PathwayEngine.evaluate_eligibility(pathway=pathway, person=person, evaluation_date=eval_date)

        # 3. Care-Gap Evaluation
        care_gaps = []
        if eligibility.get("is_eligible"):
            care_gaps = PathwayEngine.evaluate_care_gaps(db=db, pathway=pathway, person=person, evaluation_date=eval_date)

        # Determine Overall Assessment Status & Recommendation
        has_due_gaps = any(g.get("status") in [CareGapStatus.DUE.value, CareGapStatus.OVERDUE.value] for g in care_gaps)
        has_limitations = len(limitations) > 0

        if risk_status != AssessmentStatus.COMPLETED:
            overall_status = risk_status
        elif has_limitations:
            overall_status = AssessmentStatus.REQUIRES_REVIEW
        else:
            overall_status = AssessmentStatus.COMPLETED

        # Recommended Next Action
        if risk_tier == ClinicalRiskTier.ELEVATED_RISK:
            rec_action = "Priority clinical consultation recommended for elevated health indicators."
        elif has_due_gaps:
            rec_action = "Schedule overdue preventive screening encounter with primary care clinic."
        elif risk_status == AssessmentStatus.NEEDS_DATA:
            rec_action = "Obtain baseline health vitals and update screening history."
        else:
            rec_action = "Maintain regular preventive lifestyle; repeat screening per scheduled interval."

        # Human Review Queue: If elevated or requiring review, put in queue
        review_status = HumanReviewStatus.AWAITING_REVIEW if (risk_tier == ClinicalRiskTier.ELEVATED_RISK or overall_status == AssessmentStatus.REQUIRES_REVIEW) else HumanReviewStatus.REVIEWED

        assessment = PreventiveAssessment(
            patient_id=person.id,
            assessment_type="COMPREHENSIVE_PREVENTIVE_ASSESSMENT",
            population_category=pop_class.value if hasattr(pop_class, 'value') else pop_class,
            pathway_id=pathway.get("pathway_id"),
            pathway_version=pathway.get("version"),
            assessment_status=overall_status,
            clinical_risk_tier=risk_tier,
            eligibility_summary=json.dumps(eligibility),
            care_gaps_summary=json.dumps(care_gaps),
            contributing_factors=json.dumps(factors),
            data_quality_limitations=json.dumps(limitations),
            human_review_status=review_status,
            recommended_action=rec_action
        )
        db.add(assessment)
        db.commit()
        db.refresh(assessment)
        return assessment

    @classmethod
    def submit_clinical_review(
        cls,
        db: Session,
        assessment_id: str,
        reviewer_user: Any,
        new_status: HumanReviewStatus,
        notes: str
    ) -> PreventiveAssessment:
        """Submits authorized human clinical review and updates assessment state."""
        assessment = db.query(PreventiveAssessment).filter(PreventiveAssessment.id == assessment_id).first()
        if not assessment:
            raise ValueError("Assessment not found.")

        assessment.human_review_status = new_status
        assessment.reviewed_by_user_id = reviewer_user.id
        assessment.reviewed_at = datetime.now(timezone.utc)
        assessment.reviewer_notes = notes

        db.commit()
        db.refresh(assessment)
        return assessment
