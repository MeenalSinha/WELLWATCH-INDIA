"""
Clinical Predictive Target Definitions.
Defines formalized clinical outcomes for machine learning prevention models,
ensuring objective, clinically validated endpoints with explicit observational windows.
"""
from typing import Dict, Any
from enum import Enum


class ClinicalTargetType(str, Enum):
    CARDIOMETABOLIC_ONSET_3YR = "CARDIOMETABOLIC_ONSET_3YR"
    DIABETES_PROGRESSION_3YR = "DIABETES_PROGRESSION_3YR"
    SCREENING_NON_ADHERENCE = "SCREENING_NON_ADHERENCE"


class TargetDefinition:
    """
    Formal specification of a clinical predictive target.
    """

    TARGET_SPECS: Dict[ClinicalTargetType, Dict[str, Any]] = {
        ClinicalTargetType.CARDIOMETABOLIC_ONSET_3YR: {
            "target_name": "3-Year Cardiometabolic Risk Indicator",
            "prediction_window_months": 36,
            "inclusion_criteria": "Adults aged 18-70 without baseline documented myocardial infarction, stroke, or end-stage renal disease",
            "clinical_endpoint": "New onset of Stage 2 Hypertension (>=140/90 mmHg) or Impaired Fasting Glucose / T2D (>=126 mg/dL)",
            "adphc_guideline": "ADPHC Cardiovascular and Metabolic Disease Prevention Standards",
            "positive_label_definition": "Onset of target clinical criteria within 36 months of baseline screening"
        },
        ClinicalTargetType.DIABETES_PROGRESSION_3YR: {
            "target_name": "3-Year Type 2 Diabetes Progression",
            "prediction_window_months": 36,
            "inclusion_criteria": "Adults with impaired fasting glucose (100-125 mg/dL) at baseline",
            "clinical_endpoint": "Progression to Fasting Blood Glucose >= 126 mg/dL or HbA1c >= 6.5%",
            "adphc_guideline": "DoH Diabetes Clinical Practice Guidelines",
            "positive_label_definition": "Confirmed diabetes diagnosis within 36 months"
        },
        ClinicalTargetType.SCREENING_NON_ADHERENCE: {
            "target_name": "Preventive Screening Non-Adherence",
            "prediction_window_months": 12,
            "inclusion_criteria": "Eligible adults and pediatric dependents due for scheduled preventive visit",
            "clinical_endpoint": "Failure to complete eligible periodic screening within 90 days after due date",
            "adphc_guideline": "ADPHC Preventive Care Outreach Standards",
            "positive_label_definition": "Care gap remains open past grace period"
        }
    }

    @classmethod
    def get_specification(cls, target_type: ClinicalTargetType) -> Dict[str, Any]:
        if target_type not in cls.TARGET_SPECS:
            raise ValueError(f"Unknown target type '{target_type}'.")
        return cls.TARGET_SPECS[target_type]
