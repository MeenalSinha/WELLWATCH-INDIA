from typing import Dict, Optional, Tuple


class ClinicalValidationService:
    """
    Validates clinical observation data types, units, and physiological bounds.
    Converts multi-source units into canonical standard units without loss of raw data.
    """

    # Canonical LOINC Code Registry for Core Preventive Screening
    LOINC_REGISTRY: Dict[str, Dict[str, str]] = {
        "8480-6": {"name": "Systolic Blood Pressure", "type": "VITAL_SIGN", "canonical_unit": "mmHg"},
        "8462-4": {"name": "Diastolic Blood Pressure", "type": "VITAL_SIGN", "canonical_unit": "mmHg"},
        "8867-4": {"name": "Heart Rate / Pulse", "type": "VITAL_SIGN", "canonical_unit": "bpm"},
        "8302-2": {"name": "Body Height", "type": "ANTHROPOMETRIC", "canonical_unit": "cm"},
        "29463-7": {"name": "Body Weight", "type": "ANTHROPOMETRIC", "canonical_unit": "kg"},
        "39156-5": {"name": "Body Mass Index", "type": "ANTHROPOMETRIC", "canonical_unit": "kg/m2"},
        "2339-0": {"name": "Fasting Blood Glucose", "type": "LAB_RESULT", "canonical_unit": "mg/dL"},
        "4548-4": {"name": "Hemoglobin A1c", "type": "LAB_RESULT", "canonical_unit": "%"},
        "2093-3": {"name": "Total Cholesterol", "type": "LAB_RESULT", "canonical_unit": "mg/dL"},
        "2085-9": {"name": "HDL Cholesterol", "type": "LAB_RESULT", "canonical_unit": "mg/dL"},
        "13457-7": {"name": "LDL Cholesterol", "type": "LAB_RESULT", "canonical_unit": "mg/dL"},
        "2571-8": {"name": "Triglycerides", "type": "LAB_RESULT", "canonical_unit": "mg/dL"},
        "72166-2": {"name": "Tobacco Smoking Status", "type": "LIFESTYLE", "canonical_unit": None},
    }

    # Broad Physiological Sanity Bounds (Technical validation, not diagnostic thresholds)
    PHYSIOLOGICAL_BOUNDS: Dict[str, Tuple[float, float]] = {
        "8480-6": (40.0, 300.0),      # Systolic BP (mmHg)
        "8462-4": (30.0, 200.0),      # Diastolic BP (mmHg)
        "8867-4": (30.0, 260.0),      # Heart Rate (bpm)
        "8302-2": (30.0, 260.0),      # Height (cm)
        "29463-7": (1.0, 450.0),       # Weight (kg)
        "39156-5": (8.0, 90.0),        # BMI (kg/m2)
        "2339-0": (20.0, 800.0),      # Fasting Glucose (mg/dL)
        "4548-4": (3.0, 20.0),        # HbA1c (%)
        "2093-3": (50.0, 600.0),      # Total Cholesterol (mg/dL)
        "2085-9": (10.0, 150.0),      # HDL (mg/dL)
        "13457-7": (20.0, 450.0),     # LDL (mg/dL)
        "2571-8": (20.0, 1500.0),     # Triglycerides (mg/dL)
    }

    @classmethod
    def normalize_and_validate_observation(
        cls,
        code: str,
        numeric_value: Optional[float],
        unit: Optional[str]
    ) -> Tuple[Optional[float], Optional[str]]:
        """
        Validates numeric value against physiological bounds and normalizes to canonical unit.
        Returns: (canonical_value, canonical_unit)
        Raises: ValueError if bounds are severely violated.
        """
        if numeric_value is None:
            return None, unit

        clean_unit = (unit or "").strip().lower()
        canonical_val = numeric_value
        canonical_unit = unit

        # 1. Glucose conversions (mmol/L -> mg/dL)
        if code == "2339-0":
            canonical_unit = "mg/dL"
            if clean_unit in ["mmol/l", "mmol"]:
                canonical_val = round(numeric_value * 18.0182, 2)
            elif clean_unit in ["mg/dl", "mg%"]:
                canonical_val = numeric_value

        # 2. Height conversions (inches -> cm)
        elif code == "8302-2":
            canonical_unit = "cm"
            if clean_unit in ["in", "inch", "inches"]:
                canonical_val = round(numeric_value * 2.54, 2)
            elif clean_unit in ["m", "meter"]:
                canonical_val = round(numeric_value * 100.0, 2)

        # 3. Weight conversions (lbs -> kg)
        elif code == "29463-7":
            canonical_unit = "kg"
            if clean_unit in ["lb", "lbs", "pound", "pounds"]:
                canonical_val = round(numeric_value / 2.20462, 2)

        # 4. Cholesterol / Lipids (mmol/L -> mg/dL)
        elif code in ["2093-3", "2085-9", "13457-7"]:
            canonical_unit = "mg/dL"
            if clean_unit in ["mmol/l", "mmol"]:
                canonical_val = round(numeric_value * 38.67, 2)

        elif code == "2571-8":  # Triglycerides
            canonical_unit = "mg/dL"
            if clean_unit in ["mmol/l", "mmol"]:
                canonical_val = round(numeric_value * 88.57, 2)

        # Physiological Sanity Validation on canonical value
        if code in cls.PHYSIOLOGICAL_BOUNDS:
            min_val, max_val = cls.PHYSIOLOGICAL_BOUNDS[code]
            if canonical_val < min_val or canonical_val > max_val:
                raise ValueError(
                    f"Physiological out-of-bounds error for LOINC {code}: "
                    f"Value {canonical_val} {canonical_unit} is outside plausible range ({min_val} - {max_val})"
                )

        return canonical_val, canonical_unit

    @classmethod
    def calculate_bmi(cls, height_cm: float, weight_kg: float) -> float:
        """Computes BMI dynamically from height and weight."""
        if height_cm <= 0 or weight_kg <= 0:
            raise ValueError("Height and weight must be positive numbers.")
        height_m = height_cm / 100.0
        return round(weight_kg / (height_m * height_m), 2)
