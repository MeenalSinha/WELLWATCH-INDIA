import pytest
from backend.app.services.clinical_validation_service import ClinicalValidationService


def test_glucose_unit_conversion():
    # 5.5 mmol/L should be ~99.1 mg/dL
    val, unit = ClinicalValidationService.normalize_and_validate_observation(
        code="2339-0",
        numeric_value=5.5,
        unit="mmol/L"
    )
    assert unit == "mg/dL"
    assert val == pytest.approx(99.1, abs=0.2)

    # Already mg/dL should remain untouched
    val2, unit2 = ClinicalValidationService.normalize_and_validate_observation(
        code="2339-0",
        numeric_value=110.0,
        unit="mg/dL"
    )
    assert unit2 == "mg/dL"
    assert val2 == 110.0


def test_height_and_weight_conversion():
    # 70 inches should be ~177.8 cm
    h_val, h_unit = ClinicalValidationService.normalize_and_validate_observation(
        code="8302-2",
        numeric_value=70.0,
        unit="inches"
    )
    assert h_unit == "cm"
    assert h_val == 177.8

    # 165 lbs should be ~74.84 kg
    w_val, w_unit = ClinicalValidationService.normalize_and_validate_observation(
        code="29463-7",
        numeric_value=165.0,
        unit="lbs"
    )
    assert w_unit == "kg"
    assert w_val == pytest.approx(74.84, abs=0.1)


def test_physiological_bounds_enforced():
    # Blood pressure systolic out-of-bounds (> 300 mmHg) must fail
    with pytest.raises(ValueError, match="out-of-bounds error"):
        ClinicalValidationService.normalize_and_validate_observation(
            code="8480-6",
            numeric_value=350.0,
            unit="mmHg"
        )

    # Fasting glucose < 20 mg/dL must fail
    with pytest.raises(ValueError, match="out-of-bounds error"):
        ClinicalValidationService.normalize_and_validate_observation(
            code="2339-0",
            numeric_value=10.0,
            unit="mg/dL"
        )


def test_bmi_calculation():
    # 175 cm, 70 kg -> 70 / (1.75^2) = 22.86
    bmi = ClinicalValidationService.calculate_bmi(height_cm=175.0, weight_kg=70.0)
    assert bmi == 22.86

    with pytest.raises(ValueError):
        ClinicalValidationService.calculate_bmi(height_cm=-10.0, weight_kg=70.0)
