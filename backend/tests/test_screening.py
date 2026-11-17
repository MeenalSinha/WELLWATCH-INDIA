import pytest
from pydantic import ValidationError
from backend.app.models.screening import EncounterType, EpistemicStatus
from backend.app.models.user import UserRole
from backend.app.schemas.screening import ScreeningCreate
from backend.app.services.screening_service import ScreeningService


def test_valid_screening_encounter_and_bmi_calculation(db_session, test_users, test_organization):
    provider = test_users[UserRole.HEALTHCARE_PROVIDER]
    patient = test_users[UserRole.INDIVIDUAL]

    screening_dto = ScreeningCreate(
        patient_id=patient.id,
        facility_id=test_organization.id,
        encounter_type=EncounterType.IFHAS_COMPREHENSIVE,
        systolic_bp=135.0,
        diastolic_bp=85.0,
        pulse_rate=72,
        fasting_glucose=105.0,
        height_cm=180.0,
        weight_kg=81.0,
        epistemic_status=EpistemicStatus.VERIFIED,
        notes="Normal baseline check."
    )

    encounter = ScreeningService.create_screening_encounter(
        db=db_session,
        provider=provider,
        data=screening_dto,
        correlation_id="cid-scr-01"
    )

    assert encounter.id is not None
    # 81 / (1.8^2) = 81 / 3.24 = 25.0
    assert encounter.bmi == 25.0
    assert encounter.epistemic_status == EpistemicStatus.VERIFIED.value
    assert encounter.systolic_bp == 135.0
    assert encounter.diastolic_bp == 85.0


def test_invalid_blood_pressure_rejected():
    with pytest.raises(ValidationError, match="Diastolic BP must be strictly less than Systolic BP"):
        ScreeningCreate(
            patient_id="patient-1",
            facility_id="facility-1",
            encounter_type=EncounterType.COMMUNITY_VITALS_CHECK,
            systolic_bp=120.0,
            diastolic_bp=130.0  # Invalid!
        )


def test_out_of_bounds_blood_pressure_rejected():
    with pytest.raises(ValidationError):
        ScreeningCreate(
            patient_id="patient-1",
            facility_id="facility-1",
            encounter_type=EncounterType.COMMUNITY_VITALS_CHECK,
            systolic_bp=300.0  # Max is 260
        )
