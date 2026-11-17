from datetime import datetime
from typing import Optional
from pydantic import BaseModel, ConfigDict, Field, field_validator
from backend.app.models.screening import EncounterType, EpistemicStatus


class ScreeningCreate(BaseModel):
    patient_id: str
    facility_id: str
    encounter_type: EncounterType
    
    systolic_bp: Optional[float] = Field(None, ge=50.0, le=260.0, description="Systolic BP (mmHg)")
    diastolic_bp: Optional[float] = Field(None, ge=30.0, le=160.0, description="Diastolic BP (mmHg)")
    pulse_rate: Optional[int] = Field(None, ge=30, le=220, description="Pulse rate (bpm)")
    fasting_glucose: Optional[float] = Field(None, ge=30.0, le=600.0, description="Fasting glucose (mg/dL)")
    height_cm: Optional[float] = Field(None, ge=30.0, le=250.0, description="Height (cm)")
    weight_kg: Optional[float] = Field(None, ge=1.0, le=350.0, description="Weight (kg)")
    epistemic_status: EpistemicStatus = EpistemicStatus.UNVERIFIED
    notes: Optional[str] = Field(None, max_length=2000)

    @field_validator("diastolic_bp")
    def validate_bp_ratio(cls, v: Optional[float], info) -> Optional[float]:
        systolic = info.data.get("systolic_bp")
        if v is not None and systolic is not None:
            if v >= systolic:
                raise ValueError("Diastolic BP must be strictly less than Systolic BP.")
        return v


class ScreeningRead(BaseModel):
    id: str
    patient_id: str
    provider_id: Optional[str] = None
    facility_id: str
    encounter_type: str
    encounter_date: datetime
    systolic_bp: Optional[float] = None
    diastolic_bp: Optional[float] = None
    pulse_rate: Optional[int] = None
    fasting_glucose: Optional[float] = None
    height_cm: Optional[float] = None
    weight_kg: Optional[float] = None
    bmi: Optional[float] = None
    epistemic_status: str
    notes: Optional[str] = None
    created_at: datetime

    model_config = ConfigDict(from_attributes=True)
