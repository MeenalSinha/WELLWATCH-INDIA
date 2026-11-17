from datetime import date, datetime
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field
from backend.app.models.person import CompletenessStatus, Gender, IdentifierNamespace, PopulationClassification, RecordStatus
from backend.app.models.screening import EpistemicStatus


class PersonIdentifierSchema(BaseModel):
    namespace: IdentifierNamespace
    identifier_value: str
    issuing_authority: str = "ICP UAE"
    verification_status: str = "UNVERIFIED"


class PersonCreate(BaseModel):
    first_name: str = Field(..., min_length=1, max_length=100)
    last_name: str = Field(..., min_length=1, max_length=100)
    date_of_birth: date
    gender: Gender
    preferred_language: str = Field("ar", max_length=10)
    nationality: str = Field("ARE", max_length=50)
    emirate: str = Field("Abu Dhabi", max_length=50)
    district: Optional[str] = None
    
    # Optional Primary Identifier
    identifier_namespace: Optional[IdentifierNamespace] = IdentifierNamespace.EMIRATES_ID
    identifier_value: Optional[str] = None
    
    # Optional guardian linkage for pediatric registration
    guardian_user_id: Optional[str] = None
    relationship_to_guardian: Optional[str] = None


class PersonSummaryResponse(BaseModel):
    id: str
    full_name: str
    date_of_birth: date
    age_years: int
    age_months: int
    gender: Gender
    emirate: str
    population_classification: PopulationClassification
    completeness_score: float
    completeness_status: CompletenessStatus
    record_status: RecordStatus
    created_at: datetime


class PersonListResponse(BaseModel):
    items: List[PersonSummaryResponse]
    total: int
    skip: int
    limit: int


class TimelineEvent(BaseModel):
    event_id: str
    event_type: str                     # "ENCOUNTER", "OBSERVATION", "SCREENING", "CONDITION", "FAMILY_HISTORY"
    title: str
    description: str
    timestamp: datetime
    source_name: str
    epistemic_status: EpistemicStatus
    value_display: Optional[str] = None


class TimelineResponse(BaseModel):
    patient_id: str
    total_events: int
    events: List[TimelineEvent]


class ObservationCreate(BaseModel):
    code: str                           # LOINC code, e.g. "8480-6"
    display_name: str
    numeric_value: Optional[float] = None
    string_value: Optional[str] = None
    unit: Optional[str] = None
    collection_time: Optional[datetime] = None
    observation_type: str = "VITAL_SIGN"
    notes: Optional[str] = None


class FamilyHistoryCreate(BaseModel):
    relationship_to_patient: str        # "FATHER", "MOTHER", etc.
    condition_name: str
    condition_code: Optional[str] = None
    approximate_age_of_onset: Optional[int] = None
    is_deceased: int = 0


class ScreeningRecordCreate(BaseModel):
    screening_type: str                 # e.g. "WELL_CHILD_12M", "IFHAS_DIABETES_PANEL"
    screening_date: date
    screening_status: str = "COMPLETED"
    findings_summary: Optional[str] = None
    completion_evidence_ref: Optional[str] = None


class PediatricProfileResponse(BaseModel):
    person: PersonSummaryResponse
    guardians: List[Dict[str, Any]]
    well_child_visits: List[Dict[str, Any]]
    developmental_milestones: List[Dict[str, Any]]
    family_history: List[Dict[str, Any]]
    completeness: Dict[str, Any]


class AdultProfileResponse(BaseModel):
    person: PersonSummaryResponse
    latest_vitals: Dict[str, Any]
    chronic_conditions: List[Dict[str, Any]]
    lifestyle_indicators: Dict[str, Any]
    screening_history: List[Dict[str, Any]]
    family_history: List[Dict[str, Any]]
    completeness: Dict[str, Any]
