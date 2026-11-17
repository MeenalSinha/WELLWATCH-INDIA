from datetime import datetime
from typing import Optional
from pydantic import BaseModel, ConfigDict, Field
from backend.app.models.guardian import GuardianshipStatus, RelationshipType


class GuardianshipCreate(BaseModel):
    dependent_user_id: str
    relationship_type: RelationshipType
    verification_evidence_ref: Optional[str] = Field(None, description="Registry or civil documentation ID")


class GuardianshipVerify(BaseModel):
    verification_status: GuardianshipStatus
    notes: Optional[str] = None


class GuardianshipRead(BaseModel):
    id: str
    guardian_user_id: str
    dependent_user_id: str
    relationship_type: str
    verification_status: str
    verification_evidence_ref: Optional[str] = None
    verified_by_user_id: Optional[str] = None
    created_at: datetime
    updated_at: datetime

    model_config = ConfigDict(from_attributes=True)
