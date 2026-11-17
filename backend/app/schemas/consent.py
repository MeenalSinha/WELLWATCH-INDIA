from datetime import datetime
from typing import List, Optional
from pydantic import BaseModel, ConfigDict, Field
from backend.app.models.consent import ConsentPurpose, ConsentStatus


class ConsentCreate(BaseModel):
    user_id: Optional[str] = None  # Defaults to current user if not provided
    authorized_representative_id: Optional[str] = None
    purpose: ConsentPurpose
    data_categories: List[str] = Field(..., min_length=1, description="List of consented scopes (e.g. ['vitals', 'family_history'])")
    consent_version: str = Field(default="v1.0", max_length=50)
    collection_source: str = Field(default="PORTAL_DIRECT", max_length=50)
    valid_until: Optional[datetime] = None


class ConsentRevoke(BaseModel):
    reason: Optional[str] = None


class ConsentRead(BaseModel):
    id: str
    user_id: str
    authorized_representative_id: Optional[str] = None
    purpose: str
    data_categories: List[str]
    status: str
    consent_version: str
    collection_source: str
    valid_from: datetime
    valid_until: Optional[datetime] = None
    revoked_at: Optional[datetime] = None
    created_at: datetime

    model_config = ConfigDict(from_attributes=True)
