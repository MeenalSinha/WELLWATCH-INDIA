from datetime import datetime
from typing import Optional
from pydantic import BaseModel, ConfigDict, Field


class OrganizationBase(BaseModel):
    name: str = Field(..., min_length=2, max_length=255)
    license_number: str = Field(..., min_length=3, max_length=100)
    organization_type: str = Field(..., description="HEALTH_AUTHORITY, HOSPITAL, CLINIC, COMMUNITY_CENTER")
    region: str = Field(..., description="ABU_DHABI_CITY, AL_AIN, AL_DHAFRA")
    is_active: bool = True


class OrganizationCreate(OrganizationBase):
    pass


class OrganizationRead(OrganizationBase):
    id: str
    created_at: datetime
    updated_at: datetime

    model_config = ConfigDict(from_attributes=True)
