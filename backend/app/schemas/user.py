from datetime import datetime
from typing import Optional
from pydantic import BaseModel, ConfigDict, EmailStr, Field
from backend.app.models.user import UserRole


class UserBase(BaseModel):
    email: EmailStr
    full_name: str = Field(..., min_length=2, max_length=255)
    role: UserRole
    phone_number: Optional[str] = Field(None, max_length=50)
    organization_id: Optional[str] = None


class UserCreate(UserBase):
    password: str = Field(..., min_length=8, description="Password with minimum 8 characters")


class UserLogin(BaseModel):
    email: EmailStr
    password: str


class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    expires_in: int
    refresh_token: str
    user_id: str
    role: str
    full_name: str
    organization_id: Optional[str] = None


class RefreshTokenRequest(BaseModel):
    refresh_token: str


class UserRead(UserBase):
    id: str
    is_active: bool
    is_verified: bool
    created_at: datetime
    updated_at: datetime

    model_config = ConfigDict(from_attributes=True)
