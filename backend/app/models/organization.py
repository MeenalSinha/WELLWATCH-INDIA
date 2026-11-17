from sqlalchemy import Boolean, Column, String
from sqlalchemy.orm import relationship
from backend.app.models.base import BaseDBModel


class Organization(BaseDBModel):
    """Healthcare organization, hospital, clinic, or public health centre."""
    __tablename__ = "organizations"

    name = Column(String(255), unique=True, nullable=False, index=True)
    license_number = Column(String(100), unique=True, nullable=False, index=True)
    organization_type = Column(String(50), nullable=False)  # HEALTH_AUTHORITY, HOSPITAL, CLINIC, COMMUNITY_CENTER
    region = Column(String(50), nullable=False)  # ABU_DHABI_CITY, AL_AIN, AL_DHAFRA
    is_active = Column(Boolean, default=True, nullable=False)

    users = relationship("User", back_populates="organization", cascade="all, delete-orphan")
    screenings = relationship("ScreeningEncounter", back_populates="facility")
