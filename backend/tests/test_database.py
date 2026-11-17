from sqlalchemy.exc import IntegrityError
import pytest
from backend.app.models.organization import Organization
from backend.app.core.database import check_database_readiness


def test_database_readiness():
    assert check_database_readiness() is True


def test_unique_constraint_enforcement(db_session):
    org1 = Organization(
        name="Unique Health Center",
        license_number="LIC-UNIQUE-001",
        organization_type="CLINIC",
        region="ABU_DHABI_CITY"
    )
    db_session.add(org1)
    db_session.commit()

    org2 = Organization(
        name="Different Name",
        license_number="LIC-UNIQUE-001",  # Duplicate license number
        organization_type="CLINIC",
        region="ABU_DHABI_CITY"
    )
    db_session.add(org2)
    with pytest.raises(IntegrityError):
        db_session.commit()
    db_session.rollback()
