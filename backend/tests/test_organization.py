from fastapi import HTTPException
import pytest
from backend.app.models.user import UserRole
from backend.app.security.rbac import enforce_organization_boundary


def test_cross_organization_boundary_blocked(test_users, test_organization, test_organization_2):
    # Provider belongs to test_organization
    provider = test_users[UserRole.HEALTHCARE_PROVIDER]
    provider.organization_id = test_organization.id

    # Trying to access test_organization_2 should be blocked
    with pytest.raises(HTTPException) as exc_info:
        enforce_organization_boundary(provider, test_organization_2.id)
    assert exc_info.value.status_code == 403
    assert "cross-organizational record access is strictly forbidden" in exc_info.value.detail


def test_same_organization_allowed(test_users, test_organization):
    provider = test_users[UserRole.HEALTHCARE_PROVIDER]
    provider.organization_id = test_organization.id

    # Accessing same organization succeeds without exception
    enforce_organization_boundary(provider, test_organization.id)


def test_cross_organization_operational_roles_permitted(test_users, test_organization_2):
    # System Admin can manage cross-organization settings
    admin = test_users[UserRole.SYSTEM_ADMIN]
    enforce_organization_boundary(admin, test_organization_2.id)

    # Data Governance Officer can audit cross-organization
    dpo = test_users[UserRole.DATA_GOVERNANCE_OFFICER]
    enforce_organization_boundary(dpo, test_organization_2.id)
