from fastapi import HTTPException
import pytest
from backend.app.models.user import UserRole
from backend.app.security.permissions import Permission
from backend.app.security.rbac import ROLE_PERMISSIONS, require_permission, require_roles


def test_all_8_roles_have_defined_permissions():
    for role in UserRole:
        assert role in ROLE_PERMISSIONS
        assert isinstance(ROLE_PERMISSIONS[role], set)


def test_individual_role_permissions(test_users):
    individual = test_users[UserRole.INDIVIDUAL]
    checker_read = require_permission(Permission.PROFILE_READ_OWN)
    # Individual can read own profile
    assert checker_read(individual) == individual

    # Individual cannot create clinical screenings
    checker_screen = require_permission(Permission.SCREENING_CREATE)
    with pytest.raises(HTTPException) as exc_info:
        checker_screen(individual)
    assert exc_info.value.status_code == 403
    assert "lacks permission 'screening:create'" in exc_info.value.detail


def test_chw_role_permissions(test_users):
    chw = test_users[UserRole.COMMUNITY_HEALTH_WORKER]
    checker = require_permission(Permission.SCREENING_CREATE)
    assert checker(chw) == chw

    # CHW cannot approve clinical pathways (Physician only)
    checker_pathway = require_permission(Permission.PATHWAY_APPROVE)
    with pytest.raises(HTTPException) as exc_info:
        checker_pathway(chw)
    assert exc_info.value.status_code == 403


def test_healthcare_provider_role_permissions(test_users):
    provider = test_users[UserRole.HEALTHCARE_PROVIDER]
    assert require_permission(Permission.SCREENING_CREATE)(provider) == provider
    assert require_permission(Permission.CLINICAL_REVIEW)(provider) == provider
    assert require_permission(Permission.PATHWAY_APPROVE)(provider) == provider
    assert require_permission(Permission.GAP_RESOLVE)(provider) == provider


def test_system_admin_separation_of_duties(test_users):
    admin = test_users[UserRole.SYSTEM_ADMIN]
    # Admin can manage users and orgs
    assert require_permission(Permission.USER_MANAGE)(admin) == admin
    assert require_permission(Permission.ORGANIZATION_MANAGE)(admin) == admin

    # Admin CANNOT record clinical screenings or review clinical records
    with pytest.raises(HTTPException) as exc:
        require_permission(Permission.SCREENING_CREATE)(admin)
    assert exc.value.status_code == 403

    with pytest.raises(HTTPException) as exc:
        require_permission(Permission.CLINICAL_REVIEW)(admin)
    assert exc.value.status_code == 403


def test_data_governance_officer_permissions(test_users):
    dpo = test_users[UserRole.DATA_GOVERNANCE_OFFICER]
    assert require_permission(Permission.AUDIT_READ)(dpo) == dpo
    assert require_permission(Permission.CONSENT_AUDIT)(dpo) == dpo

    # DPO cannot approve clinical pathways
    with pytest.raises(HTTPException) as exc:
        require_permission(Permission.PATHWAY_APPROVE)(dpo)
    assert exc.value.status_code == 403
