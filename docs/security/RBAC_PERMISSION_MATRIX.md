# WELLWATCH PREVENT — Role-Based Access Control & Permission Matrix
## Granular Role Definitions, Permissions, and Organizational Boundaries

**Standard:** Least-Privilege Access Control (ADHICS & HIPAA Aligned)  
**Status:** IMPLEMENTED & TESTED BASELINE (PHASE 3)  

---

## 1. The Eight System Roles

1. `INDIVIDUAL`: Citizen or resident managing personal preventive health profile and consents.
2. `PARENT_GUARDIAN`: Authorized adult with verified legal relationship to pediatric dependents.
3. `COMMUNITY_HEALTH_WORKER`: Frontline healthcare worker recording vitals and managing local outreach.
4. `HEALTHCARE_PROVIDER`: Licensed physician or nurse reviewing risk profiles, approving pathways, and recording clinical diagnoses.
5. `CARE_COORDINATOR`: Operational specialist scheduling appointments, managing referrals, and tracking care-gap resolution evidence.
6. `POPULATION_HEALTH_ADMIN`: Public health leader monitoring de-identified population coverage and health disparities.
7. `SYSTEM_ADMIN`: IT and identity administrator managing users, facilities, and platform configuration.
8. `DATA_GOVERNANCE_OFFICER`: Compliance and privacy officer inspecting audit logs and managing consent enforcement.

---

## 2. Granular Permission Catalog

| Permission Key | Scope Description |
|---|---|
| `profile:read_own` | View own demographic and longitudinal health records. |
| `profile:update_own` | Update own communication preferences and contact details. |
| `dependent:read` | View verified pediatric dependent's clinical records. |
| `screening:create` | Record clinical screening observations and vitals. |
| `screening:read_scoped`| View screening records within user's assigned facility. |
| `clinical:review` | Conduct deep clinical review and view AI risk attribution. |
| `pathway:approve` | Confirm or override clinical pathway assignments. |
| `outreach:manage` | Dispatch invitations and coordinate appointments. |
| `gap:resolve` | Verify clinical encounter evidence and close open care gaps. |
| `analytics:population` | Access aggregated, de-identified population health dashboards. |
| `user:manage` | Provision user accounts and assign roles. |
| `organization:manage` | Configure healthcare organizations and clinics. |
| `audit:read` | Inspect system-wide tamper-evident audit logs. |
| `consent:audit` | Audit consent records, scopes, and revocations. |

---

## 3. The Granular Access Matrix

| Permission | Indiv | Parent | CHW | Provider | Coord | PopAdmin | SysAdmin | GovOfficer |
|---|---|---|---|---|---|---|---|---|
| `profile:read_own` | ✅ | ✅ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ |
| `profile:update_own` | ✅ | ✅ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ |
| `dependent:read` | ❌ | ✅ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ |
| `screening:create` | ❌ | ❌ | ✅ | ✅ | ❌ | ❌ | ❌ | ❌ |
| `screening:read_scoped` | ❌ | ❌ | ✅ | ✅ | ✅ | ❌ | ❌ | ❌ |
| `clinical:review` | ❌ | ❌ | ❌ | ✅ | ❌ | ❌ | ❌ | ❌ |
| `pathway:approve` | ❌ | ❌ | ❌ | ✅ | ❌ | ❌ | ❌ | ❌ |
| `outreach:manage` | ❌ | ❌ | ✅ | ❌ | ✅ | ❌ | ❌ | ❌ |
| `gap:resolve` | ❌ | ❌ | ❌ | ✅ | ✅ | ❌ | ❌ | ❌ |
| `analytics:population` | ❌ | ❌ | ❌ | ❌ | ❌ | ✅ | ❌ | ❌ |
| `user:manage` | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ✅ | ❌ |
| `organization:manage` | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ✅ | ❌ |
| `audit:read` | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ✅ |
| `consent:audit` | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ✅ |

---

## 4. Enforcement Architecture

### 4.1 Dependency Injection Guards
Endpoints enforce permissions declaratively at the route declaration:
```python
@router.post("", status_code=201)
def record_screening(
    screening_in: ScreeningCreate,
    current_user: User = Depends(require_permission(Permission.SCREENING_CREATE))
):
    ...
```
If the authenticated user's role lacks the permission, FastAPI raises `HTTP 403 Forbidden` immediately before executing any business logic.

### 4.2 Separation of Duties (SysAdmin vs Clinical PHI)
A critical healthcare invariant is enforced: **System Administrators cannot view patient health records or record clinical vitals**. 
- System Administrators have `user:manage` and `organization:manage`.
- They lack `screening:read_scoped`, `clinical:review`, and `screening:create`.
- Attempts by an admin to query patient clinical endpoints are blocked with `HTTP 403 Forbidden`.

### 4.3 Multi-Organization Facility Isolation
Clinical personnel (CHWs, Providers, Coordinators) are tied to a specific `organization_id`. The function `enforce_organization_boundary(user, target_facility_id)` verifies that the target facility matches the user's facility. Cross-organizational access without explicit authorization raises `HTTP 403 Forbidden`.
