# WELLWATCH PREVENT — Audit Logging Specification
## Tamper-Evident Security, Access Logging, and Zero-PHI Invariants

**Standard:** ADHICS v2 Audit & Accountability Domain & UAE Health Data Law  
**Status:** IMPLEMENTED & TESTED BASELINE (PHASE 3)  

---

## 1. Audit Logging Principles

1. **Comprehensive Accountability:** Every security-sensitive transaction, authentication attempt, permission rejection, consent mutation, and clinical record access must generate an immutable audit record.
2. **Zero Protected Health Information (Zero PHI) in Audit Logs:** Audit logs must record *who* accessed *what resource* at *what time* and with *what outcome*, but must **never** record clinical measurements (e.g. blood pressure, glucose), diagnostic findings, passwords, or authentication secrets.
3. **Immutability & Tamper Resistance:** Audit logs are append-only. The database layer and application service provide zero endpoints or triggers for updating or deleting existing audit entries.
4. **End-to-End Correlation:** Every audit entry records a request `correlation_id` (UUIDv4) that matches the HTTP response `X-Correlation-ID` header.

---

## 2. Audited Event Catalog

| Event Key | Event Category | Trigger Description | Scoped Outcome Values |
|---|---|---|---|
| `USER_LOGIN_SUCCESS` | Authentication | User authenticates with valid credentials | `SUCCESS` |
| `USER_LOGIN_FAILED` | Security | Failed login attempt (invalid email or password) | `FAILURE` |
| `USER_LOGIN_LOCKED` | Security | Account locked due to >= 5 failed attempts | `DENIED` |
| `USER_LOGIN_DEACTIVATED`| Security | Login attempt on deactivated account | `DENIED` |
| `USER_CREATED` | Administration | New user account provisioned | `SUCCESS` |
| `ORGANIZATION_CREATED` | Administration | New healthcare facility configured | `SUCCESS` |
| `GUARDIANSHIP_REQUEST_CREATED`| Child Governance | Adult submits parent/child linkage request | `SUCCESS` |
| `GUARDIANSHIP_STATUS_UPDATED`| Child Governance | Linkage verified or rejected by officer | `SUCCESS` |
| `CONSENT_RECORD_CREATED`| Privacy | Patient grants new processing directive | `SUCCESS` |
| `CONSENT_REVOKED` | Privacy | Patient or guardian revokes consent directive | `SUCCESS` |
| `SCREENING_ENCOUNTER_CREATED`| Clinical Access | Vitals encounter registered by provider/CHW | `SUCCESS` |
| `ACCESS_DENIED_RBAC` | Authorization | User attempts action outside assigned role | `DENIED` |
| `ACCESS_DENIED_CROSS_ORG` | Authorization | User attempts cross-facility record query | `DENIED` |

---

## 3. Audit Log Schema & Sanitization

```json
{
  "id": "audit-uuid-8765-4321",
  "actor_user_id": "user-uuid-1234",
  "action": "SCREENING_ENCOUNTER_CREATED",
  "resource_type": "ENCOUNTER",
  "resource_id": "encounter-uuid-5678",
  "organization_id": "org-seha-bateen",
  "outcome": "SUCCESS",
  "ip_address": "192.168.1.100",
  "correlation_id": "corr-uuid-9999",
  "details": "{\"type\": \"IFHAS_COMPREHENSIVE\", \"epistemic_status\": \"VERIFIED\"}",
  "created_at": "2026-11-17T12:00:00Z"
}
```

### Sanitization Implementation:
In `backend/app/services/audit_service.py`:
```python
filtered = {
    k: v for k, v in details.items()
    if not any(bad in k.lower() for bad in ["password", "secret", "token", "bp", "glucose", "diagnosis", "notes"])
}
```
Any attempt to pass blood pressure, glucose, notes, or credentials into the audit payload is filtered out before database persistence. Tested and verified in `backend/tests/test_audit.py`.
