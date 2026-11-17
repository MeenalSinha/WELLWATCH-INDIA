# Security & Governance Audit Report
**Project:** WELLWATCH PREVENT — Life-Course Preventive Care Intelligence  
**Compliance Standards:** Abu Dhabi Healthcare Information and Cyber Security (ADHICS v2), UAE Federal Decree-Law No. 45 of 2021 (Personal Data Protection)  

---

## 1. Authentication & Session Management
- **Password Hashing:** Argon2id with 64MB memory cost, 3 iterations, and 4 degrees of parallelism (`backend/app/core/security.py`).
- **Token Security:** Asymmetric RS256 / HS256 JWT tokens with short expiry (15 minutes for access tokens, 7 days for rotation refresh tokens).
- **Account Lockout:** Account lockout triggered after 5 consecutive failed authentication attempts (`backend/app/services/auth_service.py`).
- **Production Guardrails:** Application startup safely aborts if `ENVIRONMENT=production` and default test secret keys or SQLite database engines are detected (`backend/app/core/config.py`).

---

## 2. Role-Based Access Control (RBAC) & Separation of Duties
All 8 user personas have defined granular permissions in `backend/app/security/permissions.py`:
1. `INDIVIDUAL`: View own records and manage own consents.
2. `PARENT_GUARDIAN`: View dependent pediatric records under verified legal guardianship proxy.
3. `COMMUNITY_HEALTH_WORKER`: Record vital signs and screening encounters. Blocked from clinical review and genomic data.
4. `HEALTHCARE_PROVIDER`: Full clinical viewing, ordering, assessment generation, and human review sign-off.
5. `POPULATION_HEALTH_ADMIN`: Aggregated cohort analytics and pathway compliance. Blocked from individual unmasked PHI.
6. `DATA_GOVERNANCE_OFFICER`: Audit trail oversight and consent compliance.
7. `SYSTEM_ADMINISTRATOR`: Technical infrastructure and user provisioning. **Explicitly blocked from patient PHI and clinical records.**
8. `RESEARCHER`: Fully de-identified cohort queries requiring active governance approval.

---

## 3. Data Privacy & Sovereignty
- **Facility Boundary Isolation:** All clinical patient queries strictly enforce organizational tenant boundaries (`OrganizationBoundary`). Cross-facility patient record access is rejected with HTTP 403 Forbidden.
- **National Identifier Protection:** National IDs (Emirates ID) are cryptographically hashed using SHA-256 with an HMAC salt (`identifier_hash`) for deduplication without storing plaintext identifiers in query logs.
- **Audit Trail Sanitization:** All actions are captured in `audit_logs` with correlation IDs. Sensitive fields (passwords, tokens, blood pressures, diagnoses) are automatically stripped prior to persistence.
- **HTTP Security Headers:** Implemented via middleware (`backend/app/core/security_headers.py`), enforcing `Strict-Transport-Security`, `X-Content-Type-Options: nosniff`, `X-Frame-Options: DENY`, `Referrer-Policy: strict-origin-when-cross-origin`, and `Content-Security-Policy`.
