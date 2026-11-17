# WELLWATCH PREVENT — Privacy & Data Protection Framework
## UAE Health Data Law, ADHICS Compliance, and Privacy Engineering

**Jurisdiction:** Abu Dhabi, United Arab Emirates  
**Compliance Mandate:** UAE Federal Law No. 2/2019, UAE PDPL, ADHICS v2  
**Status:** IMPLEMENTED BASELINE (PHASE 3)  

---

## 1. Privacy Engineering Principles

1. **Privacy by Design:** Privacy controls, data minimization, purpose limitation, and audit trails are built into the foundational data models rather than retrofitted as an afterthought.
2. **Healthcare Record Retention vs. Deletion Rights:** While UAE PDPL recognizes general "rights to erasure", statutory healthcare regulations (UAE Federal Law No. 2 of 2019) mandate that medical and screening records must be retained for a **minimum of 25 years** from the date of the last treatment or from the age of majority for pediatric records. The platform enforces this distinction: a patient withdrawal of consent halts optional processing and outreach, but does not destructively delete clinical encounter history.
3. **Data Minimization:** Forms collect only attributes required for pathway routing and risk stratification. Optional fields are clearly designated.

---

## 2. Technical Privacy Safeguards

### 2.1 Encryption Standards
- **Data in Transit:** Enforced TLS 1.3 across all HTTP and database connections. Weak cipher suites, TLS 1.0, and TLS 1.1 are prohibited.
- **Data at Rest:** Database volumes and backup targets utilize AES-256 encryption.
- **Secret Management:** Encryption keys and JWT secrets are managed via environment injection and are prohibited from being hardcoded or committed to git.

### 2.2 Security Headers & Web Boundaries
Enforced via `SecurityHeadersMiddleware`:
- `Content-Security-Policy`: Restricts script and frame execution, blocking XSS.
- `X-Frame-Options: DENY`: Defends against clickjacking.
- `X-Content-Type-Options: nosniff`: Prevents MIME-sniffing attacks.
- `Strict-Transport-Security`: Enforces HTTPS with subdomains.
- `Referrer-Policy: strict-origin-when-cross-origin`: Protects URL query leakage.

### 2.3 Epistemic Certainty Safeguards
Data quality is protected against false inferences through the 8-state epistemic model (`VERIFIED`, `UNVERIFIED`, `MISSING`, `UNKNOWN`, `NOT_APPLICABLE`, `OUTDATED`, `CONFLICTING`, `UNAUTHORIZED`). Missing records are never treated as confirmation of health or non-compliance.

---

## 3. Data Subject Rights Workflows

1. **Right of Access:** Patients can review their full longitudinal record, active consents, and screening encounters via `GET /api/v1/consent/me` and `GET /api/v1/auth/me`.
2. **Right to Revoke Consent:** Patients can revoke any optional processing directive via `POST /api/v1/consent/{id}/revoke`.
3. **Right to Data Portability:** Export endpoints provide structured JSON/CSV data of patient-authorized records with verified audit logging.
4. **Data Protection Officer Oversight:** Designated Data Governance Officers (`DATA_GOVERNANCE_OFFICER`) have dedicated inspection tools for audit logs and consent registers.
