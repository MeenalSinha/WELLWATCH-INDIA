# HEALTH DATA EXCHANGE GOVERNANCE & SOVEREIGNTY
## Phase 14 Data Residency & Consent Enforcement

**Project:** WELLWATCH PREVENT — Life-Course Preventive Care Intelligence  
**Target Market:** Abu Dhabi, United Arab Emirates  
**Statutory Authority:** UAE Health Data Law (Federal Law No. 2 of 2019), ADHICS Standard  

---

## 1. UAE Data Residency & In-Country Processing

In strict adherence to Article 13 of the UAE Health Data Law:
- All patient health information (PHI) must be stored and processed **within the geographic boundaries of the United Arab Emirates**.
- Cloud infrastructure hosting integration endpoints must maintain UAE local data center residency (e.g. Microsoft Azure UAE North / Central, AWS UAE Region).
- Direct egress of unmasked health data outside UAE borders without explicit ministerial authorization is strictly prohibited.

---

## 2. Dynamic Consent Enforcement at Boundaries

Before transmitting any patient data via external FHIR export:
1. The Interoperability Service verifies that the patient has an active `ConsentRecord` granting authorization for the requested purpose (`purpose = 'PREVENTIVE_SCREENING'` or `'HEALTH_CARE_OPERATION'`).
2. If consent has been revoked or expired, the export operation is **DENIED** with HTTP 403.
3. Every export and ingestion event records a SHA-256 payload hash in `provenance_records` and logs an entry in `audit_logs`.
