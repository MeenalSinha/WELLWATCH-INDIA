# EMR & DIAGNOSTIC LAB INTEGRATIONS SPECIFICATION
## Phase 14 Health Systems Data Feeds

**Project:** WELLWATCH PREVENT — Life-Course Preventive Care Intelligence  
**Target Market:** Abu Dhabi, United Arab Emirates  

---

## 1. SEHA EMR Integration

- **Target System:** Abu Dhabi Health Services Company (SEHA) Epic Electronic Medical Record system.
- **Protocol:** SMART-on-FHIR / OAuth2 authenticated REST APIs.
- **Data Flow:**
  - Ingestion: Primary care encounters, outpatient diagnostic notes, discharge summaries.
  - Care Gap Resolution: Hospital-confirmed screenings automatically satisfy active recall tasks.
- **Production Status:** Reported honestly as `NOT_CONFIGURED` awaiting client secret provisioning (`SEHA_CLIENT_SECRET`).

---

## 2. National Diagnostic Laboratory Network

- **Target Systems:** National Reference Laboratory (NRL), Pure Health, and hospital outpatient labs.
- **Protocol:** FHIR R4 DiagnosticReport / Observation messages via HTTPS REST or secure SFTP.
- **Data Flow:**
  - Blood Glucose (Fasting), HbA1c, Lipid Profiles, Kidney/Liver Panels.
  - Identity matching by Emirates ID (`PersonIdentifier.namespace == 'EMIRATES_ID'`).
  - Cryptographic provenance tracking in `ProvenanceRecord`.
- **Production Status:** Reported honestly as `NOT_CONFIGURED` awaiting lab portal API key provisioning (`LAB_PORTAL_API_KEY`).
