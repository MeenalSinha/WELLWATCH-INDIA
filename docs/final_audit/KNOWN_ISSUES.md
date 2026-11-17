# Known Limitations, Constraints & Deferred Technical Items

**Project:** WELLWATCH PREVENT — Life-Course Preventive Care Intelligence  
**Target Market:** Abu Dhabi, United Arab Emirates  
**Status:** FULLY DOCUMENTED & AUDITED

---

## 1. Resolved Limitations & Technical Debt

### Resolved in Phases 0–8 Audit:
- **Hardcoded Clinical Thresholds:** Extracted in-code numbers for hypertension, glucose, and BMI into `configs/clinical_rules/clinical_risk_indicators.json`.
- **Static Actionable Genes:** Extracted hardcoded genomic dictionaries into `configs/genomics/actionable_guidelines.json`.
- **CI Dependency & Build Breakages:** Resolved `email-validator` requirement in Pydantic, corrected TypeScript `bool` to `boolean` in `api.ts`, added missing `Tuple` typing import in `data_quality_service.py`, and verified git tracking for frontend libraries.
- **Frontend Gaps:** Implemented missing production pages for Patient Directory, Life-Course Chart, Care Pathway Browser, and Explainable Risk Assessments.

### Resolved in Phases 9–11 Audit:
- **Mock Provider Deliveries & Fake UUIDs:** Completely removed synthetic `DELIVERED` responses and fake UUID generators for SMS, WhatsApp, and Email. The engine now reports `NOT_CONFIGURED` honestly when credentials are unset.
- **In-App Messaging Persistence:** Created database-backed `in_app_notifications` table (Migration 006) and added inbox retrieval and read-state management endpoints.
- **Default Affirmative Consent Anti-Pattern:** Eradicated automatic opt-in. Default preferences require explicit opt-in (`consent_opt_in=False`, `OutreachAuthorizationStatus.NOT_PROVIDED`).
- **Unverified Care-Gap Closure:** Enforced database lookup for closing evidence (`ScreeningEncounter`, `Encounter`, `ClinicalObservation`, `Referral`) and eliminated clinical attestation privileges for `SYSTEM_ADMIN`.
- **Heuristic Recommendation Triggers:** Coupled recommendations directly to Phase 6 pathway care gap evaluations (`PathwayEngine`) and Phase 7 assessments.
- **Operational Policy Decentralization:** Created `configs/outreach_policy.json` (quiet hours, duplicate suppression, retry limits) and `configs/navigation_policy.json` (deadlines for missed appointments and rejected referrals).

---

## 2. Documented Operational Constraints

### 1. Database Engine Compatibility
- **Local Dev / Unit Tests:** Uses SQLite 3 for fast, ephemeral test execution.
- **Production Requirement:** PostgreSQL 15+ is required for production deployments to support concurrent row-level locking, JSONB querying, and enterprise data encryption at rest.
- **Guardrail:** The application automatically halts boot if configured for `ENVIRONMENT=production` while pointing to an SQLite database.

### 2. Malisart / Legacy Synthetic Dataset Quarantine
- The legacy `data/synthetic_patients.csv` artifact from the initial hackathon prototype contains uncalibrated synthetic data.
- **Constraint:** The model registry permanently quarantines models referencing this dataset (`LEGACY-XGBOOST-SYNTHETIC-V1`), forbidding promotion to clinical inference status. Real clinical trials or validated de-identified cohorts from DoH / Malaffi are required for production model training.

### 3. Asynchronous Malaffi HIE Integration
- Current architecture supports direct EHR / Laboratory ingestion through standardized REST endpoints and LOINC observation ingestion.
- Direct HL7 FHIR R4 connectivity with the Malaffi Health Information Exchange (HIE) requires mutual TLS (mTLS) certificate provisioning and secure VPN tunnel establishment with the DoH Abu Dhabi gateway. The API models have been designed around FHIR resources (`Observation`, `Patient`, `Condition`) to ensure seamless adapter development.

### 4. External Provider API Credentials
- Dispatches over SMS, WhatsApp, and Email require active credentials (`SMS_GATEWAY_URL`, `WHATSAPP_API_TOKEN`, `SMTP_HOST`).
- When credentials are not provisioned, the platform cleanly falls back to In-App portal notifications and honestly marks external dispatches as `NOT_CONFIGURED`.
