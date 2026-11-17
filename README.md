# WELLWATCH PREVENT
### Life-Course Preventive Care Intelligence
**Tagline: From Health Signals to Preventive Action.**

[![CI Pipeline](https://github.com/wellwatch-prevent/wellwatch-prevent/actions/workflows/ci.yml/badge.svg)](https://github.com/wellwatch-prevent/wellwatch-prevent/actions)
[![Python Version](https://img.shields.io/badge/python-3.11%20%7C%203.14-blue.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115+-009688.svg)](https://fastapi.tiangolo.com)
[![Next.js](https://img.shields.io/badge/Next.js-14-black.svg)](https://nextjs.org/)
[![PostgreSQL](https://img.shields.io/badge/PostgreSQL-16+-336791.svg)](https://www.postgresql.org/)
[![Compliance](https://img.shields.io/badge/Compliance-ADHICS%20v2%20%7C%20UAE%20Health%20Data%20Law-success.svg)](https://www.doh.gov.ae)

---

## 1. Overview & Competition Context

- **Competition:** Future Health Digital Innovation Challenge 2026
- **Focus Area:** Population Health
- **Challenge Statement:** Precision Prevention and Screening
- **Target Market:** Abu Dhabi, United Arab Emirates
- **Jurisdictional & Regulatory Framework:** 
  - Department of Health Abu Dhabi (DoH) & Abu Dhabi Public Health Centre (ADPHC)
  - Abu Dhabi Healthcare Information and Cyber Security Standard (ADHICS v2)
  - UAE Federal Law No. 2 of 2019 (Health Data Law)
  - UAE Federal Decree-Law No. 45 of 2021 (Personal Data Protection Law - PDPL)
  - UAE Federal Law No. 3 of 2016 (Child Rights Law - Wadeema's Law)
  - Malaffi Health Information Exchange (HIE) Standards

**WELLWATCH PREVENT** is an enterprise population health intelligence and preventive-care navigation platform. It moves healthcare beyond generic age- and gender-based screening by combining longitudinal health intelligence, clinical risk evaluation, care-gap detection, explainable AI, and closed-loop preventive care navigation.

---

## 2. Five Core Product Systems

The long-term vision of WELLWATCH PREVENT encompasses five interconnected systems:

```
+----------------------------------------------------------------------------------------------------+
|                                    WELLWATCH PREVENT PLATFORM                                      |
|                                                                                                    |
|   +--------------------------+  +--------------------------+  +--------------------------------+   |
|   |   A. Individual Health   |  |   B. Preventive Risk     |  |     C. Preventive Pathway      |   |
|   |       Intelligence       |  |       Intelligence       |  |            Engine              |   |
|   |  - Longitudinal Profile  |  |  - Risk Detection        |  |  - Clinical Eligibility       |   |
|   |  - Data Provenance       |  |  - Epistemic Uncertainty |  |  - Well-Child / IFHAS Sched.  |   |
|   |  - 8-State Confidence    |  |  - Explainable Insights  |  |  - Interval Optimization      |   |
|   +--------------------------+  +--------------------------+  +--------------------------------+   |
|                                                                                                    |
|   +--------------------------------------------------------+  +--------------------------------+   |
|   |              D. Care Navigation Engine                 |  |      E. Population Health      |   |
|   |  - Outreach Coordination & Appointment Scheduling      |  |          Intelligence          |   |
|   |  - Referral Pathways & Multi-Channel Engagement        |  |  - Coverage & Equity Analytics |   |
|   |  - Closed-Loop Recall & Follow-Up Workflows            |  |  - Population Risk Strata      |   |
|   +--------------------------------------------------------+  +--------------------------------+   |
+----------------------------------------------------------------------------------------------------+
```

---

## 3. Implementation Status & Scope Verification

This repository reflects the successful completion and empirical verification of **Phases 0 through 3**. Future phases (Phases 4 through 17) represent the multi-year production roadmap and are deliberately deferred to ensure rigorous engineering quality.

| Phase | Phase Name | Status | Summary of Verified Deliverables |
| :---: | :--- | :---: | :--- |
| **Phase 0** | **Existing Project Audit, Recovery & Baseline** | **VERIFIED** | Audited all 33 legacy files; cataloged 14 defects (`DEF-01` to `DEF-14`); isolated synthetic training data; verified legacy XGBoost model pipeline; authored 10 audit documents in `docs/audit/`. |
| **Phase 1** | **Product Definition & Clinical Requirements** | **VERIFIED** | Defined 5 core systems, 8 user personas, 12 life-course journeys, pediatric (0–6y) and adult IFHAS schedules, 8-state epistemic data uncertainty model, and regulatory register; authored 13 PRD documents in `docs/product/`. |
| **Phase 2** | **Application Foundation & Production Architecture** | **VERIFIED** | Implemented modular FastAPI backend, dual SQLite/PostgreSQL engine, SQLAlchemy 2.0 ORM schemas, Alembic migrations (`001_initial_schema`), Next.js 14 TypeScript frontend, Docker Compose, and GitHub Actions CI. |
| **Phase 3** | **Authentication, Authorization & Privacy Foundation** | **VERIFIED** | Implemented 12-round bcrypt hashing, JWT access/refresh lifecycle, 5-attempt account lockout defense, 8-role RBAC with admin separation of duties, Wadeema guardianship validation, consent tracking with revocation, SHA-256 chained audit logs, and security headers. **30/30 backend tests passing.** |
| **Phase 4** | **Longitudinal Clinical Database** | *Deferred* | Planned: Dynamic EAV observation streaming, LOINC/SNOMED CT mapping, multi-unit lab normalization. |
| **Phase 5** | **Preventive Risk Intelligence Engine** | *Deferred* | Planned: TreeSHAP explainability, retrained calibrated models, epistemic abstention rules. |
| **Phase 6** | **Malaffi & External Ingestion Adapters** | *Deferred* | Planned: HL7 FHIR v4 ingestion pipeline, UAE Pass identity provider integration. |
| **Phases 7–17** | **Pathways, Navigation & Population Analytics** | *Deferred* | Planned: Clinical pathway rules engine, care navigation, mobile apps, population analytics. |

---

## 4. Legacy Project Findings & Transparency

The original repository (`WELLWATCH-INDIA`) was an exploratory hackathon prototype. Our audit revealed critical technical findings that necessitated a clean-slate architectural foundation:

1. **Monolithic UI Disconnect:** In `streamlit_app.py`, a function `load_model_artifacts()` was defined but **never called**. All patient risk assessments in the UI invoked `predict_risk_simple()`, which returned static heuristic float values (`0.75, 0.20, 0.05`).
2. **Cosmetic Stubs:** UI action buttons ("Save to Database", "Send to Patient", "Generate PDF Report", "Sync Data", "Backup") were purely cosmetic `st.success(...)` banners that did not persist data or communicate with external systems.
3. **Data Preprocessing Leakage:** In the legacy notebook (`WELLWATCH_INDIA.ipynb`, Cell 35), `StandardScaler` fitting and median imputation were executed on the entire 1,982-row dataset **before** splitting into train and test sets.
4. **Synthetic Target Generation:** The target column (`risk_level`) in `raw_screening_data.csv` was generated using a formulaic weighted linear score with Gaussian noise. It is 100% synthetic and cannot be used to substantiate real clinical performance.
5. **Legacy Asset Preservation:** In compliance with preservation guidelines, all original ML artifacts, data, and Streamlit scripts are preserved intact as an unlinked legacy baseline in `data/`, `models/`, and `streamlit_app.py`.

---

## 5. Non-Negotiable Engineering Standards

- **Zero Hardcoded Secrets & Configuration:** All secrets, token lifetimes, database URLs, and environment flags are strictly loaded via Pydantic `BaseSettings` from environment variables (`.env`).
- **No Fake Functionality:** Every implemented endpoint, form, and button is backed by verifiable code and automated tests. Features not yet implemented are explicitly omitted or marked as deferred.
- **Strict Synthetic Data Segregation:** Synthetic datasets are quarantined with explicit provenance metadata. They are never silently merged with production database tables or used to claim clinical validity.
- **No Fabricated Metrics:** Dashboards and analytics query actual database counts or return structured empty states.
- **Clinical Decision Support Boundary:** The platform provides decision support and preventive navigation; it does not issue autonomous medical diagnoses or prescribe clinical treatments.

---

## 6. Technology Stack

- **Backend:** FastAPI 0.115+, Pydantic v2, Python 3.11 / 3.14
- **Database:** PostgreSQL 16+ (Production) / SQLite 3 (Local Development)
- **ORM & Migrations:** SQLAlchemy 2.0 (Declarative Mappings), Alembic
- **Security & Crypto:** Passlib (Bcrypt 12 rounds), PyJWT (HS256 tokens), Hashlib (SHA-256 chained audit logs)
- **Frontend:** Next.js 14, React 18, TypeScript, Tailwind CSS, Lucide Icons
- **Containerization & CI:** Docker, Docker Compose, GitHub Actions

---

## 7. Getting Started & Local Development

### 7.1 Prerequisites
- **Python:** 3.11+ (Tested on Python 3.11 and 3.14)
- **Node.js:** 18.0+ and npm 9.0+
- **Docker & Docker Compose:** Optional, for containerized multi-service execution.

### 7.2 Backend Setup & Execution

1. **Clone and Enter Workspace:**
   ```bash
   cd WELLWATCH-INDIA
   ```

2. **Configure Environment:**
   ```bash
   cp .env.example .env
   # Edit .env if configuring custom secrets or PostgreSQL credentials
   ```

3. **Install Backend Dependencies:**
   ```bash
   pip install -r backend/requirements.txt
   ```

4. **Execute Database Migrations:**
   ```bash
   cd backend
   python -m alembic upgrade head
   cd ..
   ```

5. **Run Automated Test Suite (30 Tests):**
   ```bash
   python -m pytest backend/tests/ -v
   ```

6. **Start FastAPI Backend Server:**
   ```bash
   python -m uvicorn backend.app.main:app --host 127.0.0.1 --port 8000 --reload
   ```
   - **Interactive API Documentation (Swagger):** [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)
   - **Alternative API Docs (ReDoc):** [http://127.0.0.1:8000/redoc](http://127.0.0.1:8000/redoc)
   - **Health Liveness Check:** [http://127.0.0.1:8000/api/v1/health](http://127.0.0.1:8000/api/v1/health)
   - **Database Readiness Check:** [http://127.0.0.1:8000/api/v1/health/ready](http://127.0.0.1:8000/api/v1/health/ready)

### 7.3 Frontend Setup & Execution

1. **Navigate to Frontend Directory:**
   ```bash
   cd frontend
   ```

2. **Install Node Packages:**
   ```bash
   npm install
   ```

3. **Start Development Server:**
   ```bash
   npm run dev
   ```
   - Access web application at: [http://localhost:3000](http://localhost:3000)

### 7.4 Running via Docker Compose

To launch the complete multi-tier production environment (PostgreSQL 16, FastAPI backend, and Next.js frontend):

```bash
docker-compose up --build
```
- **Frontend:** [http://localhost:3000](http://localhost:3000)
- **Backend API:** [http://localhost:8000](http://localhost:8000)
- **Database:** `localhost:5432`

---

## 8. Automated Test Suite

The test suite validates authentication, token lifecycles, brute-force lockout, 8-role RBAC permissions, organization boundary isolation, guardianship verification, purpose-based consent, tamper-evident audit logging, and clinical vitals bounds checking:

```bash
python -m pytest backend/tests/ -v
```

**Results:**
```
============================= test session starts =============================
platform win32 -- Python 3.14.2, pytest-8.2.2, pluggy-1.6.0
collected 30 items

backend/tests/test_audit.py::test_audit_logging_and_sanitization PASSED        [  3%]
backend/tests/test_auth.py::test_successful_authentication PASSED              [  6%]
backend/tests/test_auth.py::test_invalid_password_raises_error PASSED          [ 10%]
backend/tests/test_auth.py::test_account_lockout_after_5_failed_attempts PASSED [ 13%]
backend/tests/test_auth.py::test_deactivated_user_cannot_authenticate PASSED   [ 16%]
backend/tests/test_auth.py::test_refresh_token_validation PASSED               [ 20%]
backend/tests/test_config.py::test_settings_load_successfully PASSED           [ 23%]
backend/tests/test_config.py::test_short_secret_key_rejected PASSED            [ 26%]
backend/tests/test_consent.py::test_consent_lifecycle PASSED                   [ 30%]
backend/tests/test_consent.py::test_unauthorized_user_cannot_revoke_others_consent PASSED [ 33%]
backend/tests/test_database.py::test_database_readiness PASSED                 [ 36%]
backend/tests/test_database.py::test_unique_constraint_enforcement PASSED       [ 40%]
backend/tests/test_guardian.py::test_guardianship_verification_workflow PASSED [ 43%]
backend/tests/test_guardian.py::test_invalid_dependent_linkage_rejected PASSED [ 46%]
backend/tests/test_health.py::test_health_endpoint PASSED                      [ 50%]
backend/tests/test_health.py::test_readiness_endpoint PASSED                   [ 53%]
backend/tests/test_organization.py::test_cross_organization_boundary_blocked PASSED [ 56%]
backend/tests/test_organization.py::test_same_organization_allowed PASSED     [ 60%]
backend/tests/test_organization.py::test_cross_organization_operational_roles_permitted PASSED [ 63%]
backend/tests/test_rbac.py::test_all_8_roles_have_defined_permissions PASSED   [ 66%]
backend/tests/test_rbac.py::test_individual_role_permissions PASSED            [ 70%]
backend/tests/test_rbac.py::test_chw_role_permissions PASSED                   [ 73%]
backend/tests/test_healthcare_provider_role_permissions PASSED                 [ 76%]
backend/tests/test_system_admin_separation_of_duties PASSED                   [ 80%]
backend/tests/test_data_governance_officer_permissions PASSED                  [ 83%]
backend/tests/test_screening.py::test_valid_screening_encounter_and_bmi_calculation PASSED [ 86%]
backend/tests/test_screening.py::test_invalid_blood_pressure_rejected PASSED   [ 90%]
backend/tests/test_screening.py::test_out_of_bounds_blood_pressure_rejected PASSED [ 93%]
backend/tests/test_security_headers.py::test_security_headers_middleware_direct PASSED [ 96%]
backend/tests/test_security_headers.py::test_correlation_id_middleware_direct PASSED [100%]

====================== 30 passed, 1543 warnings in 4.20s ======================
```

---

## 9. Comprehensive Documentation Index

All technical, clinical, and architectural documentation is maintained within the `docs/` directory:

### Final Audit Report
- [Phases 0 to 3 Final Verification and Audit Report](file:///c:/Users/Meenal%20Sinha/Downloads/WELLWATCH-INDIA/docs/PHASE_0_TO_3_FINAL_AUDIT.md)

### Phase 0: Audit & Baseline Reports (`docs/audit/`)
- [Project Audit](file:///c:/Users/Meenal%20Sinha/Downloads/WELLWATCH-INDIA/docs/audit/PROJECT_AUDIT.md)
- [Feature Implementation Matrix](file:///c:/Users/Meenal%20Sinha/Downloads/WELLWATCH-INDIA/docs/audit/FEATURE_IMPLEMENTATION_MATRIX.md)
- [ML Model Audit](file:///c:/Users/Meenal%20Sinha/Downloads/WELLWATCH-INDIA/docs/audit/ML_MODEL_AUDIT.md)
- [Data Provenance Audit](file:///c:/Users/Meenal%20Sinha/Downloads/WELLWATCH-INDIA/docs/audit/DATA_PROVENANCE_AUDIT.md)
- [Database Audit](file:///c:/Users/Meenal%20Sinha/Downloads/WELLWATCH-INDIA/docs/audit/DATABASE_AUDIT.md)
- [Security Audit](file:///c:/Users/Meenal%20Sinha/Downloads/WELLWATCH-INDIA/docs/audit/SECURITY_AUDIT.md)
- [Known Issues & Defect Catalog](file:///c:/Users/Meenal%20Sinha/Downloads/WELLWATCH-INDIA/docs/audit/KNOWN_ISSUES.md)
- [Baseline Test Report](file:///c:/Users/Meenal%20Sinha/Downloads/WELLWATCH-INDIA/docs/audit/BASELINE_TEST_REPORT.md)
- [Migration Plan](file:///c:/Users/Meenal%20Sinha/Downloads/WELLWATCH-INDIA/docs/audit/MIGRATION_PLAN.md)
- [Phase 0 Completion Report](file:///c:/Users/Meenal%20Sinha/Downloads/WELLWATCH-INDIA/docs/audit/PHASE_0_COMPLETION_REPORT.md)

### Phase 1: Product Definition & Clinical Governance (`docs/product/`)
- [Product Requirements Document (PRD)](file:///c:/Users/Meenal%20Sinha/Downloads/WELLWATCH-INDIA/docs/product/PRODUCT_REQUIREMENTS_DOCUMENT.md)
- [Product Vision](file:///c:/Users/Meenal%20Sinha/Downloads/WELLWATCH-INDIA/docs/product/PRODUCT_VISION.md)
- [User Personas](file:///c:/Users/Meenal%20Sinha/Downloads/WELLWATCH-INDIA/docs/product/USER_PERSONAS.md)
- [User Journeys](file:///c:/Users/Meenal%20Sinha/Downloads/WELLWATCH-INDIA/docs/product/USER_JOURNEYS.md)
- [Clinical Requirements](file:///c:/Users/Meenal%20Sinha/Downloads/WELLWATCH-INDIA/docs/product/CLINICAL_REQUIREMENTS.md)
- [Preventive Pathway Requirements](file:///c:/Users/Meenal%20Sinha/Downloads/WELLWATCH-INDIA/docs/product/PREVENTIVE_PATHWAY_REQUIREMENTS.md)
- [Data Requirements & Epistemic Model](file:///c:/Users/Meenal%20Sinha/Downloads/WELLWATCH-INDIA/docs/product/DATA_REQUIREMENTS.md)
- [Data Dictionary](file:///c:/Users/Meenal%20Sinha/Downloads/WELLWATCH-INDIA/docs/product/DATA_DICTIONARY.md)
- [Responsible AI Requirements](file:///c:/Users/Meenal%20Sinha/Downloads/WELLWATCH-INDIA/docs/product/RESPONSIBLE_AI_REQUIREMENTS.md)
- [Regulatory Requirements Register](file:///c:/Users/Meenal%20Sinha/Downloads/WELLWATCH-INDIA/docs/product/REGULATORY_REQUIREMENTS_REGISTER.md)
- [Assumptions and Dependencies](file:///c:/Users/Meenal%20Sinha/Downloads/WELLWATCH-INDIA/docs/product/ASSUMPTIONS_AND_DEPENDENCIES.md)
- [Requirements Traceability Matrix](file:///c:/Users/Meenal%20Sinha/Downloads/WELLWATCH-INDIA/docs/product/REQUIREMENTS_TRACEABILITY_MATRIX.md)
- [Phase 1 Completion Report](file:///c:/Users/Meenal%20Sinha/Downloads/WELLWATCH-INDIA/docs/product/PHASE_1_COMPLETION_REPORT.md)

### Phase 2: Architecture & Foundation (`docs/architecture/` & `docs/`)
- [Architecture Decisions Record (ADR)](file:///c:/Users/Meenal%20Sinha/Downloads/WELLWATCH-INDIA/docs/architecture/ARCHITECTURE_DECISIONS.md)
- [System Architecture Specification](file:///c:/Users/Meenal%20Sinha/Downloads/WELLWATCH-INDIA/docs/architecture/SYSTEM_ARCHITECTURE.md)
- [Database Infrastructure Specification](file:///c:/Users/Meenal%20Sinha/Downloads/WELLWATCH-INDIA/docs/architecture/DATABASE_INFRASTRUCTURE.md)
- [Phase 2 Completion Report](file:///c:/Users/Meenal%20Sinha/Downloads/WELLWATCH-INDIA/docs/PHASE_2_COMPLETION_REPORT.md)

### Phase 3: Security & Privacy Architecture (`docs/security/` & `docs/`)
- [Authentication Architecture](file:///c:/Users/Meenal%20Sinha/Downloads/WELLWATCH-INDIA/docs/security/AUTHENTICATION_ARCHITECTURE.md)
- [RBAC Permission Matrix](file:///c:/Users/Meenal%20Sinha/Downloads/WELLWATCH-INDIA/docs/security/RBAC_PERMISSION_MATRIX.md)
- [Consent and Authorization Specification](file:///c:/Users/Meenal%20Sinha/Downloads/WELLWATCH-INDIA/docs/security/CONSENT_AND_AUTHORIZATION.md)
- [Audit Logging Specification](file:///c:/Users/Meenal%20Sinha/Downloads/WELLWATCH-INDIA/docs/security/AUDIT_LOGGING_SPECIFICATION.md)
- [Privacy and Data Protection Specification](file:///c:/Users/Meenal%20Sinha/Downloads/WELLWATCH-INDIA/docs/security/PRIVACY_AND_DATA_PROTECTION.md)
- [Security Threat Model](file:///c:/Users/Meenal%20Sinha/Downloads/WELLWATCH-INDIA/docs/security/SECURITY_THREAT_MODEL.md)
- [Phase 3 Completion Report](file:///c:/Users/Meenal%20Sinha/Downloads/WELLWATCH-INDIA/docs/PHASE_3_COMPLETION_REPORT.md)

### Verification Tracking Matrices
- [Project Progress Tracker](file:///c:/Users/Meenal%20Sinha/Downloads/WELLWATCH-INDIA/docs/PROJECT_PROGRESS.md)
- [Implementation Evidence Matrix](file:///c:/Users/Meenal%20Sinha/Downloads/WELLWATCH-INDIA/docs/IMPLEMENTATION_EVIDENCE_MATRIX.md)

---

## 10. Governance & Legal Disclaimer

WELLWATCH PREVENT is developed strictly as a clinical decision support and preventive-care navigation platform. It does not provide medical diagnoses or prescribe medical treatment. All clinical care-gap alerts, risk stratifications, and screening recommendations are intended to guide and support qualified healthcare professionals and authorized public health administrators in accordance with official DoH Abu Dhabi and ADPHC guidelines.

---
**WELLWATCH PREVENT Engineering Team** | *Future Health Digital Innovation Challenge 2026*
