# Known Limitations, Constraints & Deferred Technical Items
**Project:** WELLWATCH PREVENT — Life-Course Preventive Care Intelligence  
**Target Market:** Abu Dhabi, United Arab Emirates  

---

## 1. Resolved Limitations & Technical Debt
During the Phase 4–8 audit and stabilization, the following technical issues were identified and permanently resolved:
- **Hardcoded Thresholds:** Extracted in-code numbers for hypertension, glucose, and BMI into `configs/clinical_rules/clinical_risk_indicators.json`.
- **Static Actionable Genes:** Extracted hardcoded genomic dictionaries into `configs/genomics/actionable_guidelines.json`.
- **CI Dependency & Build Breakages:** Resolved `email-validator` requirement in Pydantic, corrected TypeScript `bool` to `boolean` in `api.ts`, added missing `Tuple` typing import in `data_quality_service.py`, and verified git tracking for frontend libraries.
- **Frontend Gaps:** Implemented missing production pages for Patient Directory, Life-Course Chart, Care Pathway Browser, and Explainable Risk Assessments.

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

### 4. Upstream Python 3.14 Warnings
- The environment runs Python 3.14.2 which issues deprecation warnings for `asyncio.iscoroutinefunction` in third-party dependencies (`pytest_asyncio`) and `datetime.utcnow()` in SQLAlchemy. These warnings do not affect platform correctness and will resolve naturally as upstream packages release Python 3.14 wheels.
