# WELLWATCH PREVENT — Prioritized Defect Register
## Inventory of Known Defects, Architectural Limitations, and Technical Debt

**Repository:** `WELLWATCH-INDIA`  
**Audit Date:** 2026-11-17  
**Classification Standard:** Software Quality, Clinical Safety & Security  

---

## 1. Defect Severity Definitions
- **Critical (P0):** Severe defect causing patient safety risks, security breaches, data loss, or total feature failure.
- **Major (P1):** Core functional defect where a feature does not perform its advertised function, uses fake data, or has significant architectural flaws.
- **Moderate (P2):** Defect causing performance degradation, missing edge-case validation, or degraded user experience.
- **Low (P3):** Minor styling, cosmetic, or documentation discrepancy.

---

## 2. Prioritized Defect Register

| Defect ID | Severity | Category | Component | Description & Code Location | Remediation Phase |
|---|---|---|---|---|---|
| **DEF-01** | **Critical (P0)** | ML / Safety | `streamlit_app.py:741` | **Model completely bypassed in screening:** Form submission calls heuristic `predict_risk_simple()` rather than trained XGBoost model. | Phase 0/2 |
| **DEF-02** | **Critical (P0)** | Data Integrity | `streamlit_app.py:860` | **Fake Database Save:** "Save to Database" button displays success banner without executing SQL INSERT. Patient data is discarded. | Phase 2 |
| **DEF-03** | **Critical (P0)** | Security | Whole Repo | **Zero Authentication / RBAC:** The application lacks all access controls, login mechanisms, and permission checks. | Phase 3 |
| **DEF-04** | **Major (P1)** | ML Methodology | `WELLWATCH_INDIA.ipynb:35,54`| **Data Leakage in Training:** Feature scaling (`StandardScaler`) and median imputation fitted on full dataset before train/test split. | Phase 0/4 |
| **DEF-05** | **Major (P1)** | Data / Integrity | `streamlit_app.py:870-885` | **Fabricated Community Dashboard:** Generates random numbers on every render using `np.random` instead of querying database. | Phase 2 |
| **DEF-06** | **Major (P1)** | Data / Integrity | `streamlit_app.py:1435-1450`| **Fabricated Admin Record Table:** Admin panel displays 20 randomly generated records rather than records from `wellwatch.db`. | Phase 2 |
| **DEF-07** | **Major (P1)** | Functionality | `streamlit_app.py:856` | **Fake Patient Alert Notification:** "Send to Patient" button is an empty stub with no telephony/SMS/email integration. | Phase 2 |
| **DEF-08** | **Major (P1)** | Functionality | `streamlit_app.py:852` | **Fake PDF Report Generation:** "Generate Report" button is an empty stub that does not invoke the notebook's PDF generator. | Phase 2 |
| **DEF-09** | **Major (P1)** | Infrastructure | `streamlit_app.py:1590-1625`| **Simulated Sync & Backup Operations:** "Sync Now" and "Create Backup" buttons use `time.sleep()` delays and return fake success alerts. | Phase 2 |
| **DEF-10** | **Moderate (P2)**| Architecture | `streamlit_app.py` | **Monolithic 1,906-Line File:** Entire app is contained in one file with 4 functions; HTML, SQL, and state management are tangled. | Phase 2 |
| **DEF-11** | **Moderate (P2)**| Data Quality | `cleaned_data.csv` | **Uncleaned Missing Values:** 700 missing values in `physical_activity` were left unhandled during data cleaning. | Phase 0 |
| **DEF-12** | **Moderate (P2)**| Database | `wellwatch.db` | **Missing Foreign Keys & Indexes:** Database lacks relational constraints, foreign keys, or indexing on key lookup fields. | Phase 2 |
| **DEF-13** | **Moderate (P2)**| Security | `requirements.txt` | **Unpinned Dependencies & Insecure Tunneling:** Packages lack version pins; `pyngrok` included in production requirements. | Phase 2 |
| **DEF-14** | **Low (P3)** | Documentation | `README.md:87` | **Inaccurate Sample Count Claims:** Claims "400 test samples" when actual stratified split contains 397 samples. | Phase 0 |

---

## 3. Resolution Strategy

- **Immediate Deprecation:** The fake actions and random data generators in `streamlit_app.py` must be retired.
- **Core Replacement:** Implement a production-grade FastAPI REST API with real PostgreSQL persistence, Alembic migrations, and true Pydantic request validation.
- **ML Encapsulation:** Encapsulate model inference into an isolated, versioned service with strict feature engineering pipelines.
- **Enterprise Security:** Enforce OIDC authentication, RBAC, and tamper-evident audit logging for all patient and clinical data interactions.
