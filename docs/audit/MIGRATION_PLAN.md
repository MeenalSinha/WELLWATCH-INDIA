# WELLWATCH PREVENT — Architecture Migration Plan
## Asset Disposition: Retain, Refactor, Replace, and Deprecate

**Project:** `WELLWATCH PREVENT — Life-Course Preventive Care Intelligence`  
**Migration Date:** 2026-11-17  
**Strategy:** Phased Evolution & Preservation of Legacy Foundations  

---

## 1. Migration Principles
1. **Preserve Legacy Assets:** Do not overwrite or destructively delete original hackathon files (`streamlit_app.py`, `WELLWATCH_INDIA.ipynb`, `models/`, `data/`). They serve as verified historical benchmarks.
2. **API-First Modular Architecture:** Transition from a single-file Streamlit script to a modern multi-tier web application (FastAPI backend + Next.js frontend + PostgreSQL).
3. **No Cosmetic Stubs:** Replace all fake buttons and static data generators with functional REST APIs and real database transactions.
4. **Separation of Clinical Rules & Machine Learning:** Strictly distinguish deterministic clinical screening eligibility (DoH / IFHAS / Well-Child) from statistical ML risk models.

---

## 2. Component Disposition Matrix

| Original Component | Action | Destination / New Component | Rationale |
|---|---|---|---|
| `streamlit_app.py` | **Deprecate** (Preserve in root) | `frontend/` (Next.js) & `backend/` (FastAPI) | Monolithic, no RBAC, uncalled ML model, fake database actions, random dashboards. |
| `WELLWATCH_INDIA.ipynb` | **Retain** (Historical) | `docs/legacy/` / Root | Retained as documentation of the exploratory data analysis and original synthetic experiment. |
| `models/chronic_risk_model.pkl` | **Retain & Refactor** | `ml/artifacts/legacy/` & `backend/app/services/ml/` | Pickled model is functional when properly fed 28 features; preserved as a legacy baseline estimator. |
| `models/scaler.pkl` | **Retain** | `ml/artifacts/legacy/` | Preserved for legacy inference verification. |
| `models/label_encoder.pkl` | **Retain** | `ml/artifacts/legacy/` | Preserved for legacy class decoding. |
| `models/feature_list.json` | **Retain** | `backend/app/services/ml/` | Specifies legacy feature order; reused for legacy model compatibility. |
| `models/model_metadata.json` | **Retain** | `docs/legacy/` | Historical metadata record. |
| `data/wellwatch.db` | **Deprecate** (Preserve in `data/`) | PostgreSQL / SQLite Async | Unindexed SQLite DB with 5 rows and no foreign keys; superseded by relational migrations. |
| `data/raw_screening_data.csv` | **Retain** (Fixture) | `data/legacy/` | Preserved as synthetic baseline fixture. |
| `data/cleaned_data.csv` | **Retain** (Fixture) | `data/legacy/` | Preserved as synthetic baseline fixture. |
| `requirements.txt` | **Replace** | `backend/requirements.txt` & `pyproject.toml` | Needs pinned modern dependencies: `fastapi`, `uvicorn`, `pydantic`, `sqlalchemy`, `alembic`, `pytest`. |
| `README.md` | **Refactor** | `README.md` | Update with WellWatch Prevent vision, architecture, Abu Dhabi focus, and setup instructions. |
| `DEPLOYMENT.md` | **Replace** | `docs/operations/DEPLOYMENT.md` & `docker-compose.yml` | Replace insecure Heroku/pyngrok instructions with Docker Compose production setup. |
| `outputs/*` | **Retain** (Historical) | `outputs/` | Retained as baseline visualization artifacts. |

---

## 3. Target Modular Monolith Architecture

```text
wellwatch-prevent/
├── backend/                  # FastAPI Application
│   ├── app/
│   │   ├── api/              # Versioned API Endpoints (v1)
│   │   │   └── v1/
│   │   │       ├── api.py
│   │   │       └── endpoints/ # auth, users, screening, health, consent, audit
│   │   ├── core/             # Config, Database, Logging, Security
│   │   ├── models/           # SQLAlchemy 2.0 ORM Entities
│   │   ├── schemas/          # Pydantic Schemas (Request/Response)
│   │   ├── services/         # Domain Services (Auth, Audit, Screening)
│   │   └── middleware/       # Correlation ID, Security Headers, Rate Limiting
│   ├── alembic/              # Database Migrations
│   ├── tests/                # Pytest Suite (Unit, Integration, Security)
│   └── requirements.txt      # Pinned Backend Dependencies
│
├── frontend/                 # Next.js 14 / TypeScript / Tailwind CSS
│   ├── src/
│   │   ├── app/              # App Router Pages
│   │   ├── components/       # Accessible UI Primitives & Healthcare Layouts
│   │   ├── lib/              # API Client, Session, Token Management
│   │   └── types/            # TypeScript Domain Definitions
│   ├── package.json
│   └── tsconfig.json
│
├── infrastructure/           # Docker, Compose & Deployment Configs
│   ├── docker/
│   └── docker-compose.yml
│
├── docs/                     # Centralized Documentation
│   ├── audit/                # Phase 0 Audit Documents
│   ├── product/              # Phase 1 Clinical & Product Specifications
│   ├── architecture/         # Phase 2 System & Database Architecture
│   └── security/             # Phase 3 Security, RBAC & Privacy Specifications
│
├── data/                     # Legacy Data Preserved
├── models/                   # Legacy ML Artifacts Preserved
└── outputs/                  # Legacy Notebook Outputs Preserved
```

---

## 4. Phased Execution Roadmap

1. **Phase 0 (Complete):** Deep audit, baseline testing, known issue classification, migration blueprint.
2. **Phase 1 (Next):** Complete product requirements, user personas, clinical governance, and pathway engine specifications tailored to Abu Dhabi (IFHAS & Well-Child Visits).
3. **Phase 2:** Implement modular FastAPI backend, Next.js frontend, PostgreSQL / Alembic database migrations, and Docker Compose development environment.
4. **Phase 3:** Implement production authentication, RBAC, organization boundaries, consent management, and audit logging with rigorous test suites.
