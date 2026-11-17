# WELLWATCH PREVENT — Architecture Decision Records (ADRs)
## Key Architectural, Technology, and Infrastructure Decisions

**Status:** APPROVED ARCHITECTURE BASELINE (PHASE 2)  
**Target Market:** Abu Dhabi, United Arab Emirates  

---

## ADR-01: Adoption of Modular Monolith Over Premature Microservices
- **Context:** The legacy project was a single-file Streamlit script (`streamlit_app.py`) with coupled presentation, SQL, and arithmetic heuristics. We required a modern, scalable, maintainable architecture.
- **Decision:** Build a **Modular Monolith** initially (FastAPI backend + Next.js frontend + PostgreSQL relational database).
- **Rationale:** Microservices introduce distributed transaction complexity, network overhead, and security boundary fragmentation that are unjustified for an early-stage healthcare platform. Clear domain module boundaries (`auth`, `organizations`, `consent`, `screening`, `audit`) allow clean separation of concerns and straightforward future extraction into standalone services if throughput justifies it.
- **Consequences:** Low operational complexity, shared in-process transaction safety, rapid local iteration, zero inter-service network latency.

---

## ADR-02: Backend Framework Selection: FastAPI & Pydantic
- **Context:** Healthcare applications demand rigorous input validation, explicit data schemas, low latency, and automated OpenAPI documentation.
- **Decision:** Implement the backend using **Python 3.11+, FastAPI, and Pydantic v2**.
- **Rationale:** FastAPI provides native OpenAPI/Swagger generation, high-performance async capabilities, dependency-injection-based RBAC, and robust Pydantic data modeling for validating physiological ranges (e.g. Diastolic < Systolic, blood glucose bounds).
- **Consequences:** Clear contract-first schemas, rapid prototyping, automatic client generation readiness.

---

## ADR-03: Relational Database Strategy: PostgreSQL with SQLAlchemy 2.0 & Alembic
- **Context:** The legacy application used an unindexed SQLite database (`data/wellwatch.db`) with 3 tables, no foreign key constraints, and 0 row persistence from the UI.
- **Decision:** Standardize on **PostgreSQL 16** as the primary relational store, managed via **SQLAlchemy 2.0 Declarative ORM** and **Alembic** migrations. Support SQLite dual-engine compatibility for local test isolation.
- **Rationale:** PostgreSQL provides ACID compliance, JSONB support for consent scopes and audit metadata, row-level security capabilities, and robust connection pooling. Alembic provides deterministic, auditable migration history.
- **Consequences:** Clean schema evolution, reproducible migrations, transactional integrity across multi-tenant boundaries.

---

## ADR-04: Frontend Framework: Next.js 14, TypeScript & Tailwind CSS
- **Context:** Frontline community health workers, clinicians, and citizens require a responsive, accessible, professional interface without bloated templates.
- **Decision:** Adopt **Next.js 14 (App Router), TypeScript, and Tailwind CSS**.
- **Rationale:** Server-side rendering capabilities, zero runtime styling overhead, strict type safety matching backend Pydantic schemas, and WCAG 2.1 AA accessibility compliance.
- **Consequences:** High performance, accessible user experience, strong typing preventing frontend runtime bugs.

---

## ADR-05: Strict Decoupling of Clinical Rules from Machine Learning
- **Context:** The legacy project conflated rule points and ML probabilities into an uncalled, heuristic risk function.
- **Decision:** Strictly separate **Deterministic Clinical Screening Eligibility** (governed by Abu Dhabi DoH guidelines) from **Statistical Machine Learning Risk Estimation** (probabilistic chronic trajectory).
- **Rationale:** Clinical safety dictates that statistical machine learning models must never cancel statutory age-based screening rights (e.g. IFHAS or Well-Child Visits). ML risk estimates are strictly decision-support aids for prioritization and clinical vigilance.
- **Consequences:** Auditable clinical compliance, physician trust, transparent explainability.
