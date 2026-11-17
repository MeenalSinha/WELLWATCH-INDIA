# WELLWATCH PREVENT — Pre-Audit Baseline Record

**Project:** WELLWATCH PREVENT — Life-Course Preventive Care Intelligence  
**Document:** Initial Baseline Audit Report  
**Date:** November 17, 2026  
**Auditor:** Senior Healthcare Technology Architect & Production Readiness Auditor  

---

## 1. Environment & Repository Baseline

- **Repository Root:** `c:\Users\Meenal Sinha\Downloads\WELLWATCH-INDIA`
- **Git Branch:** `main`
- **Current Commit:** `132ecc4` (`fix(backend): add missing Tuple import in data_quality_service for Python 3.11 compatibility`)
- **Remote:** `https://github.com/MeenalSinha/WELLWATCH-INDIA`
- **Python Version:** 3.14.2 (local) / 3.11 (CI Ubuntu-latest)
- **Node.js / npm Version:** Node.js v22.19.0 / npm 11.6.0
- **Database Backend:** SQLite (dev/test) / PostgreSQL 16+ (production-ready configuration)
- **Current Alembic Revision:** `004_genomic_intelligence` (Head)

---

## 2. Dependency Inventory

### Backend (`backend/requirements.txt`)
- `fastapi==0.141.1`
- `uvicorn==0.52.1`
- `pydantic==2.13.0`
- `pydantic-settings==2.13.0`
- `email-validator>=2.0.0`
- `sqlalchemy==2.1.0b3`
- `alembic==1.13.2`
- `bcrypt==5.0.0`
- `PyJWT==2.12.1`
- `python-jose==3.3.0`
- `passlib==1.7.4`
- `httpx==0.28.1`
- `pytest==8.2.2`
- `pytest-asyncio==0.23.7`
- `xgboost==3.2.0`
- `scikit-learn==1.8.0`
- `pandas>=2.0.0`
- `numpy>=1.24.0`

### Frontend (`frontend/package.json`)
- `next`: 14.2.15
- `react`: ^18
- `react-dom`: ^18
- `lucide-react`: ^0.453.0
- `typescript`: ^5
- `tailwindcss`: ^3.4.1

---

## 3. Baseline Test Execution Results

- **Test Framework:** Pytest 8.2.2
- **Command:** `python -m pytest backend/tests/ -v`
- **Collected Test Files:** 19 files
- **Total Test Cases:** 71 test cases
- **Passed:** 71 (100%)
- **Failed:** 0
- **Skipped:** 0
- **Execution Time:** ~8.37s

---

## 4. Key Areas Identified for In-Depth Audit

1. **Hardcoding & Heuristics Check:**
   - Scan for embedded clinical thresholds in services (e.g., blood pressure cutoffs in `preventive_assessment_service.py`, fixed days in `data_quality_service.py`).
   - Move operational parameters and clinical thresholds into validated, versioned JSON configuration schemas.
2. **Genomic Actionability & Knowledge Base:**
   - The gene-to-disease mappings in `genomic_service.py` (`LDLR`, `BRCA1`, etc.) currently reside directly in Python dictionary code. These must be moved to an external, versioned, authoritative knowledge base (`configs/genomics/actionable_variants.json`) with official ClinVar/EGP/ACMG evidence citations.
3. **Frontend Mock Elimination:**
   - Review all Next.js pages and components in `frontend/src/` to confirm that all forms, directories, profiles, and timelines connect to live API endpoints and handle empty/loading/error states properly with zero mock arrays or fake data.
4. **End-to-End Workflow Verification:**
   - Execute integration workflows (Pediatric, Adult, Insufficient Data, Genomic, and ML) testing real database persistence and state transitions.
