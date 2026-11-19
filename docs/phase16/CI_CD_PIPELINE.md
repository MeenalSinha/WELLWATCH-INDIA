# WELLWATCH PREVENT — CI/CD Pipeline Architecture & Governance
**Platform:** GitHub Actions & Enterprise Container Registry  
**Configuration File:** `.github/workflows/ci.yml`  
**Target Environments:** Staging & Production (UAE Sovereign Cloud)

---

## 1. Pipeline Overview & Automated Stages

```
[GIT PUSH / PULL REQUEST]
           │
           ├───────────────────────────────────────────┐
           ▼                                           ▼
┌───────────────────────────────────────┐   ┌───────────────────────────────────────┐
│     BACKEND QUALITY & TEST SUITE      │   │     FRONTEND QUALITY & BUILD SUITE    │
├───────────────────────────────────────┤   ├───────────────────────────────────────┤
│ • Set up Python 3.11                  │   │ • Set up Node.js 20                   │
│ • Install dependencies                │   │ • Install frontend dependencies       │
│ • Alembic migration check             │   │ • Next.js 14 production build         │
│ • Pytest 136 test suite with coverage │   │ • 19 / 19 Static Routes verified      │
│ • Automated backup restore test       │   │ • Zero TypeScript / ESLint errors     │
└──────────────────┬────────────────────┘   └───────────────────┬───────────────────┘
                   │                                            │
                   └─────────────────────┬──────────────────────┘
                                         ▼
                   ┌───────────────────────────────────────────┐
                   │   CONTAINER BUILD & HARDENING CHECKS      │
                   ├───────────────────────────────────────────┤
                   │ • Build Dockerfile.backend (multi-stage)  │
                   │ • Build Dockerfile.frontend (multi-stage) │
                   │ • Validate docker-compose configurations  │
                   └─────────────────────┬─────────────────────┘
                                         ▼
                   ┌───────────────────────────────────────────┐
                   │        STAGING ENVIRONMENT DEPLOYMENT     │
                   │  (Automated smoke test & integration)     │
                   └─────────────────────┬─────────────────────┘
                                         ▼ (Manual Approval Gate)
                   ┌───────────────────────────────────────────┐
                   │       PRODUCTION CANARY / ROLLOUT         │
                   └───────────────────────────────────────────┘
```

---

## 2. Quality Gate Metrics Enforced in Pipeline
1. **Zero Failing Tests:** If any single test out of 136 fails, the pipeline aborts immediately.
2. **Disaster Recovery Pre-flight:** `python backend/scripts/backup_restore.py --test-restore` runs during the backend job to ensure backup mechanisms are operative.
3. **Frontend Static Compilation:** All 19 pages must compile without CSR bailout or missing Suspense boundaries.
4. **Container Linting & Build:** Both containers must build cleanly with unprivileged user definitions.

---

## 3. Branching Strategy & Release Tags
- `feature/*`: Continuous testing and linting on every commit.
- `develop`: Automatic deployment to Staging environment upon PR merge.
- `main`: Protected branch requiring 2 peer reviews, 100% CI pass, and signed commits. Tags (`v1.0.0`) trigger staging-to-prod promotion workflows.
