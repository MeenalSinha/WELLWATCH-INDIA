# WELLWATCH PREVENT — Production Deployment & Operations Guide
**Project:** WELLWATCH PREVENT — Life-Course Preventive Care Intelligence  
**Target Market:** Abu Dhabi, United Arab Emirates  
**Regulatory Framework:** ADHICS v2, UAE Federal Law No. 2 of 2019, Wadeema's Law

---

## 1. Architecture Overview
WELLWATCH PREVENT is an enterprise population health intelligence platform built with:
- **Backend:** Python 3.11+ / FastAPI with SQLAlchemy 2.0 and Alembic.
- **Frontend:** Next.js 14 App Router (React 18 / TypeScript / Tailwind CSS).
- **Database:** PostgreSQL 16 Enterprise with connection pooling.
- **Security:** Non-root OCI containers, AES-256 encrypted storage, TLS 1.3 in transit.
- **Sovereign Hosting:** Bound strictly to UAE Sovereign Cloud regions (G42 Cloud / Microsoft Azure UAE Central - Abu Dhabi).

---

## 2. Deployment Tiers

### Tier 1: Local Development
```bash
# Clone repository
git clone https://github.com/MeenalSinha/WELLWATCH-INDIA.git
cd WELLWATCH-INDIA

# Launch local stack with Docker Compose
docker compose -f docker-compose.yml up -d --build

# Verify services
# Backend: http://localhost:8000/api/v1/health
# Frontend: http://localhost:3000
```

### Tier 2: Staging Environment (`docker-compose.staging.yml`)
Deploys isolated staging backend, frontend, and PostgreSQL database with synthetic Abu Dhabi test cohorts:
```bash
# Set staging environment variables
export ENVIRONMENT=staging
export SECRET_KEY=staging_secret_key_32chars_minimum_length_required_here
export POSTGRES_PASSWORD=staging_secure_pass_change_me_2026

# Launch staging stack
docker compose -f docker-compose.staging.yml up -d --build
```

### Tier 3: Production Sovereign Cloud Deployment (`docker-compose.prod.yml`)
Hardened production topology with unprivileged users (`wellwatch` 10001, `nextjs` 1001), compute/memory quotas, and zero hardcoded secrets:
```bash
# 1. Export production secrets from Enterprise Key Vault
export ENVIRONMENT=production
export SECRET_KEY=$(cat /run/secrets/wellwatch_secret_key)
export POSTGRES_USER=wellwatch_admin
export POSTGRES_PASSWORD=$(cat /run/secrets/wellwatch_db_password)
export POSTGRES_DB=wellwatch_prevent_prod
export DATABASE_POOL_SIZE=20
export DATABASE_MAX_OVERFLOW=10
export BACKEND_CORS_ORIGINS='["https://wellwatch.ae","https://app.wellwatch.ae"]'
export ALLOWED_HOSTS='["wellwatch.ae","app.wellwatch.ae","backend"]'

# 2. Run database migrations
alembic -c backend/alembic.ini upgrade head

# 3. Launch production stack
docker compose -f docker-compose.prod.yml up -d --remove-orphans
```

---

## 3. Disaster Recovery & Automated Snapshots
The platform includes an automated backup, integrity verification, and recovery engine:
```bash
# Create an atomic snapshot with SHA-256 sidecar metadata
python backend/scripts/backup_restore.py --create-snapshot --output-dir /backups

# Verify snapshot integrity against SHA-256 checksum
python backend/scripts/backup_restore.py --verify-snapshot /backups/backup_LATEST.db

# Execute automated test restore measuring RTO and data integrity
python backend/scripts/backup_restore.py --test-restore
```

---

## 4. Production Health & Readiness Probes
- **Liveness Probe:** `GET /api/v1/health` (Returns HTTP 200 `{"status": "healthy"}`)
- **Readiness Probe:** `GET /api/v1/readiness` (Validates database connectivity via `SELECT 1`)
- **Metrics Endpoint:** Prometheus telemetry for Golden Signals (Latency, Traffic, Errors, Saturation).

---

## 5. Security & Regulatory Compliance
1. **In-Country Data Residency:** Zero patient data or telemetry may egress outside the United Arab Emirates.
2. **Clinical Licensure:** Care-gap closures require authentication by a licensed healthcare provider (`UserRole.HEALTHCARE_PROVIDER`). Administrative roles are barred from clinical signoff.
3. **Audit Trails:** Immutable audit logging (`audit_logs`) retained for 25 years per ADHICS standards.
