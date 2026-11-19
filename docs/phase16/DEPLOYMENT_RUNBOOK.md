# WELLWATCH PREVENT — Production Deployment Runbook
**Target Market:** Abu Dhabi, United Arab Emirates  
**Runbook Version:** 1.0.0  
**Estimated Maintenance Window:** 15 Minutes (Zero-Downtime Rolling Update)

---

## 1. Pre-Deployment Checklist
Prior to commencing deployment, the Release Engineer must verify:
- [ ] CI/CD pipeline on `main` passed with 100% green status across all jobs.
- [ ] Production database snapshot created and SHA-256 verified via `backup_restore.py`.
- [ ] Production secrets populated in Key Vault / environment variables (`SECRET_KEY`, `POSTGRES_PASSWORD`).
- [ ] Clinical Safety Officer and Infrastructure Lead approval logged.
- [ ] Off-peak deployment window confirmed (recommended: Thursday 22:00 GST).

---

## 2. Step-by-Step Deployment Procedure

### Step 1: Pre-Deployment Backup
```bash
# Execute on primary database host
python backend/scripts/backup_restore.py --create-snapshot --output-dir /backups/pre_deploy
```

### Step 2: Database Schema Migration
```bash
# Apply additive database migrations
alembic -c backend/alembic.ini upgrade head
```

### Step 3: Container Deployment (Rolling Update)
```bash
# Deploy updated production stack
docker compose -f docker-compose.prod.yml pull
docker compose -f docker-compose.prod.yml up -d --remove-orphans
```

### Step 4: Verification & Smoke Testing
Execute the automated post-deployment health check:
```bash
# Verify backend liveness & database readiness
curl -f https://app.wellwatch.ae/api/v1/health
curl -f https://app.wellwatch.ae/api/v1/readiness

# Verify frontend response
curl -I https://app.wellwatch.ae/login
```

---

## 3. Post-Deployment Verification Steps
1. Log in with a verified clinician test account.
2. Open `/dashboard` and verify real-time statistics load without errors.
3. Open `/population` and verify regional aggregation cards render.
4. Open `/chw` and test offline cache refresh.
5. Inspect `audit_logs` table to confirm login events are recording.

---

## 4. Emergency Contacts & Escalation
- **Lead DevOps Engineer:** +971-50-XXX-XXXX
- **Clinical Safety Officer:** +971-50-YYY-YYYY
- **Database Administrator:** +971-50-ZZZ-ZZZZ
