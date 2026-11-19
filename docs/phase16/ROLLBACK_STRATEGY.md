# WELLWATCH PREVENT — Production Rollback Strategy & Procedures
**Objective:** Rapid restoration of service to last known stable release without patient data loss  
**Maximum Target Rollback Window:** < 10 Minutes  
**Version:** 1.0.0

---

## 1. Rollback Criteria & Decision Triggers
A rollback is triggered when any of the following conditions persist for > 5 minutes post-deployment:
1. **Critical Clinical Decision Support Regression:** Incorrect recommendation generation or rule miscalculation affecting patient care.
2. **Elevated Error Rate:** HTTP 5xx errors exceed 1.0% of total inbound clinical traffic.
3. **Database Performance Collapse:** Unresolvable database lock contention or latency p95 > 1,500 ms.
4. **Authentication Breakdown:** Clinicians unable to authenticate or sign care plans.

---

## 2. Multi-Tier Rollback Execution

### 2.1 Container & Application Rollback
```bash
# 1. Update image tags to previous stable version
export BACKEND_TAG=v1.0.4-stable
export FRONTEND_TAG=v1.0.4-stable

# 2. Redeploy previous stable release
docker compose -f docker-compose.prod.yml up -d --no-deps backend frontend

# 3. Verify health of restored release
curl -f https://app.wellwatch.ae/api/v1/health
```

### 2.2 Database Schema Rollback
- **Additive Migrations (Preferred):** If migrations added nullable columns or new tables, **do NOT downgrade the database schema immediately**. The previous container version will ignore unused columns, avoiding table locks and preserving records created during the new deployment window.
- **Destructive Reversion (Critical Emergency Only):**
  If a schema bug corrupts data:
  ```bash
  alembic -c backend/alembic.ini downgrade -1
  ```

### 2.3 Data Preservation & Recovery
- Any encounters or screening vitals recorded during the failed release window are identified via `audit_logs` and reconciled into the primary database.
- Zero clinical records are dropped during rollback.
