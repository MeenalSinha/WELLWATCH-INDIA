# WELLWATCH PREVENT — Production Database Operations Manual
**Database Engine:** PostgreSQL 16 (Enterprise / Cloud Managed)  
**ORM / Migration Tool:** SQLAlchemy 2.0 / Alembic  
**Target Market:** Abu Dhabi, UAE (ADHICS v2 Data Management Standard)

---

## 1. Connection Pooling & Resource Configuration
To ensure resilient, low-latency database access across all primary care clinics:
- **Pool Sizing (`backend/app/core/database.py`):**
  - `DATABASE_POOL_SIZE = 20`: Baseline persistent connection pool per backend worker.
  - `DATABASE_MAX_OVERFLOW = 10`: Burst buffer during peak screening intake (up to 30 concurrent connections per replica).
  - `pool_pre_ping = True`: Emits lightweight `SELECT 1` ping before leasing connection to detect and prune dead sockets.
  - `pool_recycle = 1800`: Recycles idle sockets every 30 minutes to prevent firewall connection drops.
  - `pool_timeout = 30`: Rejects requests waiting > 30 seconds for a connection with explicit timeout error.

---

## 2. Schema Migration Governance (Alembic)
1. **Migration Version Control:** All schema changes are tracked via Alembic migration scripts in `backend/alembic/versions/`.
2. **Current Schema Baseline:** `007_phases_12_to_14_schema` (verified `head`).
3. **Zero-Downtime Migration Policy:**
   - Additive changes (new nullable columns, new tables) applied online prior to application rollout.
   - Destructive changes (column renames, column drops) require a two-phase deprecation cycle across releases.
   - Pre-deployment validation runs `alembic upgrade head` in staging before production cutover.

---

## 3. High-Volume Partitioning & Indexing Strategy
For tables expected to grow beyond 10 million rows:
1. `audit_logs`:
   - Range-partitioned by month (`created_at`).
   - Retained on high-speed SSD storage for 3 years, archived to cold encrypted blob storage for 25 years per ADHICS requirements.
2. `clinical_observations` & `screening_encounters`:
   - Indexed on `(patient_id, observation_type, effective_date)`.
   - Index on `(organization_id, status)` for clinic dashboard query acceleration.

---

## 4. Maintenance Schedules
- **Autovacuum Tuning:** Configured aggressively for transactional tables (`autovacuum_vacuum_scale_factor = 0.05`, `autovacuum_analyze_scale_factor = 0.02`).
- **Nightly Integrity Checks:** Automated checksum scan executed during low-traffic maintenance window (02:00 – 04:00 GST).
