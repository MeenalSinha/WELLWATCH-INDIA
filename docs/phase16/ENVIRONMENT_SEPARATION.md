# WELLWATCH PREVENT — Environment Separation & Isolation Policy
**Standard:** ADHICS v2 Section 14 (System Operations & Environment Management)  
**Tiers:** Local Development, Dedicated Staging, Production Sovereign Cloud

---

## 1. Environment Architecture & Data Boundaries

```
┌────────────────────────────────┐    ┌────────────────────────────────┐    ┌────────────────────────────────┐
│      LOCAL DEVELOPMENT         │    │       STAGING ENVIRONMENT      │    │     PRODUCTION ENVIRONMENT     │
├────────────────────────────────┤    ├────────────────────────────────┤    ├────────────────────────────────┤
│ • Docker Compose (`dev`)       │    │ • Compose / K8s (`staging`)    │    │ • Production K8s Cluster       │
│ • SQLite or Local Postgres     │    │ • Isolated Cloud PostgreSQL    │    │ • Dedicated Sovereign Cloud    │
│ • Pure Synthetic / Mock Data   │    │ • Synthetically De-identified  │    │ • Real Patient PHI / Encounters│
│ • Localhost CORS Binding       │    │ • Staging Domain Allowlist     │    │ • Strict UAE Geo-fence         │
│ • Debug Logging Allowed        │    │ • Production-like Sim          │    │ • Zero Debug / Audit Enforced  │
└────────────────────────────────┘    └────────────────────────────────┘    └────────────────────────────────┘
                 ▲                                     ▲                                     ▲
                 │                                     │                                     │
           NO PHI ALLOWED                        NO REAL PHI                           RESTRICTED ACCESS
```

---

## 2. Security Separation Rules

### 2.1 Credential & Secret Isolation
1. **Zero Secret Sharing:** Production secrets (`SECRET_KEY`, database passwords, TLS private keys) must NEVER be used in Development or Staging.
2. **Fail-Safe Startup Verification:**
   - In `backend/app/core/config.py`, the backend checks `validate_production_settings()`.
   - If `ENVIRONMENT == "production"` and the `SECRET_KEY` matches a development default, the application deliberately aborts initialization with `ValueError`.
   - If `DATABASE_URL` starts with `sqlite://` in production, startup aborts immediately.

### 2.2 Network & Database Segregation
- **Development:** Runs on developer workstation; database volume is destroyed on rebuild.
- **Staging (`docker-compose.staging.yml`):** Deployed to an isolated VNet; database populated with synthetic test cohorts mimicking Abu Dhabi demography.
- **Production (`docker-compose.prod.yml`):** Hosted in sovereign cloud; network ingress strictly locked down to healthcare WAF; database protected by private endpoints.

---

## 3. Promotion Pipeline & Quality Gates
Before any build is promoted from Staging to Production, it must satisfy:
1. 100% test pass rate in CI/CD pipeline.
2. 19/19 Next.js routes compiled cleanly without errors.
3. Cryptographic disaster recovery snapshot verified (`backup_restore.py --test-restore`).
4. Clinical Safety Officer and Security Lead electronic sign-off.
