# WELLWATCH PREVENT — Secrets Management & Cryptographic Key Governance
**Standard:** ADHICS v2 Cryptographic Controls & ISO 27001 Key Management  
**Applicability:** Production & Staging Environments

---

## 1. Secrets Management Hierarchy

```
┌────────────────────────────────────────────────────────┐
│               ENTERPRISE KEY VAULT                     │
│    (Azure Key Vault / HashiCorp Vault in UAE Region)   │
├────────────────────────────────────────────────────────┤
│ • Production Database Passwords                        │
│ • Application Signing Keys (`SECRET_KEY`)              │
│ • Malaffi Mutual TLS (mTLS) Certificates & Keys        │
│ • Outreach Gateway API Credentials (SMS / WhatsApp)    │
└───────────────────────────┬────────────────────────────┘
                            │ Encrypted Injection via Runtime Environment
                            ▼
┌────────────────────────────────────────────────────────┐
│              CONTAINER RUNTIME ENVIRONMENT             │
│ (Zero plain text on disk, unprivileged memory mapping) │
└────────────────────────────────────────────────────────┘
```

---

## 2. Engineered Safety Controls

### 2.1 Fail-Safe Startup Validator
In `backend/app/core/config.py`:
```python
@classmethod
def validate_production_settings(cls, settings_instance: "Settings") -> None:
    if settings_instance.ENVIRONMENT.lower() == "production":
        if "development" in settings_instance.SECRET_KEY.lower() or settings_instance.SECRET_KEY == "wellwatch-prevent-secure-development-secret-key-32chars-min":
            raise ValueError("CRITICAL SECURITY ERROR: In production, SECRET_KEY must be explicitly set and cannot use the development default.")
        if settings_instance.DATABASE_URL.startswith("sqlite"):
            raise ValueError("CRITICAL SECURITY ERROR: In production, DATABASE_URL must point to a production-grade database (e.g. PostgreSQL), not SQLite.")
```
- **Defense Mechanism:** The application refuses to initialize in production if any developer default secret is present.

### 2.2 Secret Rotation Policy
1. **JWT Secret (`SECRET_KEY`):**
   - Rotated every 90 days.
   - During key transition windows, the backend accepts tokens signed by both `SECRET_KEY_CURRENT` and `SECRET_KEY_PREVIOUS` for a 48-hour overlap window, ensuring zero user disconnection.
2. **Database Credentials:**
   - Managed via cloud managed identities or dynamic Vault database roles with 30-day lease expiration.
3. **mTLS Certificates:**
   - Renewed annually with Department of Health / Malaffi PKI authorities.
