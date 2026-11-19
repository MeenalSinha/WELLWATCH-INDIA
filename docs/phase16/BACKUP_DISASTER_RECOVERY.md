# WELLWATCH PREVENT — Backup & Disaster Recovery Strategy
**Regulatory Framework:** ADHICS v2 Business Continuity Management & ISO 22301  
**Disaster Recovery Engine:** `backend/scripts/backup_restore.py`  
**Target RTO:** < 15 Minutes (Local) / < 4 Hours (Full Regional Disaster)  
**Target RPO:** < 1 Hour (Continuous WAL Archiving)

---

## 1. Disaster Recovery Topology

```
┌──────────────────────────────────────────────┐       Replication       ┌──────────────────────────────────────────────┐
│        PRIMARY PRODUCTION REGION             │ ──────────────────────> │        DISASTER RECOVERY REGION              │
│       (Azure UAE Central - Abu Dhabi)        │       (Encrypted)       │         (Azure UAE North - Dubai)            │
├──────────────────────────────────────────────┤                         ├──────────────────────────────────────────────┤
│ • Active PostgreSQL Cluster                  │                         │ • Standby PostgreSQL Read Replica            │
│ • Daily Full Snapshot (`backup_restore.py`)  │                         │ • Replicated Snapshots with SHA-256 Hashes   │
│ • Continuous Write-Ahead Log (WAL) Streaming │                         │ • Cold Standby Backend & Frontend Clusters   │
└──────────────────────────────────────────────┘                         └──────────────────────────────────────────────┘
```

---

## 2. Backup Schedules & Retention Policy

| Backup Type | Frequency | Destination | Retention Period | Verification Protocol |
| :--- | :--- | :--- | :--- | :--- |
| **Continuous WAL** | Real-time (< 1 min) | Azure Sovereign Blob Storage | 30 Days | Automated replay verification |
| **Daily Full Snapshot** | Daily at 01:00 GST | Encrypted Backup Vault | 90 Days | SHA-256 checksum & test restore |
| **Monthly Archive** | 1st of each month | Immutable WORM Cloud Storage | 25 Years (ADHICS) | Cryptographic signature audit |

---

## 3. Disaster Recovery Execution Steps

### In Case of Database Corruption / Primary Failure:
1. **Assess Impact & Halt Ingress:** Shift WAF to maintenance holding page.
2. **Identify Latest Known Good Snapshot:**
   ```bash
   python backend/scripts/backup_restore.py --list-snapshots
   ```
3. **Verify Cryptographic Integrity:**
   ```bash
   python backend/scripts/backup_restore.py --verify-snapshot /backups/backup_LATEST.db
   ```
4. **Execute Restoration:**
   ```bash
   python backend/scripts/backup_restore.py --restore-snapshot /backups/backup_LATEST.db --target-db /data/wellwatch_prevent.db
   ```
5. **Execute Health Checks & Resume Traffic:** Verify `/api/v1/health` and restore traffic.
