# COMMUNITY HEALTH WORKER FIELD WORKFLOW SPECIFICATION
## Phase 13 Operational Protocol

**Project:** WELLWATCH PREVENT — Life-Course Preventive Care Intelligence  
**Target Market:** Abu Dhabi, United Arab Emirates  
**Role:** Community Health Worker (`COMMUNITY_HEALTH_WORKER`)  

---

## 1. Field Screening Workflow

```
+-----------------------------------------------------------------------------------+
| STEP 1: PRE-MISSION PREPARATION (Clinic / Wi-Fi Base)                              |
| - CHW logs in via PIN / password; device registration validated.                  |
| - Synchronizes current assignment package (up to 50 patients).                   |
| - Cryptographic assignment bundle valid for 72 hours cached in local storage.     |
+-----------------------------------------------------------------------------------+
                                         |
                                         v
+-----------------------------------------------------------------------------------+
| STEP 2: FIELD VISIT & INTAKE (Community / Home Setting - Offline)                 |
| - CHW selects assigned patient from local cohort list.                            |
| - Validates patient identity via Emirates ID or full demographic confirmation.    |
| - Conducts vital signs screening (Systolic BP, Diastolic BP, Glucose, BMI).       |
| - Enters field clinical notes and observation context.                            |
+-----------------------------------------------------------------------------------+
                                         |
                                         v
+-----------------------------------------------------------------------------------+
| STEP 3: CLIENT-SIDE VALIDATION & OUTBOX QUEUING                                   |
| - Physiological bounds enforced (Systolic 60-260 mmHg, Diastolic 40-150 mmHg, etc)|
| - Unique operation ID (UUID) and Idempotency Key generated.                       |
| - Stored in offline sync outbox queue; user notified of local persistence.       |
+-----------------------------------------------------------------------------------+
                                         |
                                         v
+-----------------------------------------------------------------------------------+
| STEP 4: RECONNECTION & BATCH SYNCHRONIZATION (Cellular / Clinic Network)         |
| - Network restored; CHW triggers "Synchronize Outbox" or auto-sync engages.       |
| - Batch payload submitted to POST /api/v1/offline/sync (up to 25 ops / batch).    |
| - Server evaluates idempotency and performs clinical conflict detection.          |
| - Outbox cleared upon receipt of successful server status (SYNCED).               |
+-----------------------------------------------------------------------------------+
                                         |
                                         v
+-----------------------------------------------------------------------------------+
| STEP 5: CLINICAL CONFLICT REVIEW & ADJUDICATION (Facility Clinician)              |
| - If hospital recorded newer observations during disconnection, conflict logged.  |
| - Licensed physician/provider reviews discrepancy in conflict queue.              |
| - Provider records mandatory clinical rationale: ACCEPT_CLIENT or KEEP_SERVER.    |
+-----------------------------------------------------------------------------------+
```

---

## 2. Operational Constraints

- **Assignment Limit:** Maximum 50 patients per active assignment package.
- **Assignment Window:** 72 hours maximum before re-authentication and fresh package retrieval are required.
- **Batch Processing:** Maximum 25 sync operations per network request to avoid timeouts over low-bandwidth mobile connections.
- **Role Isolation:** CHWs can register observations and sync outboxes, but CANNOT adjudicate or resolve clinical conflicts; conflict resolution is restricted to licensed healthcare providers (`HEALTHCARE_PROVIDER`, `POPULATION_HEALTH_ADMIN`, `SYSTEM_ADMIN`).
