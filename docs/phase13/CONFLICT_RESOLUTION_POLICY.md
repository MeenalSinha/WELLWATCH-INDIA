# CLINICAL CONFLICT RESOLUTION POLICY
## Phase 13 Clinical Safety & Observation Adjudication

**Project:** WELLWATCH PREVENT — Life-Course Preventive Care Intelligence  
**Target Market:** Abu Dhabi, United Arab Emirates  
**Governing Standard:** ADPHC Clinical Safety & Information Governance Policy  

---

## 1. The Clinical Conflict Problem

In an offline-first healthcare application, a Community Health Worker may screen a patient while disconnected from the central server. Concurrently, that same patient may visit a tertiary hospital (e.g. Cleveland Clinic Abu Dhabi or Tawam Hospital), where licensed clinicians perform diagnostic laboratory work or vitals checks.

If the CHW device later synchronizes with a naive **"Last-Write-Wins" (LWW)** strategy, the CHW's potentially older or less precise field measurement would silently overwrite the verified hospital observation. This is a severe clinical safety risk.

---

## 2. Detection Criteria

A clinical conflict is triggered if and only if:
1. An incoming offline operation (`operation_type = 'CREATE'` or `'UPDATE'`) attempts to record a clinical observation or screening for a patient.
2. The server already holds a `ClinicalObservation` or `ScreeningRecord` for that same patient with `collection_time` or `screening_date` **newer than** the client's cached baseline (within the last 72 hours).
3. The discrepancy exceeds defined clinical tolerance thresholds:
   - Blood pressure delta: $|\Delta \text{Systolic}| \ge 15\text{ mmHg}$ or $|\Delta \text{Diastolic}| \ge 10\text{ mmHg}$.
   - Glucose delta: $|\Delta \text{Glucose}| \ge 20\text{ mg/dL}$.

---

## 3. Mandatory Quarantine Protocol

When a conflict is detected:
1. The server **REJECTS** silent overwrite.
2. The server creates a `SyncConflict` record (`status = 'PENDING'`) recording:
   - `client_payload`: Raw values captured by the CHW in the field.
   - `server_payload`: Verified observation values currently in the electronic health record.
   - `conflict_reason`: Explicit clinical explanation of the discrepancy.
3. The batch synchronization operation returns `status = 'CONFLICT'`.
4. The field record is placed in a quarantined state where it does not alter aggregate population metrics or active clinical care plans until reviewed.

---

## 4. Licensed Provider Adjudication Workflow

Only authorized clinical personnel (`HEALTHCARE_PROVIDER`, `POPULATION_HEALTH_ADMIN`, `SYSTEM_ADMIN`) can resolve conflicts:
- **`KEEP_SERVER` (Default & Preferred):** Preserves the hospital record; logs the CHW measurement as secondary community observation with non-authoritative tag.
- **`ACCEPT_CLIENT`:** Overwrites or reconciles with the CHW measurement.
- **Mandatory Justification:** Every resolution requires a mandatory clinical rationale note (`notes`) signed and audit-logged under the resolving clinician's user ID.
