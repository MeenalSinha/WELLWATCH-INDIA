# PHASES 12–14: SECURITY, PRIVACY & COMPLIANCE AUDIT
**Project:** WELLWATCH PREVENT — Life-Course Preventive Care Intelligence  
**Target Market:** Abu Dhabi, United Arab Emirates  
**Regulatory Standards:**  
- Abu Dhabi Healthcare Information and Cyber Security (ADHICS) Standard  
- UAE Federal Health Data Law (Law No. 2 of 2019)  
- UAE Child Protection Law (Wadeema's Law, Federal Law No. 3 of 2016)  
- Department of Health Abu Dhabi (DoH) Patient Data Privacy Guidelines  

---

## 1. Small-Cell Privacy Suppression & Statistical Disclosure Control (SDC)

### Risk Analysis
In population health surveillance dashboards, reporting disease prevalence or screening counts for small cohorts ($N < 5$) creates a high risk of patient re-identification, especially in sparsely populated geographic sectors like Al Dhafra. Furthermore, adversaries can execute **differencing attacks** by subtracting known sub-cohorts from aggregate totals to reveal individual identities.

### Technical Implementation & Controls
1. **Primary Small-Cell Masking:**
   - Evaluated in `backend/app/services/population_analytics_service.py` (`get_geographic_analytics`).
   - Any geographic region where `registered_population < 5` or `completed_screenings < 5` is automatically masked:
     - Count display replaced with governed label `"< 5"` (configured in `configs/population_policy.json`).
     - Percentage rates are omitted (`screening_coverage_percentage = None`).
     - Flag `is_suppressed = True` is transmitted to the client for clear UI rendering.
2. **Complementary (Secondary) Suppression:**
   - If exactly one region in a multi-region aggregate is suppressed, an adversary could compute the suppressed value by subtracting the remaining regions from the reported emirate total.
   - When `apply_complementary_suppression: true` is configured in `population_policy.json`, WELLWATCH PREVENT automatically suppresses the second-smallest non-empty cell, neutralizing mathematical differencing.
3. **Query Access Auditing:**
   - Every execution of `/api/v1/population/geographic` creates an immutable audit row in `analytics_access_logs` tracking user ID, organization, timestamp, query parameters, and whether suppression was active.

---

## 2. National & Healthcare Identifier Privacy

### Risk Analysis
Exposing raw Emirates IDs (UID: 784-YYYY-XXXXXXX-Z) or Medical Record Numbers (MRN) across unencrypted channels or in centralized analytics creates privacy vulnerability.

### Technical Implementation & Controls
1. **Zero-Knowledge Identifier Hashing:**
   - In `backend/app/models/person.py`, the `PersonIdentifier` entity separates the primary display value from the lookup index:
     - `identifier_value`: Identifiable value stored with restricted column-level access.
     - `identifier_hash`: One-way SHA-256 cryptographic digest used for zero-knowledge patient lookup and deduplication without exposing plaintext identifiers in logs or SQL index dumps.
2. **Keyed Hash / Salt Recommendations for Production:**
   - For enterprise production deployment under ADHICS, the platform supports prefixing a deployment-specific secret pepper (`PEPPER + identifier_value`) before SHA-256 computation to prevent rainbow-table precomputation attacks against 15-digit Emirates ID formats.

---

## 3. Role-Based Access Control (RBAC) & Tenant Isolation

### Access Control Matrix for Phases 12–14

| Role | Population Surveillance (`/population`) | CHW Field Workspace (`/chw`) | Integration Center (`/integrations`) | Audit Logs (`/audit`) |
| :--- | :--- | :--- | :--- | :--- |
| `POPULATION_HEALTH_ADMIN` | **Full Access** (All tabs, exports) | Forbidden | Read-Only | Read-Only |
| `COMMUNITY_HEALTH_WORKER` | Forbidden | **Full Access** (Assigned cohort, field intake, outbox) | Forbidden | Forbidden |
| `HEALTHCARE_PROVIDER` | Read-Only | Conflict Adjudication | Read-Only | Forbidden |
| `DATA_GOVERNANCE_OFFICER` | Read-Only (Audit view) | Forbidden | Read-Only (Logs) | **Full Access** |
| `SYSTEM_ADMIN` | Configuration & Metrics | Device Management | **Full Access** (Ingest, Export, Health) | **Full Access** |
| `INDIVIDUAL` (Patient) | Forbidden | Forbidden | Forbidden | Forbidden |

### Tenant Isolation Enforcement
- Multitenancy is enforced at the database query level:
  ```python
  if user.role != UserRole.SYSTEM_ADMIN and pat_org_id != user.organization_id:
      raise HTTPException(status_code=403, detail="Tenant boundary violation")
  ```
- CHWs cannot synchronize or view patients outside their licensed healthcare organization or assigned district.

---

## 4. Community Health Worker (CHW) Offline Data Security

### Policy Conformance (`configs/offline_policy.json`)
1. **Device Binding:**
   - Field tablets must be actively registered (`DeviceRegistration`) with serial numbers, OS versions, and user bindings before downloading patient cohorts.
2. **72-Hour Cache Eviction:**
   - Assignment packages automatically expire after 72 hours.
   - The frontend IndexedDB storage engine (`offlineStorage.ts`) monitors package age and triggers policy warnings if offline cache exceeds the allowable duration.
3. **PHI Storage Standard:**
   - Plain `window.localStorage` is prohibited for unencrypted PHI.
   - The client employs structured IndexedDB stores (`chw_outbox`, `chw_cache`, `chw_device`) with private object-store isolation.
4. **Non-Destructive Conflict Resolution:**
   - Field screening sync never performs last-write-wins over hospital EHR records.
   - All conflicts are quarantined in `sync_conflicts` with client and server snapshots preserved for licensed provider review.
