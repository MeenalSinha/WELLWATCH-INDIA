# OFFLINE THREAT MODEL & RISK MITIGATION MATRIX
## Phase 13 STRIDE Security Assessment

**Project:** WELLWATCH PREVENT — Life-Course Preventive Care Intelligence  
**Target Market:** Abu Dhabi, United Arab Emirates  

---

## 1. STRIDE Threat Analysis

| Threat Category | Specific Threat Description | Architectural Mitigation in Phase 13 |
| :--- | :--- | :--- |
| **Spoofing** | Rogue client submits fabricated screening batches impersonating a licensed CHW. | Strict device registration requirement (`DeviceRegistration`), bearer token verification, and CHW role enforcement. |
| **Tampering** | Local client modification of vital signs or clinical observations on the device before sync. | Client-side cryptographic hashing of observation payload, physiological range validation, and server-side verification. |
| **Repudiation** | Clinician or CHW denies recording a disputed field observation. | Cryptographic audit trail in `SyncOperationAudit` and immutable `ProvenanceRecord` logging actor ID, client timestamp, and server receipt time. |
| **Information Disclosure** | Stolen or lost field tablet subjected to offline memory or file forensics. | 72-hour time-bounded caching (`expires_at`), AES-GCM local storage encryption, and remote revocation wipe protocol. |
| **Denial of Service** | Flooding the sync endpoint with massive operation batches or malformed payloads. | Strict batch size enforcement (`max_batch_size: 25`), rate limiting, and schema validation. |
| **Elevation of Privilege** | CHW attempts to unilaterally resolve clinical conflicts or overwrite tertiary hospital observations. | Strict RBAC: conflict resolution endpoint `POST /api/v1/offline/conflicts/{id}/resolve` strictly rejects CHWs (HTTP 403) and requires licensed provider roles. |
