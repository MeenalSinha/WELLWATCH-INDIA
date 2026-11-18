# FIELD DEVICE MANAGEMENT & LIFECYCLE
## Phase 13 Device Governance & Security Policy

**Project:** WELLWATCH PREVENT — Life-Course Preventive Care Intelligence  
**Target Market:** Abu Dhabi, United Arab Emirates  

---

## 1. Device Registration Lifecycle

```
[ UNREGISTERED DEVICE ]
         |
         |  CHW / Field Supervisor submits Device ID + Model
         v
[ REGISTERED / PENDING_ENROLLMENT ]
         |
         |  System Admin / Security Officer Approves
         v
[ ACTIVE ] <---------------------+
    |    |                       |
    |    | Sync Operations       |
    |    v                       |
    |  [ HEARTBEAT OK ] ---------+
    |
    | Device Lost / Compromised / CHW Deprovisioned
    v
[ REVOKED / SUSPENDED ]
    |
    v
(Triggers Local Storage Wipe & Denies Sync Handshake)
```

---

## 2. API Endpoints for Device Governance

- `POST /api/v1/offline/devices/register`: Registers a field tablet or smartphone.
- `GET /api/v1/offline/assignments/current`: Issues time-bounded assignment packages (<= 50 patients, 72 hours).
- `POST /api/v1/offline/sync`: Batch delta synchronization.
- `GET /api/v1/offline/conflicts`: Returns quarantined conflicts.
- `POST /api/v1/offline/conflicts/{id}/resolve`: Clinician adjudication.
