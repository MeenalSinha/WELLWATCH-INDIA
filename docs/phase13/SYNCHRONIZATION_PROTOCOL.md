# OFFLINE SYNCHRONIZATION PROTOCOL SPECIFICATION
## Phase 13 Batch & Idempotency Protocol

**Project:** WELLWATCH PREVENT — Life-Course Preventive Care Intelligence  
**Target Market:** Abu Dhabi, United Arab Emirates  

---

## 1. Protocol Overview

The WELLWATCH PREVENT Offline Synchronization Protocol provides guaranteed at-least-once transmission with exactly-once processing semantics for field health operations collected during disconnected sessions.

---

## 2. Request & Response Payload Structure

### Request: `POST /api/v1/offline/sync`
```json
{
  "device_id": "CHW-DEV-001",
  "operations": [
    {
      "operation_id": "9b1deb4d-3b7d-4bad-9bdd-2b0d7b3dcb6d",
      "idempotency_key": "IDEMP-PATIENT-123-1700000000",
      "resource_type": "SCREENING",
      "operation_type": "CREATE",
      "client_timestamp": "2026-11-18T04:30:00Z",
      "payload": {
        "patient_id": "pat-12345",
        "encounter_type": "COMMUNITY_VITALS_CHECK",
        "blood_pressure_systolic": 125,
        "blood_pressure_diastolic": 80,
        "glucose_value": 95.0,
        "bmi_value": 24.1,
        "clinical_notes": "Routine home visit."
      }
    }
  ]
}
```

### Response: `200 OK`
```json
{
  "batch_id": "b12e34-5678",
  "total_processed": 1,
  "synced_count": 1,
  "conflict_count": 0,
  "rejected_count": 0,
  "results": [
    {
      "operation_id": "9b1deb4d-3b7d-4bad-9bdd-2b0d7b3dcb6d",
      "status": "SYNCED",
      "resource_id": "rec-998877",
      "server_timestamp": "2026-11-18T05:00:00Z",
      "error_message": null
    }
  ],
  "server_time": "2026-11-18T05:00:00Z"
}
```

---

## 3. Idempotency Guarantees

1. The server checks `SyncOperationAudit` using `idempotency_key`.
2. If an audit record with `status = 'SYNCED'` exists, the server short-circuits re-execution and immediately returns the previously committed `resource_id`.
3. Duplicate network transmissions never insert duplicate screenings or double-count population health statistics.
