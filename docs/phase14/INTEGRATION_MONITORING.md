# INTEROPERABILITY MONITORING & HEALTH CHECKS
## Phase 14 Health Diagnostics & Circuit Breaking

**Project:** WELLWATCH PREVENT — Life-Course Preventive Care Intelligence  
**Target Market:** Abu Dhabi, United Arab Emirates  

---

## 1. Monitoring Endpoint Architecture

The integration engine surfaces health and telemetry diagnostics via `GET /api/v1/integrations/monitoring`:
- `total_providers`: Total registered external integration adapters (4).
- `configured_providers`: Number of adapters with validated production credentials (0).
- `not_configured_providers`: Number of adapters operating under honest `NOT_CONFIGURED` boundaries (4).
- `providers`: Detailed adapter telemetry including prerequisites, status, and last heartbeat.
- `recent_jobs`: Last 50 integration batch jobs with execution timestamp, status, and duration.

---

## 2. Circuit Breaking & Fault Isolation

To prevent external network outages from cascading into the clinical application:
- **Timeout Policy:** 5-second timeout on all outbound FHIR HTTP requests.
- **Circuit Breaker:** If 5 consecutive failures occur within a 2-minute window, the adapter trips to `CIRCUIT_OPEN`, failing fast locally and preventing thread pool exhaustion.
- **Dead-Letter Handling:** Ingested messages failing structural validation are preserved in `integration_job_logs` with raw payload and stack trace for informatics review.
