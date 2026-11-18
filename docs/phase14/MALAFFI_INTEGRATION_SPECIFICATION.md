# MALAFFI HIE INTEGRATION SPECIFICATION
## Phase 14 Health Information Exchange Adapter

**Project:** WELLWATCH PREVENT — Life-Course Preventive Care Intelligence  
**Target Market:** Abu Dhabi, United Arab Emirates  
**Integration Partner:** Malaffi (Abu Dhabi Health Data Services / DoH)  

---

## 1. Malaffi Overview & Purpose

Malaffi is the first Health Information Exchange platform in the MENA region, connecting 100% of public and private hospitals, clinics, and pharmacies in Abu Dhabi.

WELLWATCH PREVENT interacts with Malaffi for:
1. **Reciprocal Ingestion:** Ingesting external laboratory findings, hospital admissions, and specialist consultations to update preventive risk profiles and close care gaps.
2. **Preventive Export:** Transmitting screening summaries, IFHAS assessments, and community health worker observations to the central Abu Dhabi longitudinal health record.

---

## 2. Production Prerequisites & Honest Status

In accordance with strict security standards, live connection requires:
- DOH-issued Mutual TLS Client Certificate (`MALAFFI_MTLS_CERT_PATH`).
- Private Key matching registered certificate (`MALAFFI_MTLS_KEY_PATH`).
- DOH Malaffi Organization Identifier (`MALAFFI_ORG_ID`).
- Production Endpoint Provisioning (`MALAFFI_BASE_URL`).

### Current Status: `NOT_CONFIGURED`
In this deployment, the adapter correctly and honestly reports `NOT_CONFIGURED` because production mTLS certificates have not been provisioned. Zero fake connections or mock success tokens are accepted.
