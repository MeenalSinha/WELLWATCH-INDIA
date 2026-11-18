# INTEROPERABILITY SECURITY SPECIFICATION
## Phase 14 Transport Security, mTLS & Credential Isolation

**Project:** WELLWATCH PREVENT — Life-Course Preventive Care Intelligence  
**Target Market:** Abu Dhabi, United Arab Emirates  

---

## 1. Transport Security & Mutual TLS (mTLS)

All communication between WELLWATCH PREVENT and external healthcare entities (Malaffi, SEHA, National Lab Network) mandates:
- **TLS 1.3** as primary protocol (TLS 1.2 with strict AEAD ciphers as minimum).
- **Mutual TLS (mTLS):** Both client and server must present X.509 digital certificates issued by a trusted UAE healthcare certificate authority (DoH PKI).
- **Certificate Pinning:** External endpoints enforce certificate thumbprint pinning to mitigate man-in-the-middle attacks.

---

## 2. Cryptographic Provenance Tracking

Every external health observation ingested into the system is cryptographically linked to its origin via the `ProvenanceRecord` model:
- `source_system`: Identifying string (e.g. `MALAFFI`, `SEHA_EMR`, `LAB_NETWORK`).
- `external_resource_id`: Origin identifier in the external system.
- `payload_hash`: SHA-256 cryptographic digest of the received raw FHIR payload.
- `ingested_at`: UTC timestamp of arrival.
- `target_resource_type` & `target_resource_id`: Local database foreign key pointers.
