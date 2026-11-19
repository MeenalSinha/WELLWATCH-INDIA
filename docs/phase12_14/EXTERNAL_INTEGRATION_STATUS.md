# PHASES 12–14: EXTERNAL HEALTHCARE INTEGRATION STATUS & PREREQUISITES
**Project:** WELLWATCH PREVENT — Life-Course Preventive Care Intelligence  
**Target Market:** Abu Dhabi, United Arab Emirates  
**Governing Authorities:** Department of Health Abu Dhabi (DoH), Abu Dhabi Public Health Centre (ADPHC)  
**Standard:** Honest Integration Boundaries, Zero Fake Claims, Complete Interoperability Specifications  

---

## 1. Architectural Philosophy: The Honest Integration Boundary

In mission-critical healthcare software, pretending that external health information exchanges (HIE), electronic medical record (EMR) systems, or national identity gateways are connected when credentials are absent is an unacceptable safety risk.

WELLWATCH PREVENT enforces an **Honest Integration Boundary**:
- Without authorized production/sandbox credentials and mutual TLS (mTLS) certificates, every adapter returns `NOT_CONFIGURED`.
- No simulated success tokens or mock patient data are injected into clinical workflows in production.
- For local unit and integration testing, validated static test bundles conforming strictly to DoH FHIR profiles are processed by the ingestion engine to verify compliance without claiming live network links.

---

## 2. Provider Integration Register

### 1. Abu Dhabi Health Information Exchange (Malaffi)
- **Adapter Class:** `backend.app.integrations.adapters.malaffi.MalaffiAdapter`
- **Provider Type:** `HIE` (Health Information Exchange)
- **Status:** `NOT_CONFIGURED` (Honest Boundary)
- **Target Specifications:** DoH Malaffi FHIR R4 Ingestion & Query Implementation Guide v2.1
- **Required Configuration Keys:**
  - `MALAFFI_API_URL`: Secured Malaffi gateway endpoint (HTTPS)
  - `MALAFFI_CLIENT_ID`: DoH assigned OAuth2 client identifier
  - `MALAFFI_CLIENT_SECRET`: Secure client secret
  - `MALAFFI_FACILITY_ID`: Licensed healthcare facility identifier issued by DoH
- **Legal & Technical Prerequisites:**
  1. Department of Health Abu Dhabi Health Information Exchange Participation Agreement signed.
  2. Malaffi Technical Integration Architecture Review approval.
  3. Client Mutual TLS (mTLS) certificate issued by the UAE National Health PKI.
  4. IP Whitelisting on the Malaffi Enterprise Service Bus (ESB).

### 2. SEHA Ambulatory & Hospital EMR Gateway
- **Adapter Class:** `backend.app.integrations.adapters.seha_emr.SehaEmrAdapter`
- **Provider Type:** `EMR` (Electronic Medical Record)
- **Status:** `NOT_CONFIGURED` (Honest Boundary)
- **Target Specifications:** SEHA Clinical Interoperability Service Layer (HL7 FHIR R4 / REST)
- **Required Configuration Keys:**
  - `SEHA_EMR_API_URL`: SEHA FHIR integration gateway
  - `SEHA_API_KEY`: Authorized integration API key
- **Legal & Technical Prerequisites:**
  1. SEHA Clinical Data Exchange Service Level Agreement (SLA).
  2. Direct MPLS or encrypted site-to-site VPN link with SEHA Datacenter.
  3. Clinical encounter and referral bi-directional synchronization clearance.

### 3. Abu Dhabi Laboratory Network
- **Adapter Class:** `backend.app.integrations.adapters.lab_network.LabNetworkAdapter`
- **Provider Type:** `LAB` (Reference Laboratory Network)
- **Status:** `NOT_CONFIGURED` (Honest Boundary)
- **Target Specifications:** DoH Laboratory Observation Reporting Standard (LOINC + UCUM)
- **Required Configuration Keys:**
  - `LAB_NETWORK_API_URL`: Lab network EDI / FHIR ingestion endpoint
  - `LAB_NETWORK_API_KEY`: Partner API authorization key
- **Legal & Technical Prerequisites:**
  1. Laboratory network participation and data sharing memorandum.
  2. Standardized LOINC vocabulary validation agreement.
  3. Automatic panic-value alert callback registration.

### 4. UAE PASS (National Digital Identity & Authentication)
- **Adapter Class:** `backend.app.integrations.adapters.uae_pass.UaePassAdapter`
- **Provider Type:** `IDENTITY` (Federated Identity & Authentication)
- **Status:** `NOT_CONFIGURED` (Honest Boundary)
- **Target Specifications:** UAE PASS OpenID Connect (OIDC) / OAuth2 Specification
- **Required Configuration Keys:**
  - `UAE_PASS_API_URL`: UAE PASS authentication server (Staging / Production)
  - `UAE_PASS_CLIENT_ID`: Telecommunications and Digital Government Regulatory Authority (TDRA) Client ID
  - `UAE_PASS_CLIENT_SECRET`: TDRA client secret
- **Legal & Technical Prerequisites:**
  1. TDRA Commercial Service Agreement.
  2. UAE PASS Onboarding clearance and domain verification.
  3. Single Sign-On (SSO) and digital signature authorization for medical consents.

---

## 3. Provenance & Audit Verification

When external FHIR bundles are ingested into WELLWATCH PREVENT (e.g. from laboratory networks or clinic encounters):
1. An immutable `ProvenanceRecord` is created in the database.
2. The record stores:
   - `target_resource_type`: e.g. `ClinicalObservation` or `ScreeningRecord`
   - `target_resource_id`: Internal primary key
   - `source_system`: Authenticated source system identifier (e.g. `ABU_DHABI_NATIONAL_REFERENCE_LAB`)
   - `agent_id`: Authenticated user or service account
   - `signature_hash`: SHA-256 hash of the ingested payload ensuring non-repudiation
   - `recorded_at`: UTC timestamp of ingestion
3. Ingested observations are tagged with `EpistemicStatus.VERIFIED` or `EpistemicStatus.EXTERNAL_UNVERIFIED` depending on provider accreditation status.
