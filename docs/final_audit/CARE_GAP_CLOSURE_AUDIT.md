# CARE GAP CLOSURE EVIDENCE & CLINICAL LICENSURE AUDIT

**Project:** WELLWATCH PREVENT — Life-Course Preventive Care Intelligence  
**Component:** `NavigationService.verify_and_close_care_gap` (Module G)  
**Regulatory Standards:** DoH Abu Dhabi Healthcare Quality / Clinical Practice Standards  
**Status:** FULLY COMPLIANT

---

## 1. Objective & Mandate

Care-gap closure is the definitive clinical endpoint in the preventive care life-course. Falsely marking care gaps as closed without verified clinical evidence poses severe medical risks (missed cancers, undetected cardiovascular events, incomplete immunizations).
This audit verifies Module G: Verified Care-Gap Closure under strict clinical licensure and database-backed evidence lookup.

---

## 2. Security & Clinical Governance Invariants

### 2.1 Strict Clinical Licensure (Exclusion of System Admin)
- **Invariant:** Clinical care-gap closure requires medical licensure.
- **Enforcement:** `NavigationService.verify_and_close_care_gap` verifies that `current_user.role == UserRole.HEALTHCARE_PROVIDER`.
- System Administrators, Care Coordinators, and Community Health Workers are rejected with HTTP 403:
  `Role 'SYSTEM_ADMIN' is not authorized to certify clinical care gap closure. Only licensed healthcare providers may attest.`

### 2.2 Database Evidence Lookup
Care-gap closure strictly rejects fabricated string identifiers:
- `SCREENING_RECORD` / `ENCOUNTER`: The system looks up `ScreeningEncounter` or `Encounter` in the database. If not found, closure is rejected with HTTP 400.
- `OBSERVATION` / `LAB_RESULT`: Queries `ClinicalObservation`. If absent, rejected with HTTP 400.
- `REFERRAL_OUTCOME`: Queries `Referral` and verifies `referral.status == ReferralStatus.COMPLETED`.
- `CLINICAL_NOTE`: Requires non-empty attestation notes (`len(notes) >= 5`).

### 2.3 Cross-Patient Boundary Protection
- Even if a valid evidence ID exists in the database, the engine verifies that the record's `patient_id` matches the target patient (`person_id` or `person.user_id`).
- Using Patient B's encounter to close Patient A's care gap is rejected with HTTP 400: `Evidence record belongs to a different patient.`

### 2.4 Cross-Organization Boundary Protection
- Providers cannot attest to care gaps for patients enrolled at a different primary organization unless granted explicit multi-facility credentials.
- Cross-organization attestation attempts are rejected with HTTP 403.

---

## 3. Empirical Verification Summary

1. `test_care_gap_cannot_be_closed_without_evidence`: Missing evidence ID rejected.
2. `test_verified_care_gap_closure_completes_recommendation`: Genuine database `Encounter` successfully closes gap and marks recommendation `COMPLETED`.
3. `test_system_admin_rejected_from_closing_care_gap`: System Admin rejected with HTTP 403.
4. `test_cross_patient_evidence_rejected`: Mismatched patient evidence rejected with HTTP 400.
