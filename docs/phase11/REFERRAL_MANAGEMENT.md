# WELLWATCH PREVENT — Clinical Referral Management Specification

## 1. Governance & Role Authorization
Issuing a clinical referral to a specialist (e.g., Cardiology, Endocrinology, Medical Genetics) is a clinical act.
- **RBAC Enforcement:** In `backend/app/services/navigation_service.py`, `create_referral` verifies that the issuing user holds a clinical role (`PHYSICIAN`, `CLINICIAN`, `NURSE`, `HEALTHCARE_PROVIDER`).
- Unlicensed administrators or non-clinical coordinators cannot issue medical referrals.

---

## 2. Referral Data Architecture
- `id`: UUID.
- `person_id`: Patient link.
- `recommendation_id`: Associated recommendation.
- `specialty`: Target medical specialty (e.g. `CLINICAL_GENETICS`, `CARDIOLOGY`).
- `reason`: Clinical justification.
- `referring_provider_id`: ID of referring clinician.
- `receiving_facility`: Designated DoH-licensed healthcare facility (e.g. `Sheikh Shakhbout Medical City (SSMC)`, `Cleveland Clinic Abu Dhabi (CCAD)`).
- `status`: `PENDING`, `ACCEPTED`, `COMPLETED`, `REJECTED`.
- `appointment_id`: Link to specialist visit appointment once booked.
