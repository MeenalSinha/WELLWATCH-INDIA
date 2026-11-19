# WELLWATCH PREVENT — Pilot Workflow Specifications & SOPs
**Scope:** Clinical Standard Operating Procedures for Pilot Deployment  
**Tracks:** Pediatric Well-Child, Adult IFHAS Screening, and CHW Field Outreach  
**Version:** 1.0.0

---

## 1. Track 1: Pediatric Well-Child Workflow (Ages 0–18)

```
[Parent & Infant Arrive] ──> [Nurse Records Vitals] ──> [Parent Fills Milestones]
                                                               │
                                                               ▼
[Clinician Signs Plan] <── [Clinician Reviews CDS] <── [CDS Evaluates ADPHC Rules]
```

### Operational Steps:
1. **Intake & Vitals:** Clinic triage nurse records weight, height/length, and head circumference.
2. **Developmental Assessment:** Parent completes age-appropriate milestone checklist (or M-CHAT at 18/24 months) via patient tablet.
3. **CDS Evaluation:** Engine triggers ADPHC pediatric pathway rules; flags growth percentile deviations or milestone delays.
4. **Physician Review:** Pediatrician reviews recommendations (`REVIEWED` $\rightarrow$ `APPROVED`); schedules immunizations or orders audiometry/developmental pediatrics referral.
5. **Care Plan Activation:** Care coordinator follows up on pending vaccination appointments.

---

## 2. Track 2: Adult IFHAS Comprehensive Screening Workflow (Ages 18+)

```
[Patient Check-in] ──> [Point-of-Care Vitals & Labs] ──> [CDS Generates Risk Scores]
                                                                    │
                                                                    ▼
[Closure Evidence Recorded] <── [Navigation & Recall] <── [Clinician Approves Actions]
```

### Operational Steps:
1. **Clinical Screening:** Blood pressure, BMI, waist circumference, and fasting glucose/HbA1c recorded.
2. **Algorithmic Stratification:** 10-year CVD risk score (MOD-CVD-01) and Type 2 diabetes risk (MOD-DIAB-02) computed.
3. **Cancer Screening Triage:** Age/gender eligibility verified for colorectal (FIT kit), breast (mammography), and cervical screening.
4. **Care Gap Identification:** System flags unaddressed screening gaps (e.g. `GAP-COLORECTAL-FIT`).
5. **Verified Evidence Closure:** Once completed lab result or encounter is filed, clinician attests and closes gap via `verify_and_close_care_gap`.

---

## 3. Track 3: Community Health Worker Field Screening Workflow (Offline)

```
[CHW Downloads Tablet Cache] ──> [Field Visits in Al Dhafra] ──> [Records Vitals in Outbox]
                                                                        │
                                                                        ▼
[Conflict Triage if Discordant] <── [Server Ingestion & Audit] <── [CHW Restores Connectivity]
```

### Operational Steps:
1. **Device Preparation:** CHW logs in while on clinic Wi-Fi; system caches assigned patient roster in durable IndexedDB (`chw_outbox`, `chw_cache`).
2. **Offline Field Work:** CHW travels to rural farms/homes; records screening vitals in `/chw` offline workspace.
3. **Local Encryption:** Encrypted records stored in client browser storage with SHA-256 idempotency hash.
4. **Batch Sync:** Upon reconnecting, CHW triggers batch sync. System validates lease, verifies idempotency, checks for clinical conflicts (e.g. BP deviation $> 30\%$), and commits records.
5. **Coordinator Escalation:** Any detected clinical conflicts are surfaced in the navigation queue for physician resolution.
