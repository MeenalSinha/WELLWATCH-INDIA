# Phase 7 — Preventive Risk Intelligence Architecture

**Project:** WELLWATCH PREVENT — Life-Course Preventive Care Intelligence  
**Document:** System Architecture & Integration  
**Regulatory Context:** Abu Dhabi Department of Health (DoH), ADPHC, ADHICS v2, UAE Federal Law No. 2 of 2019  

---

## 1. Executive Overview

Preventive healthcare cannot rely on isolated risk predictions or opaque algorithmic scoring. A patient does not benefit from a raw "82% cardiovascular risk" score if the system cannot explain *why*, cannot identify whether their mandatory periodic screening is overdue, and fails to state that recent lab measurements are missing or conflicting.

**Phase 7 implements the Explainable Preventive Risk Intelligence System.**

It establishes an integrated clinical intelligence engine that synthesizes:
1. **Longitudinal Clinical Data (Phase 4):** Verified LOINC observations, active ICD-10 clinical conditions, family medical history, and epistemic quality flags.
2. **Life-Course Patient Profiles (Phase 5):** Age, developmental stage (Pediatric Well-Child vs. Adult IFHAS), and household context.
3. **Preventive Pathway Engine (Phase 6):** Deterministic evaluation of screening program eligibility, milestone due dates, and care gaps.
4. **Epistemic Data Quality & Uncertainty:** Explicit tracking of data completeness, measurement freshness (>365 days), and cross-source conflicts.
5. **Human-in-the-Loop Review Workflow:** Mandatory escalation of high-risk cases or conflicting records to licensed clinicians.

---

## 2. Architectural Pipeline

```
                                      +------------------------------------+
                                      | Longitudinal Patient Record        |
                                      | - LOINC Observations               |
                                      | - ICD-10 Conditions                |
                                      | - Family History                   |
                                      +-----------------+------------------+
                                                        |
                                                        v
+------------------------------------+       +-----------------------------+
| Abu Dhabi Pathway Rules            | ----> | Preventive Assessment       |
| - Pediatric Well-Child (0-72m)     |       | Service                     |
| - Adult IFHAS (Periodic Exam)      |       +--------------+--------------+
+------------------------------------+                      |
                                                            v
                                             +-----------------------------+
                                             | 1. Clinical Risk Evaluation |
                                             | 2. Pathway Eligibility      |
                                             | 3. Care-Gap Detection       |
                                             | 4. Data Quality Audit       |
                                             +--------------+--------------+
                                                            |
                                                            v
                                             +-----------------------------+
                                             | Immutable Assessment Record |
                                             | - Risk Tier                 |
                                             | - Explainability Factors    |
                                             | - Data Limitations          |
                                             | - Human Review Queue        |
                                             +-----------------------------+
```

---

## 3. Core Design Principles

1. **Epistemic Honesty & Abstention:**
   When baseline measurements are absent, the system does not invent a "low risk" score or guess default normal values. It assigns `INSUFFICIENT_DATA` risk tier and flags status as `NEEDS_DATA`.

2. **Full Provenance & Traceability:**
   Every clinical factor cited in the assessment includes:
   - The LOINC / ICD-10 code.
   - The verified canonical value and units.
   - The primary data source provenance reference.
   - The authoritative Abu Dhabi clinical guideline citation (e.g., DoH Primary Care Hypertension Guidelines, ADPHC Adult Health Standards).

3. **Immutable Snapshots:**
   Preventive assessments are persisted as versioned, immutable historical records. An assessment represents the clinical reality at a specific point in time and is never retroactively mutated by later observations.

4. **Clinician Governance:**
   When an assessment identifies `ELEVATED_RISK` or detects `CONFLICTING_MEASUREMENTS`, it enters `AWAITING_REVIEW` status, preventing unreviewed automated notifications to patients.

---

## 4. Integration with Prior & Subsequent Phases

- **Consumes from Phase 4:** `Person`, `ClinicalObservation`, `ClinicalCondition`, `FamilyHistory`, `DataProvenance`, and `DataQualityService`.
- **Consumes from Phase 5:** `PatientProfile`, population classifications (`PEDIATRIC`, `ADULT`, `GERIATRIC`).
- **Consumes from Phase 6:** `PathwayEngine`, versioned pathway definitions (`DOH-PEDIATRIC-WELLCHILD-V1`, `ADPHC-ADULT-IFHAS-V1`).
- **Prepares for Phase 8:** Provides the structured baseline evaluation context into which validated ML models and authorized genomic summaries are calibrated and incorporated without destabilizing deterministic clinical safety.
