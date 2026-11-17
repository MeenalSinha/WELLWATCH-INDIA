# Phase 7 — Preventive Assessment Data & Entity Model

**Project:** WELLWATCH PREVENT — Life-Course Preventive Care Intelligence  
**Document:** Entity Model Specification  
**Component:** `backend/app/models/assessment.py`  

---

## 1. PreventiveAssessment Entity Schema

The `PreventiveAssessment` model captures a comprehensive clinical assessment event. It is registered in the database via Alembic migration `003_preventive_assessments.py`.

### Attributes Table

| Column Name | Type | Nullable | Description / Clinical Meaning |
|---|---|---|---|
| `id` | `VARCHAR(36)` | No | Primary Key UUID. |
| `patient_id` | `VARCHAR(36)` | No | Foreign Key to `persons.id`. The healthcare subject. |
| `pathway_id` | `VARCHAR(64)` | No | Identifier of the preventive pathway evaluated (e.g., `ADPHC-ADULT-IFHAS-V1`). |
| `assessment_date` | `DATE` | No | Effective clinical evaluation date. |
| `clinical_risk_tier` | `ENUM` | No | Tier: `LOW_RISK`, `MODERATE_RISK`, `ELEVATED_RISK`, `INSUFFICIENT_DATA`. |
| `assessment_status` | `ENUM` | No | Status: `COMPLETED`, `REQUIRES_REVIEW`, `NEEDS_DATA`, `SUPERSEDED`. |
| `human_review_status`| `ENUM` | No | Review: `NOT_REQUIRED`, `AWAITING_REVIEW`, `UNDER_REVIEW`, `APPROVED`, `MODIFIED`, `REJECTED`. |
| `eligibility_summary`| `TEXT` (JSON) | No | Snapshot of pathway eligibility determination, age, criteria match, and reasoning. |
| `care_gaps_summary` | `TEXT` (JSON) | No | List of overdue, due, and completed screening milestones. |
| `risk_factors` | `TEXT` (JSON) | No | Array of contributing clinical indicators with values, LOINC codes, and DoH citations. |
| `data_limitations` | `TEXT` (JSON) | No | Epistemic limitations (missing baseline vitals, outdated labs, conflicting records). |
| `recommended_action`| `TEXT` | No | Plain-language recommended preventive action for patient navigation. |
| `reviewed_by_user_id`| `VARCHAR(36)`| Yes | User ID of reviewing physician/nurse. |
| `reviewed_at` | `DATETIME` | Yes | Timestamp of clinical sign-off. |
| `reviewer_notes` | `TEXT` | Yes | Clinical rationale or intervention orders entered by the reviewer. |
| `created_at` | `DATETIME` | No | Audit timestamp of creation. |

---

## 2. Enums and State Spaces

### ClinicalRiskTier
- `LOW_RISK`: All verified clinical parameters are within optimal or normal physiological ranges; no active chronic diagnoses.
- `MODERATE_RISK`: Pre-hypertension (systolic 130–139 mmHg or diastolic 80–89 mmHg), prediabetes (FBG 100–125 mg/dL), Class I obesity (BMI 30–34.9 kg/m²), or positive family history of early-onset disease.
- `ELEVATED_RISK`: Stage 2 hypertension (systolic >= 140 mmHg or diastolic >= 90 mmHg), fasting glucose >= 126 mg/dL, Class II+ obesity (BMI >= 35.0 kg/m²), or documented active chronic condition (e.g., Type 2 Diabetes, ASCVD).
- `INSUFFICIENT_DATA`: Missing core baseline observations necessary to form a valid clinical assessment.

### AssessmentStatus
- `COMPLETED`: Assessment generated with complete baseline data and no conflicting observations.
- `REQUIRES_REVIEW`: Generated assessment contains data anomalies, epistemic conflicts, or elevated risks requiring clinician sign-off.
- `NEEDS_DATA`: System abstained from conclusive tiering due to missing baseline records; patient must undergo baseline measurement.
- `SUPERSEDED`: Replaced by a more recent assessment snapshot.

### HumanReviewStatus
- `NOT_REQUIRED`: Routine low-risk assessments.
- `AWAITING_REVIEW`: Automated evaluation flagged elevated risks or epistemic conflicts; placed in clinician review queue.
- `UNDER_REVIEW`: A licensed clinician has claimed the assessment for evaluation.
- `APPROVED`: Clinician verified and authorized the assessment.
- `MODIFIED`: Clinician updated the assessment tier or care-gap disposition based on clinical judgment.
- `REJECTED`: Clinician invalidated the assessment due to incorrect chart data or clinical circumstances.
