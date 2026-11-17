# Phase 7 — Uncertainty & Abstention Engine

**Project:** WELLWATCH PREVENT — Life-Course Preventive Care Intelligence  
**Document:** Algorithmic Abstention & Data Limitation Specification  
**Clinical Safety Principle:** Abstain Rather Than Hallucinate  

---

## 1. The Clinical Danger of False Certainty

In healthcare decision support, a false negative is frequently far more hazardous than an explicit admission of missing data. Traditional risk scoring systems often impute missing clinical values with population means (e.g., assuming missing blood pressure is 120/80 mmHg or missing glucose is 90 mg/dL). This practice creates an illusion of health, masking severe underlying risks and delaying vital screening.

**WELLWATCH PREVENT operates under a strict Algorithmic Abstention Policy:**
If critical diagnostic evidence is missing, outdated, or conflicting, the engine **must abstain** from assigning a definitive low or moderate risk tier. Instead, it assigns `INSUFFICIENT_DATA` and issues a `NEEDS_DATA` recommendation to obtain baseline clinical observations.

---

## 2. Epistemic Limitation Taxonomy

The `data_limitations` attribute on each assessment surfaces concrete data issues:

```json
[
  {
    "code": "CORE_VITALS",
    "issue": "MISSING_BASELINE",
    "detail": "No baseline blood pressure, glucose, or BMI measurements on file."
  },
  {
    "code": "8480-6",
    "name": "Systolic Blood Pressure",
    "issue": "CONFLICTING_MEASUREMENTS",
    "detail": "Recent measurements from different sources disagree beyond clinical threshold (160 vs 118 mmHg within 48h)."
  },
  {
    "code": "2339-0",
    "name": "Fasting Blood Glucose",
    "issue": "OUTDATED_MEASUREMENT",
    "detail": "Observation collected on 2024-03-10 exceeds 12-month recency window."
  }
]
```

### Classification Rules

1. **`MISSING_BASELINE`:**
   - Evaluated during risk evaluation. If an adult patient has zero blood pressure, fasting glucose, or BMI measurements on file, the risk engine returns `INSUFFICIENT_DATA`.
   - The assessment status is set to `NEEDS_DATA`.

2. **`CONFLICTING_MEASUREMENTS`:**
   - Detected by `DataQualityService.detect_observation_conflicts`. If two verified observations for the same concept within a 48-hour window deviate by > 30%, both are tagged as `CONFLICTING`.
   - The assessment notes the discrepancy and automatically sets `assessment_status` to `REQUIRES_REVIEW` and `human_review_status` to `AWAITING_REVIEW`.

3. **`OUTDATED_MEASUREMENT`:**
   - Detected when the most recent observation exceeds the standard clinical freshness window (12 months for routine physiological vitals).
   - The system utilizes the historical value for trend awareness but warns that updated testing is necessary.

---

## 3. Separation of Care-Gap Uncertainty from Confirmed Overdue

Under Phase 6 and Phase 7, missing screening history **never generates a false "CONFIRMED OVERDUE" care gap**.
- If a patient has an unknown screening date, the milestone status is evaluated as `NEEDS_DATA` or `POTENTIALLY_OVERDUE`.
- It remains distinct from a confirmed `OVERDUE` milestone where the historical screening date is known and has elapsed past the cycle interval.
