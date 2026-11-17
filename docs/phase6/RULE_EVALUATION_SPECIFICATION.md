# WELLWATCH PREVENT
## Rule Evaluation & Explainability Specification (Phase 6)

**Document ID:** SPEC-PH6-03  
**Target Market:** Abu Dhabi, United Arab Emirates  
**Status:** IMPLEMENTED & VERIFIED  

---

## 1. Structured Evaluation Contract

Every pathway evaluation executed by `POST /api/v1/pathways/{id}/evaluate-all/{patient_id}` returns a structured explanation:

```json
{
  "patient_id": "c1f7b7a0-...",
  "pathway_id": "ADPHC-ADULT-IFHAS-V1",
  "pathway_name": "Abu Dhabi Adult Comprehensive Preventive Screening Programme (IFHAS)",
  "pathway_version": "2.1.0",
  "clinical_authority": "Abu Dhabi Public Health Centre (ADPHC)",
  "guideline_reference": "ADPHC IFHAS Standards for Comprehensive Periodic Health Examination",
  "eligibility": {
    "status": "ELIGIBLE",
    "evaluation_date": "2026-11-17",
    "age_evaluated": { "years": 45, "months": 543 },
    "is_eligible": true,
    "explanation": "Eligible for preventive pathway."
  },
  "care_gaps": [
    {
      "milestone_id": "IFHAS-CARDIOVASCULAR",
      "milestone_name": "Cardiovascular Risk & Hypertension Screening",
      "status": "DUE",
      "last_screening_date": "2022-12-01",
      "months_since": 47,
      "reason": "Previous screening was 47m ago, exceeding the 36m cycle interval. Repeat screening is due.",
      "components": ["Blood Pressure (Systolic & Diastolic)", "Heart Rate", "BMI & Waist Circumference", "Atherosclerotic CVD Risk Calculation"]
    }
  ]
}
```

## 2. Explainability Questions Answered
1. **What was determined?** Explicit status (`ELIGIBLE`, `COMPLETED`, `DUE`, `OVERDUE`, `POTENTIALLY_OVERDUE`).
2. **Why was it determined?** Clear plain-language reason code referencing exact dates and intervals.
3. **What clinical guideline justifies this?** Full authority name and reference standard provided in response.
