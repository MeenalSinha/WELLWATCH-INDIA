# WELLWATCH PREVENT
## Adult Profile Specification (Phase 5)

**Document ID:** SPEC-PH5-02  
**Target Market:** Abu Dhabi, United Arab Emirates  
**Status:** IMPLEMENTED & VERIFIED  

---

## 1. Domain Overview
The adult profile manages preventive intelligence for individuals aged 18 to 64:
- **Clinical Vitals:** Systolic and diastolic blood pressure, heart rate, height, weight, dynamically calculated BMI, fasting glucose, HbA1c, and lipid panels (total cholesterol, HDL, LDL, triglycerides).
- **Chronic Conditions:** Documented active or resolved conditions coded via ICD-10.
- **Lifestyle Indicators:** Structured behavioral risk factors including tobacco smoking status, physical activity levels, and dietary habits.
- **IFHAS Screening History:** Comprehensive preventive screening records aligned with the Abu Dhabi Public Health Centre (ADPHC) IFHAS programme.
- **Hereditary Risk Profile:** First-degree and second-degree family medical histories.

## 2. API Schema
Endpoint: `GET /api/v1/patients/{id}/adult`
Response structure:
```json
{
  "person": {
    "id": "person-uuid",
    "full_name": "Mansoor Al Mazrouei",
    "date_of_birth": "1981-08-20",
    "age_years": 45,
    "gender": "MALE",
    "population_classification": "ADULT"
  },
  "latest_vitals": {
    "8480-6": {
      "name": "Systolic Blood Pressure",
      "value": 124.0,
      "unit": "mmHg",
      "date": "2026-10-15T09:30:00Z",
      "epistemic_status": "VERIFIED"
    }
  },
  "chronic_conditions": [],
  "lifestyle_indicators": {},
  "screening_history": [],
  "family_history": [],
  "completeness": {
    "score": 40.0,
    "status": "PARTIALLY_COMPLETE"
  }
}
```
