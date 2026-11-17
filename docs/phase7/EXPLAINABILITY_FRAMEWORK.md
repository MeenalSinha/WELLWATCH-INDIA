# Phase 7 — Explainability Framework

**Project:** WELLWATCH PREVENT — Life-Course Preventive Care Intelligence  
**Document:** Explainability Specification & Clinical Evidence Attribution  
**Standards:** DoH Abu Dhabi Clinical Guidelines, ADPHC Public Health Standards  

---

## 1. Principles of Explainable Clinical Intelligence

In healthcare, "black-box" risk scoring produces clinical distrust, alarm fatigue, and compliance risks under UAE Health Data Law and ADHICS v2.

Every risk classification or care-gap determination in WELLWATCH PREVENT must be:
1. **Attributable to Verified Data:** Pointing to explicit clinical measurements with known collection timestamps and provenance.
2. **Citing Authoritative Clinical Standards:** Linking directly to established clinical guidelines issued by the Abu Dhabi Department of Health (DoH) or Abu Dhabi Public Health Centre (ADPHC).
3. **Transparent in Magnitude:** Disclosing exact measurement values alongside canonical units and clinical thresholds.
4. **Comprehensible to Patients & Clinicians:** Formatted in structured, dual-level representations (clinician-grade medical detail and patient-friendly plain-language explanations).

---

## 2. Factor Attribution Schema

Each risk factor captured in `risk_factors` adheres to the following JSON structure:

```json
{
  "indicator": "STAGE_2_HYPERTENSION_SYSTOLIC",
  "code": "8480-6",
  "display": "Elevated Systolic Blood Pressure",
  "value": "145.0 mmHg",
  "clinical_source": "DoH Primary Care Hypertension Guidelines",
  "severity": "ELEVATED",
  "provenance_ref": "prov-uuid-5512-ab7"
}
```

### Registered Clinical Indicators & Thresholds

| Indicator ID | Concept / LOINC | Qualifying Threshold | Severity | Authoritative Source Citation |
|---|---|---|---|---|
| `STAGE_2_HYPERTENSION_SYSTOLIC` | Systolic BP (`8480-6`) | >= 140.0 mmHg | `ELEVATED` | DoH Primary Care Hypertension Guidelines |
| `STAGE_1_HYPERTENSION_SYSTOLIC` | Systolic BP (`8480-6`) | 130.0 - 139.0 mmHg | `MODERATE` | DoH Primary Care Hypertension Guidelines |
| `STAGE_2_HYPERTENSION_DIASTOLIC` | Diastolic BP (`8462-4`) | >= 90.0 mmHg | `ELEVATED` | DoH Primary Care Hypertension Guidelines |
| `STAGE_1_HYPERTENSION_DIASTOLIC` | Diastolic BP (`8462-4`) | 80.0 - 89.0 mmHg | `MODERATE` | DoH Primary Care Hypertension Guidelines |
| `DIABETES_RANGE_FASTING_GLUCOSE` | Fasting Glucose (`2339-0`)| >= 126.0 mg/dL | `ELEVATED` | DoH Diabetes Clinical Practice Guidelines |
| `IMPAIRED_FASTING_GLUCOSE_PREDIABETES` | Fasting Glucose (`2339-0`)| 100.0 - 125.0 mg/dL | `MODERATE` | DoH Diabetes Clinical Practice Guidelines |
| `OBESITY_CLASS_2` | BMI (`39156-5`) | >= 35.0 kg/m² | `ELEVATED` | ADPHC Adult Health Standards |
| `OBESITY_CLASS_1` | BMI (`39156-5`) | 30.0 - 34.9 kg/m² | `MODERATE` | ADPHC Adult Health Standards |
| `CHRONIC_CONDITION_*` | ICD-10 Diagnosis | Documented Active Status | `ELEVATED` | Verified Clinical Diagnosis |
| `HEREDITARY_RISK` | Family History | 1st-Degree Relative Onset | `MODERATE` | Patient Reported Family History |

---

## 3. Tier Aggregation Logic

Risk tiers are aggregated deterministically:
- **`ELEVATED_RISK`**: Presence of >= 1 `ELEVATED` indicator OR >= 3 `MODERATE` indicators.
- **`MODERATE_RISK`**: Presence of 1 to 2 `MODERATE` indicators and 0 `ELEVATED` indicators.
- **`LOW_RISK`**: Complete baseline data present; 0 `MODERATE` and 0 `ELEVATED` indicators.
- **`INSUFFICIENT_DATA`**: Core baseline physiological vitals (BP, glucose, BMI) absent.
