# POPULATION SEGMENTATION SPECIFICATION
## Life-Course Cohort Stratification

**Project:** WELLWATCH PREVENT — Life-Course Preventive Care Intelligence  
**Target Market:** Abu Dhabi, United Arab Emirates  

---

## 1. Life-Course Segment Definitions

| Population Cohort | Age Definition | Statutory / Clinical Standard | Primary Preventive Focus |
| :--- | :--- | :--- | :--- |
| **Pediatric** | Age < 18 years | UAE Federal Law No. 3 (Wadeema's Law), ADPHC Pediatric Guidelines | Growth monitoring, developmental milestone evaluation, pediatric immunization recall, early sensory screening. |
| **Adult** | 18 <= Age < 65 years | ADPHC IFHAS Standard 2026 | Cardiometabolic risk stratification (diabetes, hypertension, dyslipidemia), cancer screening, smoking cessation. |
| **Geriatric** | Age >= 65 years | ADPHC Healthy Aging Guidelines | Functional mobility assessment, fall risk evaluation, polypharmacy review, cognitive screening. |
| **Maternal** | Pregnancy / Postpartum | DoH Maternal & Child Health Framework | Antenatal gestational screening, gestational diabetes, pre-eclampsia risk assessment, postpartum depression screening. |

---

## 2. Pediatric Developmental Stage Rules

Segmented automatically based on computed age in days and months from `date_of_birth`:

```python
def classify_developmental_stage(dob: date) -> str:
    age_days = (date.today() - dob).days
    age_years = age_days // 365
    
    if age_days <= 28:
        return "NEWBORN"
    elif age_days < 365:
        return "INFANCY"
    elif age_years < 3:
        return "TODDLER"
    elif age_years < 6:
        return "PRESCHOOL"
    elif age_years < 12:
        return "SCHOOL_AGE"
    elif age_years < 18:
        return "ADOLESCENT"
    else:
        return "ADULT"
```

---

## 3. Adult Cardiometabolic Risk Stratification

Adults are categorized into mutually exclusive risk strata using multi-parameter clinical observation thresholds:

1. **Very High Risk:** Fasting Glucose >= 126 mg/dL OR HbA1c >= 6.5% OR Systolic BP >= 160 mmHg OR Diastolic BP >= 100 mmHg.
2. **High Risk:** Fasting Glucose 100–125 mg/dL (Impaired Fasting Glucose) OR Systolic BP 140–159 mmHg OR Diastolic BP 90–99 mmHg OR BMI >= 35.0.
3. **Moderate Risk:** Systolic BP 130–139 mmHg OR Diastolic BP 80–89 mmHg OR BMI 25.0–34.9.
4. **Low Risk:** Normotensive (BP < 120/80 mmHg), Fasting Glucose < 100 mg/dL, BMI < 25.0.
5. **Insufficient Data:** Key parameters missing or unverified (abstains from premature risk assignment under Epistemic Safety).
