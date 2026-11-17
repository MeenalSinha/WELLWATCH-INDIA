# WELLWATCH PREVENT
## Patient Intelligence Architecture (Phase 5)

**Document ID:** ARCH-PH5-01  
**Target Market:** Abu Dhabi, United Arab Emirates  
**Status:** IMPLEMENTED & VERIFIED  

---

## 1. Life-Course Population Health Paradigm
Traditional preventive applications deploy one-size-fits-all questionnaires. WELLWATCH PREVENT partitions life-course intelligence into distinct developmental and physiological domains while maintaining a single, consistent relational database foundation:

1. **Pediatric Preventive Intelligence (0 to 17 years; core focus 0–6y):**
   - Centered around developmental milestone observations (motor, speech, cognitive).
   - Milestone tracking aligned with official Abu Dhabi Department of Health Well-Child Visits.
   - Enforces legal guardianship relationships complying with UAE Child Rights Law (Wadeema's Law).
2. **Adult Comprehensive Screening (18 to 64 years):**
   - Centered around physiological vitals (blood pressure, heart rate, BMI, fasting glucose, lipid panels).
   - Cardiovascular and chronic metabolic disease prevention aligned with ADPHC IFHAS guidelines.
   - Lifestyle behavior tracking (tobacco smoking, physical activity, nutrition).
3. **Geriatric Preventive Care (65+ years):**
   - Target screening intervals for mobility, cognitive health, and sensory impairment.

---

## 2. Real API Services & Data Flow

```text
  [HTTP Client / Frontend]
             |
             v
  [GET /api/v1/patients] ---> [PersonService.list_persons]
                                   | (Filters by Org Boundary & Classification)
                                   v
                             [PostgreSQL/SQLite]
                                   |
             +---------------------+---------------------+
             |                                           |
             v                                           v
[GET /api/v1/patients/{id}/pediatric]       [GET /api/v1/patients/{id}/adult]
   - Guardianship verification                - Latest canonical vitals
   - Well-Child screening history             - Chronic conditions (ICD-10)
   - Developmental milestones                 - IFHAS screening history
   - Completeness score                       - Completeness score
```

---

## 3. Data Completeness vs. Health Quality
The platform strictly separates **Data Completeness** from **Clinical Health Status**:
- A patient with a 100% complete record may have severe chronic disease (e.g. Type 2 Diabetes with elevated HbA1c).
- A patient with a 20% complete record may be clinically healthy but lacks preventive encounter history.
- The platform never presents complete data as equivalent to good health.
