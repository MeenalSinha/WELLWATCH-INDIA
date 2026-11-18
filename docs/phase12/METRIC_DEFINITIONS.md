# POPULATION HEALTH METRIC DEFINITIONS & DENOMINATORS
## Phase 12 Metric Dictionary

**Project:** WELLWATCH PREVENT — Life-Course Preventive Care Intelligence  
**Target Market:** Abu Dhabi, United Arab Emirates  
**Compliance Standard:** ADPHC 2026 Population Health Monitoring Framework  

---

## 1. Denominator Integrity Table

| Metric Name | Numerator Criteria | Denominator Criteria | Clinical / Epidemiological Rationale |
| :--- | :--- | :--- | :--- |
| **Screening Coverage Rate** | Persons with at least one completed screening record in reporting window (`epistemic_status != 'UNVERIFIED'`). | Persons meeting clinical eligibility guidelines for preventive screening (e.g. age-appropriate, active profile). | Prevents inflating coverage by excluding unregistered or ineligible individuals from the baseline. |
| **Confirmed Overdue Rate** | Eligible persons with verified absence of required screening beyond grace period. | Total eligible population for specific clinical pathway. | Identifies true clinical gaps requiring proactive recall outreach. |
| **Appointment Attendance Rate** | Appointments with status `ATTENDED` or `COMPLETED`. | Total booked appointments with scheduled date in reporting window. | Measures patient adherence and care navigation workflow efficiency. |
| **No-Show Rate** | Appointments marked with status `NO_SHOW`. | Total booked appointments scheduled in reporting window. | Quantifies lost clinical capacity and triggers automated community health worker follow-up. |
| **Referral Completion Rate** | Specialist referrals marked `COMPLETED` with outcome documentation. | Total referrals submitted to receiving facilities in reporting window. | Assesses secondary care transitions and specialist loop-closure. |
| **Care Gap Closure Rate** | Care gaps closed with verified `CareGapClosureEvidence` (Module G). | Total open care gaps identified by clinical guidelines. | Guarantees that care gaps cannot be closed without objective clinical proof. |
| **Guardian Authorization Rate** | Pediatric profiles linked to a verified guardian user under Wadeema's Law. | Total registered pediatric individuals (age < 18). | Ensures pediatric health data privacy compliance with UAE Federal Law No. 3. |

---

## 2. Life-Course Clinical Numerator Definitions

### 2.1 Pediatric Well-Child Screening
- **Eligible Cohort:** Persons registered with `date_of_birth` yielding `age_years < 18`.
- **Completed Well-Child Visit:** A `ScreeningRecord` or `Encounter` with `encounter_type = 'WELL_CHILD_VISIT'` or `screening_type = 'WELL_CHILD_CHECK'` and `screening_status = 'COMPLETED'`.
- **Developmental Stages:**
  - Newborn: 0–28 days.
  - Infancy: 29 days to < 1 year.
  - Toddler: 1 to < 3 years.
  - Preschool: 3 to < 6 years.
  - School Age: 6 to < 12 years.
  - Adolescent: 12 to < 18 years.

### 2.2 Adult IFHAS Comprehensive Screening
- **Eligible Cohort:** Persons registered with `date_of_birth` yielding `18 <= age_years < 65`.
- **Completed IFHAS Visit:** A `ScreeningRecord` or `Encounter` with `encounter_type = 'IFHAS_COMPREHENSIVE'` and `screening_status = 'COMPLETED'`.
- **Cardiometabolic Risk Distribution:**
  - Categorized as `LOW`, `MODERATE`, `HIGH`, `VERY_HIGH`, or `INSUFFICIENT_DATA` based on multi-parameter thresholds (Systolic BP, Diastolic BP, Fasting Glucose, BMI, Total Cholesterol, Smoking Status).

---

## 3. Disparity Ratios and Equity Formulas

For any demographic or geographic subgroup $s$ evaluated against reference benchmark $b$ (`AD_CENTRAL`):

$$\text{Disparity Ratio}(s) = \frac{\text{Coverage Rate}(s)}{\text{Coverage Rate}(b)}$$

- **Equitable Range:** $0.80 \le \text{Disparity Ratio} \le 1.25$ (Four-Fifths Rule compliance).
- **Disparity Alert:** $\text{Disparity Ratio} < 0.80$ triggers automated population health alerts for targeted outreach and CHW mobile clinic deployment.
