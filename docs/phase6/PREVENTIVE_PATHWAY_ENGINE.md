# WELLWATCH PREVENT
## Abu Dhabi Preventive Pathway Engine (Phase 6)

**Document ID:** ARCH-PH6-01  
**Target Market:** Abu Dhabi, United Arab Emirates  
**Status:** IMPLEMENTED & VERIFIED  

---

## 1. Engine Architecture
The Abu Dhabi Preventive Pathway Engine evaluates an individual's eligibility for official preventive health programs and detects confirmed or potential care gaps:

```text
                  [Patient Demographic & Clinical Records]
                                     |
                                     v
                  +--------------------------------------+
                  |    ABU DHABI PREVENTIVE ENGINE       |
                  |                                      |
                  |   +------------------------------+   |
                  |   |      Eligibility Engine      |   |
                  |   |  (Program Qualification)     |   |
                  |   +--------------+---------------+   |
                  |                  |                   |
                  |                  v                   |
                  |   +------------------------------+   |
                  |   |       Care-Gap Engine        |   |
                  |   |   (Milestone Assessment)     |   |
                  |   +--------------+---------------+   |
                  |                  |                   |
                  |                  v                   |
                  |   +------------------------------+   |
                  |   |     Explanation Generator    |   |
                  |   | (Evidence & Source Lineage)  |   |
                  |   +------------------------------+   |
                  +--------------------------------------+
                                     |
                                     v
              [Structured Explainable Assessment Output]
```

---

## 2. Core Operational Principles

1. **Strict Separation of Eligibility from Care Gaps:**
   - Eligibility answers: *Does this citizen or child qualify for this preventive program based on age, gender, and coverage?*
   - Care Gaps answer: *Based on verified records, is an approved preventive visit or screening overdue, scheduled, completed, or uncertain?*
   - An individual may be `ELIGIBLE` but have all screenings `COMPLETED`. Conflating eligibility with overdue gaps is prohibited.
2. **Deterministic Reproducibility:**
   - Given the identical patient state, pathway rule version, and evaluation date, the engine produces byte-for-byte identical outputs.
   - Explicit evaluation dates are supported for retrospective cohort audits and back-testing.
3. **No Guessing or Invented Rules:**
   - Every rule is derived from published Department of Health Abu Dhabi (DoH) and Abu Dhabi Public Health Centre (ADPHC) guidelines.
   - Missing historical records generate `POTENTIALLY_OVERDUE` or `NEEDS_DATA`, never a false confirmed overdue status.
