# WELLWATCH PREVENT
## Data Quality & Epistemic Uncertainty Model (Phase 4)

**Document ID:** GOV-PH4-02  
**Target Market:** Abu Dhabi, United Arab Emirates  
**Status:** IMPLEMENTED & VERIFIED  

---

## 1. The 8-State Epistemic Model

Traditional electronic medical records represent missing or uncertain data simply as a nullable database column (`NULL`), conflating multiple clinically distinct scenarios.

WELLWATCH PREVENT implements a rigorous 8-state epistemic model:

| Status Code | Clinical Meaning | Example Scenario | System Action |
|:---|:---|:---|:---|
| `VERIFIED` | Measurement present, verified by authoritative source, within validity window. | Recent laboratory fasting blood glucose from accredited hospital. | Available for full clinical risk and eligibility evaluation. |
| `UNVERIFIED` | Measurement present, but lacks formal clinical accreditation or laboratory confirmation. | Self-reported weight from home bathroom scale. | Usable with flagged confidence tier; highlighted for provider review. |
| `MISSING` | Required measurement has never been recorded or is absent from available records. | Patient has no recorded lipid panel in their lifetime. | Triggers care gap investigation or information request; **does not infer high or low risk**. |
| `UNKNOWN` | Clinician explicitly asked or evaluated the factor, but patient or record was unable to determine value. | Patient does not know their paternal biological family medical history. | Prevents repetitive queries; models uncertainty explicitly. |
| `OUTDATED` | Measurement exists and was verified, but has passed its maximum clinical recency window. | Blood pressure reading taken 18 months ago (>12 months threshold). | Excluded from current acute risk calculation; triggers preventive vitals recall. |
| `CONFLICTING` | Multiple recent observations from different sources disagree beyond clinical variance limits. | Two blood pressure readings taken 2 hours apart show 110/70 vs. 175/105 mmHg (>30% variance). | Flags data conflict; mandates clinician review; abstains from automated risk categorization. |
| `NOT_APPLICABLE` | Clinical test is biologically or protocol-wise irrelevant for this patient's cohort. | Cervical cancer screening for a 4-year-old child or male adult. | Excluded from completeness metrics and care-gap evaluation. |
| `UNAUTHORIZED` | Health information exists in an external system, but user or organization lacks legal consent or authorization to access it. | Genomic risk report where patient explicitly revoked consent for preventive research use. | Access strictly denied; masked in UI. |

---

## 2. Recency and Staleness Policies

- **Clinical Vitals (Blood Pressure, Heart Rate, Glucose):**
  - Maximum recency: 365 days (12 months).
  - Older observations transition automatically from `VERIFIED` to `OUTDATED`.
- **Comprehensive Preventive Screenings (IFHAS Panels):**
  - Maximum recency: 1,095 days (36 months / 3 years).
  - Outdated screenings trigger eligibility for the next preventive screening cycle.
- **Pediatric Well-Child Visits:**
  - Evaluated according to strict age-milestone windows (0, 2, 4, 6, 9, 12, 18, 24, 36, 48, 60, 72 months).
