# WELLWATCH PREVENT — Clinical Governance & Requirements
## Clinical Governance Framework and Rule Specification Standards

**Jurisdiction:** Abu Dhabi, United Arab Emirates  
**Authoritative Authorities:** Department of Health Abu Dhabi (DoH), Abu Dhabi Public Health Centre (ADPHC)  
**Status:** CLINICAL GOVERNANCE BASELINE  

---

## 1. Clinical Governance Principles

1. **Clinical Decision Support, Not Autonomous Diagnosis:** WELLWATCH PREVENT operates strictly as a preventive intelligence and clinical decision-support platform. It does not issue autonomous medical diagnoses, modify pharmaceutical regimens, or override licensed medical professionals.
2. **Absolute Traceability to Authoritative Guidelines:** Every preventive pathway rule, screening interval, and eligibility condition must cite an authoritative, verifiable healthcare guideline issued by DoH Abu Dhabi, ADPHC, or international evidence-based bodies recognized in the UAE.
3. **No Unverified Clinical Inventions:** If a clinical protocol or screening interval cannot be confirmed against current DoH published documentation, it must be explicitly labeled `AWAITING_CLINICAL_VERIFICATION`. Engineers must never invent clinical thresholds or screening frequencies.
4. **Version-Controlled Clinical Rules:** Clinical rules must be stored as versioned, immutable records with explicit effective dates, review dates, and approval signatures from a designated Clinical Advisory Authority.

---

## 2. Clinical Rule Metadata Schema

Every preventive pathway rule implemented in future phases must conform to the following schema:

```json
{
  "rule_id": "DOH-PED-WCV-06M-V1",
  "rule_title": "6-Month Pediatric Well-Child Developmental Screening",
  "rule_description": "Evaluates eligibility and timeliness for the standard 6-month developmental and growth surveillance visit in infants.",
  "intended_population": "Infants aged 5 months 15 days to 7 months 15 days residing in Abu Dhabi",
  "clinical_source": "Department of Health Abu Dhabi - Child Health Screening & Vaccination Guidelines",
  "source_version": "2024.1",
  "applicable_jurisdiction": "Emirate of Abu Dhabi, UAE",
  "effective_date": "2024-01-01",
  "review_date": "2025-12-31",
  "approval_status": "APPROVED",
  "responsible_authority": "Abu Dhabi Public Health Centre (ADPHC)",
  "verification_status": "VERIFIED_OFFICIAL",
  "parameters": {
    "target_age_months": 6,
    "early_window_days": 15,
    "late_window_days": 45,
    "required_screenings": ["Growth Biometrics", "Developmental Milestones", "Vision Red Reflex", "Immunization Review"]
  },
  "audit_history": [
    {
      "timestamp": "2026-11-17T12:00:00Z",
      "author": "Clinical Governance Lead",
      "action": "BASELINE_CODIFICATION",
      "notes": "Codified from DoH official guidelines."
    }
  ]
}
```

---

## 3. Authoritative Clinical Guideline References

### 3.1 Pediatric Well-Child Visit Standards
- **Primary Source:** Department of Health Abu Dhabi (DoH) / Abu Dhabi Public Health Centre (ADPHC) — *Child Health and Screening Guidelines*.
- **Screening Schedule:**
  - Newborn (Hospital discharge / 1st week)
  - 1 Month
  - 2 Months
  - 4 Months
  - 6 Months
  - 9 Months
  - 12 Months (1 Year)
  - 18 Months
  - 24 Months (2 Years)
  - Annual visits (Years 3, 4, 5, 6).
- **Core Assessments:** Anthropometric z-scores (weight-for-age, length-for-age, head circumference), neurodevelopmental milestone screening, sensory surveillance (vision and hearing), vaccination schedule synchronization.

### 3.2 Adult Comprehensive Screening (IFHAS)
- **Primary Source:** Department of Health Abu Dhabi (DoH) — *IFHAS Periodic Comprehensive Health Assessment Program*.
- **Target Population:** UAE citizens and residents aged 18 and older.
- **Screening Interval:** Every 2 years (nominal) or annual (for individuals with identified moderate-to-high cardiovascular/metabolic risk).
- **Core Clinical Components:**
  - Cardiovascular Risk Assessment (Framingham / QRISK3 adapted, blood pressure, lipid profile).
  - Diabetes and Metabolic Syndrome Screening (Fasting blood glucose, HbA1c, waist circumference).
  - Colorectal Cancer Screening (FIT test / colonoscopy based on age >= 40 or elevated family risk).
  - Breast Cancer Screening (Mammography based on age >= 40 or high familial risk).
  - Cervical Cancer Screening (Pap smear / HPV DNA testing based on DoH guidelines).
  - Lifestyle, mental well-being, and tobacco cessation counseling.

### 3.3 Genomic Interpretation & Preventive Integration
- **Primary Source:** Emirati Genome Program (EGP) / UAE National Genome Strategy.
- **Guideline Standard:** ACMG (American College of Medical Genetics and Genomics) secondary findings recommendations (v3.2) as adopted by DoH genomics committees.
- **Scope in WellWatch Prevent:** The platform ingests *only authorized, clinical-grade interpreted genomic summary reports* (e.g. pathogenic / likely pathogenic variants in high-penetrance cardiovascular or cancer genes such as *BRCA1/2*, *LDLR*). The platform **never performs raw FASTQ/BAM alignment or unvalidated variant interpretation**.

---

## 4. Clinical Verification Status Register

| Domain / Pathway | Source Document | Authority | Verification Status | Action |
|---|---|---|---|---|
| Pediatric Schedule (0-6y) | DoH Child Health Guidelines 2024 | DoH / ADPHC | **VERIFIED_OFFICIAL** | Baseline codified for Phase 1. |
| IFHAS Adult Eligibility | DoH IFHAS Program Specs 2023/24 | DoH | **VERIFIED_OFFICIAL** | Baseline codified for Phase 1. |
| Colorectal Cancer Screening | DoH Cancer Screening Standards | ADPHC | **VERIFIED_OFFICIAL** | Incorporated into IFHAS pathway rules. |
| Breast Cancer Screening | DoH Cancer Screening Standards | ADPHC | **VERIFIED_OFFICIAL** | Incorporated into IFHAS pathway rules. |
| Genomic Secondary Risk | UAE National Genome Strategy | EGP / DoH | **AWAITING_FORMAL_SPEC** | Adapter stub ready; requires formal clinical signoff before Phase 8. |

---

## 5. Clinical Safety Invariants
1. A patient with missing historical screening records must be classified as `UNKNOWN_PREVENTIVE_STATUS` rather than `UP_TO_DATE`.
2. Critical clinical thresholds (e.g. Systolic BP >= 180 mmHg, Diastolic BP >= 120 mmHg, Fasting Glucose >= 200 mg/dL) must trigger immediate **Urgent Clinical Alert** banners, bypassing routine batch outreach.
3. Every recommendation rendered in the user interface must display the issuing clinical guideline identifier and version.
