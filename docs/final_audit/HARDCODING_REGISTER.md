# Hardcoding & Configuration Audit Register
**Project:** WELLWATCH PREVENT — Life-Course Preventive Care Intelligence  
**Target Market:** Abu Dhabi, United Arab Emirates  
**Compliance Standard:** DoH Abu Dhabi Health Data Standards, ADPHC Public Health Standards, ACMG SF v3.2

---

## 1. Executive Summary
This register documents the systematic identification, decoupling, and resolution of hardcoded clinical logic, thresholds, weights, screening intervals, and actionability mappings across the WELLWATCH PREVENT codebase. All business rules and clinical criteria have been extracted into validated, version-controlled JSON configuration files backed by strict JSON schemas.

---

## 2. Hardcoding Resolution Matrix

| Component | Previous Hardcoded State | Decoupled Architecture | Authoritative Clinical Source | Verification Status |
| :--- | :--- | :--- | :--- | :--- |
| **Cardiovascular Risk Thresholds** | In-code Python constants (Systolic >= 140, Diastolic >= 90) in `preventive_assessment_service.py` | `configs/clinical_rules/clinical_risk_indicators.json` governed by `configs/schemas/clinical_risk_indicators.schema.json` | DoH Primary Care Hypertension Guidelines 2024 | **VERIFIED & RESOLVED** |
| **Metabolic Risk Thresholds** | In-code fasting glucose (> 126 mg/dL, >= 100 mg/dL) | `configs/clinical_rules/clinical_risk_indicators.json` | DoH Diabetes Clinical Practice Guidelines | **VERIFIED & RESOLVED** |
| **Anthropometric Thresholds** | In-code BMI thresholds (>= 30.0 kg/m², >= 25.0 kg/m²) | `configs/clinical_rules/clinical_risk_indicators.json` | ADPHC Adult Health Standards | **VERIFIED & RESOLVED** |
| **Genomic Actionability Mappings** | Static hardcoded dictionary in `genomic_service.py` for `LDLR`, `APOB`, `BRCA1`, `BRCA2`, `MLH1`, `MSH2` | `configs/genomics/actionable_guidelines.json` governed by `configs/schemas/genomics.schema.json` | ACMG SF v3.2 / Emirati Genome Program (EGP) / NCCN Standards | **VERIFIED & RESOLVED** |
| **Data Quality & Staleness Windows** | Class-level integer literals (`recency_threshold_days = 365`) in `data_quality_service.py` | Loaded dynamically via `settings.OBSERVATION_RECENCY_DAYS` in `core/config.py` | ADHICS v2 Quality Assurance Standards | **VERIFIED & RESOLVED** |
| **Pediatric Milestone Windows** | In-code month intervals (0, 2, 4, 6, 9, 12, 18, 24, 36, 48, 60, 72) | Versioned pathway JSON `DOH-PEDIATRIC-WELLCHILD-V1` in `configs/pathways/` | DoH Standard for Well-Child Healthcare | **VERIFIED & RESOLVED** |
| **Adult Screening Cycles** | In-code 36-month cycle intervals for IFHAS screening | Versioned pathway JSON `ADPHC-ADULT-IFHAS-V1` in `configs/pathways/` | ADPHC Comprehensive Adult Screening Guidelines | **VERIFIED & RESOLVED** |

---

## 3. Schema Enforcement
Every clinical configuration is validated against its respective JSON Schema upon application boot and service instantiation:
1. `configs/schemas/clinical_risk_indicators.schema.json`: Validates concept codes (LOINC), clinical sources, threshold operators, and severities.
2. `configs/schemas/genomics.schema.json`: Validates actionable gene symbols, condition associations, required pathogenicity classifications, and evidence sources.

---

## 4. Invariant Verification
- [x] Zero hardcoded clinical thresholds in Python source code.
- [x] Zero magic numbers in risk evaluation pipelines.
- [x] All clinical guidelines explicitly cited with authoritative issuing bodies.
- [x] Dynamic reload supported without service redeployment.
