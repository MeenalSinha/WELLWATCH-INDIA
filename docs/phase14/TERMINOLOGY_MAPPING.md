# GOVERNED TERMINOLOGY MAPPING SPECIFICATION
## Phase 14 Clinical Vocabulary Translation

**Project:** WELLWATCH PREVENT — Life-Course Preventive Care Intelligence  
**Target Market:** Abu Dhabi, United Arab Emirates  
**Governing Standard:** LOINC, SNOMED CT, ICD-10-CM, UCUM  

---

## 1. Governed Code System Mappings

Governed in `configs/terminology_mappings.json`:

### 1.1 LOINC Observations & Laboratory Tests
| Internal Code | LOINC Code | Standard Display Name | UCUM Unit |
| :--- | :--- | :--- | :--- |
| `BLOOD_PRESSURE_SYSTOLIC` | `8480-6` | Systolic blood pressure | `mm[Hg]` |
| `BLOOD_PRESSURE_DIASTOLIC`| `8462-4` | Diastolic blood pressure | `mm[Hg]` |
| `GLUCOSE_FASTING` | `15074-8` | Glucose [Mass/volume] in Blood | `mg/dL` |
| `HBA1C` | `4548-4` | Hemoglobin A1c/Hemoglobin.total in Blood | `%` |
| `BMI` | `39156-5` | Body mass index (BMI) [Ratio] | `kg/m2` |
| `CHOLESTEROL_TOTAL` | `2093-3` | Cholesterol [Mass/volume] in Serum or Plasma | `mg/dL` |

### 1.2 SNOMED CT Clinical Encounters & Findings
| Internal Code | SNOMED CT Code | Standard Concept Display |
| :--- | :--- | :--- |
| `WELL_CHILD_VISIT` | `171149006` | Child health surveillance examination |
| `IFHAS_COMPREHENSIVE`| `410620009` | Comprehensive adult screening examination |
| `COMMUNITY_VITALS_CHECK`| `386053000` | Evaluation procedure of community vital signs |

### 1.3 ICD-10 Diagnoses & Encounter Reasons
| Internal Code | ICD-10 Code | Standard Diagnostic Display |
| :--- | :--- | :--- |
| `DX_HYPERTENSION` | `I10` | Essential (primary) hypertension |
| `DX_TYPE_2_DIABETES` | `E11` | Type 2 diabetes mellitus |
| `ENC_ROUTINE_CHILD` | `Z00.129` | Encounter for routine child health examination without abnormal findings |
