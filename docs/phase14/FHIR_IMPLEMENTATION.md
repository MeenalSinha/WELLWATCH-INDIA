# HL7 FHIR R4 IMPLEMENTATION SPECIFICATION
## Phase 14 Profile Conformance & Serialization

**Project:** WELLWATCH PREVENT — Life-Course Preventive Care Intelligence  
**Target Market:** Abu Dhabi, United Arab Emirates  
**Compliance Standard:** HL7 FHIR Release 4 (R4), DOH Abu Dhabi FHIR Implementation Guide  

---

## 1. Supported FHIR R4 Resources

### 1.1 `Patient`
- **Profile:** `http://hl7.org/fhir/StructureDefinition/Patient`
- **Identifiers:**
  - Emirates ID: System `http://fhir.doh.gov.ae/identifier/emirates-id`, value matching regex `^784-[0-9]{4}-[0-9]{7}-[0-9]{1}$`.
  - Local MRN: System `http://fhir.doh.gov.ae/identifier/local-mrn`.
- **Telecom:** Phone (`system: phone, use: mobile`) and Email (`system: email, use: home`).
- **Demographics:** Given name, family name, gender (`male`, `female`, `other`, `unknown`), date of birth (`YYYY-MM-DD`).

### 1.2 `Observation`
- **Profile:** `http://hl7.org/fhir/StructureDefinition/Observation`
- **Status:** `final`, `amended`, `preliminary`.
- **Code:** Mapped to canonical LOINC coding systems (e.g. `15074-8` for Blood Glucose, `8480-6`/`8462-4` for Blood Pressure, `39156-5` for BMI, `4548-4` for HbA1c).
- **ValueQuantity:** Numerical value with canonical UCUM units (`mg/dL`, `mm[Hg]`, `kg/m2`, `%`).
- **Subject:** Reference to `Patient/{patient_id}`.

### 1.3 `Encounter`
- **Profile:** `http://hl7.org/fhir/StructureDefinition/Encounter`
- **Class:** Ambulatory (`AMB`), Home Health (`HH`), or Community (`COMM`).
- **Type:** SNOMED CT coded concepts (e.g. `171149006` for Well Child Check, `410620009` for Comprehensive IFHAS Health Screening).
- **Period:** Start and end timestamps in UTC ISO 8601 format.

### 1.4 `Bundle`
- **Profile:** `http://hl7.org/fhir/StructureDefinition/Bundle`
- **Types Supported:** `transaction`, `collection`, `batch-response`.
- **Integrity Total:** Matches the total number of resource entries in the package.
