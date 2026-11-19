# WELLWATCH PREVENT — Clinical Safety Hazard Register (ISO 14971 / DCB0129)
**System:** WELLWATCH PREVENT — Life-Course Preventive Care Intelligence  
**Standard Compliance:** ISO 14971:2019 (Medical Device Risk Management), DCB0129 (Clinical Risk Management in Health IT)  
**Jurisdiction:** Department of Health (DoH) Abu Dhabi & Abu Dhabi Public Health Centre (ADPHC)  
**Safety Lead:** Clinical Safety Officer (CSO) & Senior Clinical Architect  
**Review Status:** Baselined (Phase 15 Verification)

---

## 1. Risk Evaluation Matrix

### Severity Classification
- **Catastrophic (5):** Death or permanent severe injury (e.g. fatal anaphylaxis, missed lethal malignancy).
- **Critical (4):** Permanent impairment or life-threatening situation (e.g. acute coronary syndrome missed).
- **Serious (3):** Reversible injury requiring medical intervention (e.g. acute hypoglycemia, delayed screening).
- **Minor (2):** Inconvenience, slight temporary discomfort, or minor redundant test.
- **Negligible (1):** No clinical consequence.

### Likelihood Classification
- **Frequent (5):** Occurs routinely (> 1 in 100 encounters).
- **Probable (4):** Occurs occasionally (1 in 100 to 1 in 1,000 encounters).
- **Remote (3):** Could occur (1 in 1,000 to 1 in 10,000 encounters).
- **Improbable (2):** Unlikely to occur (1 in 10,000 to 1 in 100,000 encounters).
- **Incredible (1):** Virtually impossible (< 1 in 100,000 encounters).

---

## 2. Hazard Register (HAZ-01 through HAZ-14)

| Hazard ID | Trigger / Hazardous Event | Potential Clinical Harm | Pre-Risk (S x L) | Engineered Safety Barriers & Mitigations | Post-Risk (S x L) | Verification Test | Status |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **HAZ-01** | False negative screening triage for pediatric developmental delay. | Delayed intervention for autism or speech delay during neuroplastic window. | 4 x 3 = 12 (High) | Strict epistemic tracking: missing parent milestone questionnaire yields `INCOMPLETE_DATA`, never `NORMAL`. Mandated clinician review. | 3 x 1 = 3 (Low) | `test_pathway_engine.py` | Controlled |
| **HAZ-02** | Administrative user closes a clinical care gap without clinical encounter. | Overdue diabetic nephropathy screening marked "done"; patient suffers renal deterioration. | 4 x 4 = 16 (Extreme) | Licensure RBAC: System Admin cannot close care gaps. Strict requirement for verified `evidence_source` and `evidence_id`. | 4 x 1 = 4 (Low) | `test_clinical_safety.py` | Controlled |
| **HAZ-03** | Cross-patient evidence linkage during bulk care-gap closure. | Patient A's colonoscopy result applied to Patient B; Patient B's colorectal tumor missed. | 5 x 2 = 10 (High) | Cross-resource validation: `evidence.patient_id` must strictly match `recommendation.person_id` and target person. | 5 x 1 = 5 (Low) | `test_clinical_safety.py` | Controlled |
| **HAZ-04** | Corrupt offline synchronization of community health worker vitals. | Severe hypertension (210/120 mmHg) dropped due to network timeout or race condition. | 4 x 3 = 12 (High) | Client-side durable IndexedDB outbox; cryptographic idempotency hash; server-side transactional batch audit. | 4 x 1 = 4 (Low) | `test_offline_chw.py` | Controlled |
| **HAZ-05** | Unchecked machine learning drift causing under-referral in Al Dhafra region. | Geographic disparity in cardiovascular risk management; increased myocardial infarction. | 4 x 3 = 12 (High) | Disparate impact ratio monitoring (> 0.80 enforced); demographic parity checks across Abu Dhabi regions. | 4 x 1 = 4 (Low) | `test_population_intelligence.py` | Controlled |
| **HAZ-06** | Epistemic overconfidence: ML model predicts low CVD risk on sparse vitals. | Patient with unrecorded family history of premature CAD suffers cardiac arrest. | 4 x 3 = 12 (High) | Model outputs epistemic confidence; sparse feature vectors trigger explicit missing-data warnings. | 4 x 1 = 4 (Low) | `test_clinical_safety.py` | Controlled |
| **HAZ-07** | Automated prescription assertion by clinical decision support. | Adverse drug reaction or toxic dosing due to unmonitored automated prescription. | 5 x 3 = 15 (Extreme) | CDS Architectural Abstention: WELLWATCH PREVENT only outputs preventive recommendations, never direct drug prescriptions. | 5 x 1 = 5 (Low) | `test_recommendations.py` | Controlled |
| **HAZ-08** | Ingestion of malformed FHIR Observation lacking LOINC code or unit. | Uncalibrated lab value (e.g. HbA1c in mmol/mol interpreted as %) triggers erroneous pathways. | 3 x 4 = 12 (High) | Strict FHIR parser validation: unmapped LOINC codes or ambiguous units are quarantined as `UNPARSED_RESOURCE`. | 3 x 1 = 3 (Low) | `test_interoperability.py` | Controlled |
| **HAZ-09** | Cultural outreach notification delivered with wrong language or tone. | Patient alienated, ignores invitation for IFHAS breast cancer screening. | 3 x 3 = 9 (Medium) | Explicit `preferred_language` resolution (Arabic/English); verified culturally validated outreach templates. | 2 x 1 = 2 (Low) | `test_outreach.py` | Controlled |
| **HAZ-10** | Database corruption or disk loss during primary PostgreSQL failure. | Loss of longitudinal screening history for pediatric vaccination tracking. | 4 x 2 = 8 (Medium) | Automated snapshot engine with SHA-256 tamper verification, point-in-time recovery runbook, RTO < 5s. | 4 x 1 = 4 (Low) | `test_backup_restore.py` | Controlled |
| **HAZ-11** | High concurrent load causing silent API timeouts during screening sync. | Community screening encounter lost in transit without user alert. | 3 x 3 = 9 (Medium) | Batch limit capped at 25 items; HTTP 413 on overflow; client retry backoff queue in durable IndexedDB. | 2 x 1 = 2 (Low) | `test_performance_reliability.py` | Controlled |
| **HAZ-12** | Premature closure of missed-care follow-up task. | Patient missed mammogram; coordinator closes ticket without contacting patient. | 4 x 3 = 12 (High) | Minimum resolution note enforcement (min 5 characters) and optional linked clinical evidence record. | 3 x 1 = 3 (Low) | `test_care_navigation.py` | Controlled |
| **HAZ-13** | Statistical disclosure in small-cell population screening analytics. | Re-identification of rare pediatric conditions in Al Ain public health dashboards. | 3 x 3 = 9 (Medium) | Secondary and primary small-cell suppression (< 5 individuals suppressed as `<5` with complementary masking). | 2 x 1 = 2 (Low) | `test_population_intelligence.py` | Controlled |
| **HAZ-14** | Premature integration of experimental Multi-Omics/PRS without clinical trials. | Untested genetic risk score causing inappropriate prophylactic intervention. | 4 x 3 = 12 (High) | Architectural decoupling: PRS/Multi-omics relegated to future post-Phase-17 research roadmap (`FUTURE_ROADMAP.md`). | 4 x 1 = 4 (Low) | Architectural Barrier | Controlled |

---

## 3. Clinical Safety Officer Conclusion
All 14 identified hazards have engineered software mitigations and verification tests. The post-mitigation residual risk across all hazards resides within the **Acceptable (Broadly Acceptable)** or **ALARP (As Low As Reasonably Practicable)** regions under ISO 14971 standards.
