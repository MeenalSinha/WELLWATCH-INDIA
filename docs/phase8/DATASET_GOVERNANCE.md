# Phase 8 — Dataset Governance & Split Specification

**Project:** WELLWATCH PREVENT — Life-Course Preventive Care Intelligence  
**Document:** Dataset Governance & Split Policy  
**Compliance Standards:** UAE Federal Health Data Law No. 2 of 2019, ADHICS v2, DoH Abu Dhabi Health Data Policy  

---

## 1. Principles of Clinical Dataset Governance

In clinical machine learning, evaluating models on contaminated or leaked test sets produces inflated performance claims that collapse upon real-world deployment. Furthermore, models trained on synthetic patient data without clinical trial validation can learn spurious associations that pose direct hazards to patients.

**WELLWATCH PREVENT enforces three foundational governance rules:**
1. **Patient-Level Split Isolation:**
   Datasets with longitudinal patient records must NEVER be split by row or random encounter sampling. The partition into training, validation, and test sets is performed strictly at the unique patient identifier level (`split_cohort_by_patient`). Zero patient overlap is mathematically verified.
2. **Synthetic Data Quarantine:**
   Synthetic datasets are strictly quarantined for unit testing and pipeline prototyping. They are prohibited from receiving clinical regulatory clearance (`DENIED - SYNTHETIC_DATA_ONLY`) and cannot be used for clinical decision support.
3. **Audit Lineage & Provenance:**
   Every dataset carries immutable metadata including source facility, extraction criteria, population classification, feature schemas, and regulatory approval identifiers.

---

## 2. Patient-Level Splitting Invariant

```python
# Invariant verified by test_dataset_split_patient_level_isolation:
train_patients = set(train_df["patient_id"])
test_patients = set(test_df["patient_id"])
assert len(train_patients.intersection(test_patients)) == 0
```

When an individual patient has multiple clinical visits or screenings over time:
- All historical observations for Patient A remain in the training set.
- All historical observations for Patient B remain in the test set.
- This prevents the model from "memorizing" an individual's baseline physiology and falsely predicting their future risk.

---

## 3. Synthetic vs. Real Data Governance Matrix

| Dataset Category | Approved Environment | Intended Use | Clinical Inference Allowed? |
|---|---|---|---|
| Synthetic Demonstration (e.g. `DATA-SYNTHETIC-WELLWATCH-2025`) | Development, CI/CD, Unit Testing | API validation, schema testing | **NO (STRICTLY PROHIBITED)** |
| De-identified Clinical EHR Cohort | Staging / Validation | Model training, calibration, back-testing | **RESEARCH ONLY** |
| Authorized Clinical Production Feed | Production | Real-time preventive risk intelligence | **YES (WITH ADPHC CLEARANCE)** |

---

## 4. Implementation Reference

The governance logic is operationalized in:
- Model: `backend/app/ml/dataset_governance.py` (`DatasetGovernance`, `DatasetMetadata`)
- Tests: `backend/tests/test_ml_pipeline.py::test_dataset_split_patient_level_isolation`, `test_synthetic_dataset_governance_audit`
