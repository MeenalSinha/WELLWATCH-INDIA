# Phase 8 — Model Registry & Governance Specification

**Project:** WELLWATCH PREVENT — Life-Course Preventive Care Intelligence  
**Document:** Model Registry Catalog & Quarantine Protocol  
**Implementation:** `backend/app/ml/model_registry.py` (`ModelRegistry`)  

---

## 1. Governance Objectives

The Model Registry provides cryptographic tracking, metadata lineage, and lifecycle gatekeeping for all machine learning models in WELLWATCH PREVENT.

It enforces a strict security invariant:
> **No machine learning model may execute within clinical care pathways or generate patient assessments unless its lifecycle status is `APPROVED_FOR_DEFINED_USE`.**

---

## 2. Model Lifecycle State Machine

```
      +-------------------+
      |   EXPERIMENTAL    |  (Development & exploration)
      +---------+---------+
                |
                v
      +-------------------+
      |   UNDER_REVIEW    |  (Clinical safety & calibration audit)
      +---------+---------+
                |
       +--------+--------+
       |                 |
       v                 v
+--------------+  +--------------+
| APPROVED_FOR |  |  SUSPENDED   |  (Performance drift or safety alert)
| DEFINED_USE  |  +--------------+
+-------+------+         |
        |                v
        +-------->+--------------+
                  |   RETIRED    |  (Permanently decommissioned)
                  +--------------+
```

### Lifecycle State Definitions

1. `EXPERIMENTAL`: Model undergoing prototyping or validation. Blocked from clinical endpoints.
2. `UNDER_REVIEW`: Submitted for multi-disciplinary evaluation by the Clinical AI Governance Committee.
3. `APPROVED_FOR_DEFINED_USE`: Authorized strictly for the clinical indications, age ranges, and settings documented in its metadata.
4. `SUSPENDED`: Temporarily halted due to population drift, sensor recalibration, or performance flags.
5. `RETIRED`: Quarantined and permanently removed from production use.

---

## 3. Legacy Model Quarantine Register

| Attribute | Legacy Model (`models/chronic_risk_model.pkl`) |
|---|---|
| **Model ID** | `LEGACY-XGBOOST-SYNTHETIC-V1` |
| **Model Name** | Legacy WELLWATCH-INDIA Chronic Risk XGBoost |
| **Lifecycle Status** | **`RETIRED` (PERMANENTLY QUARANTINED)** |
| **Dataset Provenance** | Artificially generated synthetic CSV (`data/raw_screening_data.csv`) |
| **Synthetic Flag** | `True` |
| **Clinical Clearance** | **DENIED** |
| **Quarantine Rationale** | Synthetic training dataset lacks true physiological covariance; uncalibrated probability predictions; no out-of-distribution safeguards. |
