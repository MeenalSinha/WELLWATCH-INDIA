# Phase 8 — Controlled Inference Architecture

**Project:** WELLWATCH PREVENT — Life-Course Preventive Care Intelligence  
**Document:** Runtime Inference & Safety Guardrails  
**Implementation:** `backend/app/ml/inference_service.py` (`ControlledInferenceService`)  

---

## 1. Runtime Safety Invariant Checks

The `ControlledInferenceService` wraps raw model prediction routines in a 6-stage clinical safety harness:

```
[ Incoming Patient Observations ]
              |
              v
+-------------------------------+
| 1. Authorization Verification | --> Fails if model is not APPROVED_FOR_DEFINED_USE
+---------------+---------------+
                |
                v
+-------------------------------+
| 2. Input Completeness Check   | --> Abstains (ABSTAINED_INSUFFICIENT_DATA) if > 30% features missing
+---------------+---------------+
                |
                v
+-------------------------------+
| 3. Out-of-Distribution Check  | --> Abstains (ABSTAINED_OUT_OF_DISTRIBUTION) if values > 4.0 std dev
+---------------+---------------+
                |
                v
+-------------------------------+
| 4. Frozen Preprocessing       | --> Applies training medians and standardization without fitting
+---------------+---------------+
                |
                v
+-------------------------------+
| 5. Calibrated Prediction      | --> Computes true posterior probability via Platt Scaling
+---------------+---------------+
                |
                v
+-------------------------------+
| 6. Explainability Breakdown   | --> Generates local feature attribution breakdown
+-------------------------------+
```

---

## 2. Abstention Mechanisms

1. **Missing Data Abstention:**
   If more than 30% of required clinical features are missing, the system refuses to guess or impute arbitrary values. It returns `ABSTAINED_INSUFFICIENT_DATA` and issues a clinical advisory to capture baseline vital signs.
2. **Physiological Outlier Abstention:**
   If a measurement exceeds 4 standard deviations from population norms (e.g. systolic BP = 290 mmHg), standard models are invalidated. The system returns `ABSTAINED_OUT_OF_DISTRIBUTION` and triggers emergency clinical physician review.
