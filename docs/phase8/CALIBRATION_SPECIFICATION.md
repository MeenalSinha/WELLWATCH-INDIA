# Phase 8 — Probability Calibration Specification

**Project:** WELLWATCH PREVENT — Life-Course Preventive Care Intelligence  
**Document:** Probability Calibration & Reliability Engineering  
**Implementation:** `backend/app/ml/calibration.py`  

---

## 1. Clinical Rationale for Calibration

In clinical decision support, the output of a risk model is communicated as a probability (e.g., "38% estimated 3-year risk of cardiovascular event"). 
- If a model is **uncalibrated**, a score of 0.80 might correspond to only a 40% actual event rate in the population, causing massive overtreatment, unnecessary clinical interventions, and patient anxiety.
- Conversely, underconfident predictions lead to undertreatment and missed preventive opportunities.

A model is calibrated if, for all individuals assigned a predicted probability of $p$, the empirical proportion of events is equal to $p$:
$$P(Y = 1 \mid \hat{P} = p) = p$$

---

## 2. Calibration Architecture

WELLWATCH PREVENT implements post-hoc Platt Scaling (logistic calibration) fitted exclusively on held-out validation predictions:

```python
calibrator = ProbabilityCalibrator(method="sigmoid")
calibrator.fit(uncalibrated_scores, y_validation)
calibrated_prob = calibrator.predict_proba(patient_raw_score)
```

### Metrics Tracked

1. **Brier Score:**
   $$\text{Brier} = \frac{1}{N} \sum_{i=1}^N (p_i - y_i)^2$$
   - Decomposes into reliability, resolution, and uncertainty.
   - Evaluated to ensure minimization of mean squared probabilistic error.

2. **Expected Calibration Error (ECE):**
   $$\text{ECE} = \sum_{m=1}^M \frac{|B_m|}{N} \left| \text{acc}(B_m) - \text{conf}(B_m) \right|$$
   - Samples are grouped into $M$ equally spaced confidence bins ($M=5$ default).
   - The absolute difference between bin accuracy and average bin confidence is weighted by bin size.
   - Systems with $\text{ECE} < 0.05$ demonstrate excellent clinical reliability.
