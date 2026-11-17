# Phase 8 — Explainable Machine Learning Framework

**Project:** WELLWATCH PREVENT — Life-Course Preventive Care Intelligence  
**Document:** Local Feature Attribution & Clinical Interpretability  
**Implementation:** `backend/app/ml/explainability.py` (`FeatureAttributionExplainer`)  

---

## 1. Local Explainability Requirement

Under UAE National Artificial Intelligence Guidelines and Abu Dhabi Department of Health policy, clinical AI systems cannot operate as opaque black boxes. Clinicians and patients must be provided with understandable, quantitative explanations detailing:
1. Which specific physiological features elevated the patient's predicted risk.
2. Which protective factors mitigated their overall score.
3. The exact numerical contribution and observed baseline values.

---

## 2. Decomposition Mechanism

The `FeatureAttributionExplainer` decomposes an individual model prediction into an additive sum of feature contributions relative to a baseline reference value:

$$\text{Score}(x) = \phi_0 + \sum_{i=1}^D \phi_i(x)$$

Where:
- $\phi_0$ is the base expected value across the population.
- $\phi_i(x) = w_i \cdot z_i(x)$ is the individual attribution for feature $i$.
- $\phi_i > 0$ indicates a **risk-increasing factor** (e.g., elevated BMI, high systolic BP).
- $\phi_i < 0$ indicates a **protective factor** (e.g., normal blood pressure, younger age).

---

## 3. Output Schema & Clinical Translation

Each prediction produces structured explanations formatted for immediate clinical review:

```json
{
  "base_value": 0.10,
  "total_score": 0.74,
  "top_risk_increasing_factors": [
    {
      "feature": "bmi",
      "display_name": "Body Mass Index",
      "observed_value": 34.2,
      "contribution": 0.38,
      "impact": "RISK_INCREASING"
    },
    {
      "feature": "systolic_bp",
      "display_name": "Systolic Blood Pressure",
      "observed_value": 145.0,
      "contribution": 0.26,
      "impact": "RISK_INCREASING"
    }
  ],
  "top_protective_factors": [
    {
      "feature": "fasting_glucose",
      "display_name": "Fasting Blood Glucose",
      "observed_value": 88.0,
      "contribution": -0.12,
      "impact": "PROTECTIVE_OR_NEUTRAL"
    }
  ]
}
```

Clinicians can immediately confirm whether model attributions align with clinical judgment, identifying candidates for targeted lifestyle intervention or secondary screening.
