# Phase 8 — Model Training Specification

**Project:** WELLWATCH PREVENT — Life-Course Preventive Care Intelligence  
**Document:** Clinical Model Training & Preprocessing Protocol  
**Target Outcome:** 3-Year Cardiometabolic Risk Indicator (`CARDIOMETABOLIC_ONSET_3YR`)  

---

## 1. Predictive Target Formalization

The primary clinical target formalized in `backend/app/ml/target_definitions.py` is the **3-Year Cardiometabolic Risk Indicator**:
- **Target Endpoint:** New onset of Stage 2 Hypertension (>= 140/90 mmHg) or Impaired Fasting Glucose / Type 2 Diabetes (>= 126 mg/dL) within 36 months of baseline screening.
- **Inclusion Criteria:** Adults aged 18 to 70 with complete baseline screening records and without baseline documented myocardial infarction, stroke, or end-stage renal disease.
- **Authoritative Standard:** Aligned with ADPHC Cardiovascular and Metabolic Disease Prevention Standards and DoH Primary Care Clinical Guidelines.

---

## 2. Leakage-Resistant Preprocessing Pipeline

Implemented in `backend/app/ml/preprocessing.py` (`LeakageResistantPreprocessor`):
1. **Fit Phase (Training Split Only):**
   - Imputation statistics (medians) are calculated strictly on training observations.
   - Normalization parameters (mean, standard deviation) are learned strictly on training observations.
   - Distribution bounds are extracted for physiological sanity and OOD detection.
2. **Transform Phase (Test & Production Inference):**
   - Applies frozen training parameters without computing any sample-dependent statistics.
   - Normalizes numeric inputs: `z = (x - mean_train) / std_train`.
3. **Out-of-Distribution Guard:**
   - Evaluates z-score distance against training distributions. If any vital sign deviates by > 4.0 standard deviations from population norms, the inference pipeline flags the record for algorithmic abstention.

---

## 3. Estimator Selection & Training Protocol

- **Primary Clinical Model:** Regularized Logistic Regression with L2 penalty (`C=1.0`), solver `lbfgs`.
- **Secondary Benchmark Model:** Gradient Boosted Decision Trees / Random Forest.
- **Validation Scheme:** 5-Fold Stratified Patient-Level Cross-Validation.
- **Seed & Reproducibility:** Fixed random seed (42) enforced across all split and training routines.
