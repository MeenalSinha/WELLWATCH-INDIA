# WELLWATCH PREVENT — Machine Learning Model Monitoring & Drift Governance
**Scope:** Continuous Model Observability, Drift Detection, and Calibration Maintenance  
**Monitored Models:** MOD-CVD-01 (10-Year ASCVD), MOD-DIAB-02 (Diabetes Triage), MOD-ATTEND-03 (Attendance)  
**Standard:** ISO/IEC 42001 (Artificial Intelligence Management System)

---

## 1. Monitoring Metrics & Automated Telemetry

```
┌────────────────────────────────────────────────────────────────────────┐
│                   CONTINUOUS MODEL TELEMETRY                           │
├────────────────────────────────────────────────────────────────────────┤
│ 1. Covariate Shift (Population-level biomarker distribution shifts)   │
│ 2. Concept Drift (Prediction calibration vs observed clinical events)  │
│ 3. Prediction Distribution (Shift in risk category proportions)       │
│ 4. Missingness & Epistemic Uncertainty Rates                           │
│ 5. Subgroup Fairness & Disparate Impact Ratio Monitoring              │
└────────────────────────────────────────────────────────────────────────┘
```

---

## 2. Drift Detection Triggers & Thresholds

| Telemetry Dimension | Metric / Statistical Test | Alert Threshold | Action Required |
| :--- | :--- | :--- | :--- |
| **Input Covariate Shift** | Population Stability Index (PSI) | $\text{PSI} > 0.20$ (Significant shift) | Re-evaluate feature distributions; investigate intake bias |
| **CVD Risk Calibration** | Brier Score Drift | Delta $> +0.03$ from baseline | Retune calibration curve against recent cohort data |
| **Attendance Prediction** | Area Under Curve (AUROC) | AUROC $< 0.68$ over 30 days | Re-train model with updated seasonal attendance data |
| **Regional Fairness** | Disparate Impact Ratio (DIR) | $\text{DIR} < 0.80$ (Four-Fifths violation) | Automated alert to Public Health Admin; investigate clinic access |

---

## 3. Retraining & Recalibration Governance
1. **Zero Unsupervised Autonomous Retraining:** Models are NEVER retrained or deployed automatically without clinical validation.
2. **Offline Re-benchmarking:** If drift thresholds are exceeded, the ML engineering team retrains the model offline using cross-validation.
3. **Clinical Validation Gate:** Retrained models must be appraised by the Clinical Safety Board before promotion to production.
