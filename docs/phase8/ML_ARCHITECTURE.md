# Phase 8 — Machine Learning Intelligence Architecture

**Project:** WELLWATCH PREVENT — Life-Course Preventive Care Intelligence  
**Document:** Machine Learning System Architecture & Safety Guardrails  
**Regulatory Standards:** DoH Abu Dhabi Health Data Policy, UAE AI Ethics Framework, ADHICS v2  

---

## 1. Executive Overview

Machine learning in clinical preventive health requires higher governance standards than general recommendation systems. Uncalibrated predictions, data leakage, and unvalidated models trained on synthetic datasets directly endanger patient safety, trigger alarm fatigue, and violate healthcare regulations.

**Phase 8 introduces the Governed Clinical Machine Learning Subsystem.**

It transitions WELLWATCH PREVENT from legacy exploratory notebooks to a production-grade ML architecture comprising:
1. **Dataset Governance & Isolation:** Strict patient-level cohort splitting preventing cross-split data leakage.
2. **Leakage-Resistant Preprocessing:** Freezing imputers, encoders, and scalers exclusively on training splits.
3. **Probability Calibration:** Post-hoc Platt scaling and isotonic regression guaranteeing that predicted probabilities match observed empirical clinical frequencies.
4. **Feature Attribution Explainability:** Local decomposition of individual predictions into transparent, weighted risk-increasing and protective factors.
5. **Model Registry & Quarantine:** Cryptographic verification, lifecycle status management (`EXPERIMENTAL` -> `APPROVED_FOR_DEFINED_USE`), and permanent quarantine of legacy synthetic models.
6. **Controlled Inference Engine:** Automated out-of-distribution detection, missingness bounds, and algorithmic abstention.

---

## 2. Architecture Diagram

```
+-----------------------------------------------------------------------------------+
|                            CLINICAL DATA PLATFORM                                 |
| - Longitudinal LOINC Observations                                                 |
| - ICD-10 Conditions & Diagnoses                                                   |
| - Family History & Genetic Findings                                               |
+-----------------------------------------+-----------------------------------------+
                                          |
                                          v
+-----------------------------------------------------------------------------------+
|                        DATASET GOVERNANCE LAYER (ml/)                             |
| - Patient-Level Cohort Split (0% patient overlap)                                 |
| - Provenance Tracking & Data Lineage                                              |
| - Synthetic vs Real Data Segregation                                              |
+-----------------------------------------+-----------------------------------------+
                                          |
                      +-------------------+-------------------+
                      | Train Cohort                          | Test / Inference Cohort
                      v                                       v
+-------------------------------------------+   +-----------------------------------+
| LEAKAGE-RESISTANT PREPROCESSOR            |   | CONTROLLED INFERENCE SERVICE      |
| - Compute training medians, means, stds   |-->| - Frozen training parameters      |
| - Learn physiological normal ranges       |   | - OOD Check (> 4 std dev)         |
+---------------------+---------------------+   | - Missingness Check (> 30% gap)   |
                      |                         +-----------------+-----------------+
                      v                                           |
+-------------------------------------------+                     v
| ESTIMATOR & PROBABILITY CALIBRATION       |       +-------------------------------+
| - Logistic Regression / Gradient Boosting |       | MODEL REGISTRY AUTHORIZATION  |
| - Platt Scaling (Sigmoid Calibrator)      |       | - Status: APPROVED_FOR_USE    |
| - Brier Score & ECE Optimization          |       | - Synthetic Quarantine Check  |
+---------------------+---------------------+       +---------------+---------------+
                      |                                             |
                      +----------------------+----------------------+
                                             |
                                             v
                      +---------------------------------------------+
                      | EXPLAINABLE INFERENCE OUTPUT                |
                      | - Calibrated Risk Probability [0.0, 1.0]    |
                      | - Stratified Risk Tier                      |
                      | - Local Feature Attribution Breakdown       |
                      | - Top Risk Elevating & Protective Factors   |
                      | - Clinical Abstention when Uncertain        |
                      +---------------------------------------------+
```

---

## 3. Core Safety Safeguards

1. **Zero Data Leakage:** Preprocessing statistics are NEVER computed on combined cohorts. Test and inference data use frozen training distribution values.
2. **Algorithmic Abstention:** When critical baseline features are absent (>30% missing) or physiological values are physiologically implausible (>4 standard deviations from population norms), the inference engine abstains from generating a score, returning `ABSTAINED_INSUFFICIENT_DATA` or `ABSTAINED_OUT_OF_DISTRIBUTION`.
3. **Calibrated Probabilities:** Raw classification logits or tree margins are never presented as true risk probabilities without verified sigmoid or isotonic calibration.
4. **Transparent Explainability:** Every prediction produces ranked feature attributions linking numeric features to clinical concepts.
