# POPULATION HEALTH EQUITY & ML FAIRNESS
## Demographic Parity, Disparity Analysis & Algorithmic Guardrails

**Project:** WELLWATCH PREVENT — Life-Course Preventive Care Intelligence  
**Target Market:** Abu Dhabi, United Arab Emirates  
**Compliance Standard:** UAE Council for Artificial Intelligence Ethics Guidelines  

---

## 1. Equity Evaluation Framework

Population health intelligence must detect and highlight healthcare access disparities rather than masking them behind high aggregate averages.

### Benchmark Selection
- **Governed Benchmark:** `AD_CENTRAL` (Abu Dhabi Central Region).
- Rationale: High concentration of accredited medical centers, robust clinical staffing, and mature screening infrastructure establish an empirical baseline for attainable preventive care coverage.

### Disparity Ratio Calculation

$$\text{Disparity Ratio}(s) = \frac{\text{Coverage Rate}(s)}{\text{Coverage Rate}(\text{AD\_CENTRAL})}$$

- **Equitable Parity:** $0.80 \le \text{Ratio} \le 1.25$.
- **Adverse Disparity:** $\text{Ratio} < 0.80$ (Under-screened; triggers resource reallocation).
- **Over-Surveillance / Outlier:** $\text{Ratio} > 1.25$.

---

## 2. Machine Learning Fairness Guardrail

To prevent automated clinical decision-support models from perpetuating or exacerbating demographic inequities, the Population Analytics Engine enforces strict algorithmic safety guardrails:

1. **Ground-Truth Clinical Label Mandate:**
   - Algorithmic fairness audits strictly evaluate against epistemically verified clinical outcomes (`epistemic_status = 'VERIFIED'`).
   - Synthetic labels, simulated test records, and hallucinated scores are explicitly barred from production fairness audits.

2. **Disparate Impact Ratio Evaluation:**
   - Evaluates whether positive prediction rates (e.g. recommendation generation) across gender and geographic subgroups satisfy the Four-Fifths Rule ($0.80 \le \text{DIR} \le 1.25$).

3. **Data Completeness by Subgroup:**
   - Tracks whether certain demographic cohorts suffer from higher rates of missing observations, which could lead to algorithmic under-diagnosis.
