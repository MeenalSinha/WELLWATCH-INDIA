# WELLWATCH PREVENT — Recommendation Engine Specification

## 1. Engine Execution Mechanics
The `RecommendationService` is responsible for evaluating a patient's current clinical profile against authorized rule catalogs and synthesizing recommendations.

### 1.1 Trigger Points
Recommendations are triggered through:
1. **Assessment Completion:** When a `PreventiveAssessment` is executed, identifying care gaps or elevated risk scores.
2. **Clinical Intake / Encounter:** When new clinical observations (e.g., blood pressure, HbA1c, height/weight) are submitted.
3. **Genomic Variant Ingestion:** When ACMG secondary actionable findings or polygenic scores are recorded.
4. **On-Demand Periodic Evaluation:** Explicit clinician or system request via `/api/v1/recommendations/generate/{person_id}`.

### 1.2 Evaluation Algorithm
For each patient:
```python
1. Extract age, sex, population classification (Pediatric < 18 or Adult >= 18).
2. Fetch patient's latest assessments, risk tiers, and identified care gaps.
3. If epistemic data status == INSUFFICIENT_DATA:
     Issue DATA_COLLECTION recommendation (INTAKE_ADDITIONAL_DATA) and abstain from premature clinical decisions.
4. If age < 18:
     Evaluate configs/recommendation_rules/pediatric_rules.json
5. If age >= 18:
     Evaluate configs/recommendation_rules/adult_rules.json
6. If genomic findings exist:
     Evaluate configs/recommendation_rules/genomic_rules.json
7. Query existing active recommendations for person.
     Exclude duplicates based on rule_id if existing is not CANCELLED or SUPERSEDED.
8. Persist new recommendations with lifecycle_status = PENDING_REVIEW (if human review required) or ACTIVE.
9. Log tamper-evident audit event via AuditService.
```

---

## 2. Epistemic Uncertainty & Abstention
In healthcare intelligence, missing data must never be imputed into false normality. When critical physiological indicators (e.g., blood pressure, glucose) are absent:
- The engine marks `uncertainty_status = "INSUFFICIENT_DATA_ABSTAINED"`.
- It suppresses high-intensity clinical directives that require accurate baseline data.
- It generates an epistemic intake recommendation (`DATA_COLLECTION`) directing the patient to baseline screening.
