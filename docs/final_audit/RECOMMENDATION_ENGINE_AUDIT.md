# RECOMMENDATION ENGINE ARCHITECTURE & CLINICAL GOVERNANCE AUDIT

**Project:** WELLWATCH PREVENT — Life-Course Preventive Care Intelligence  
**Component:** `RecommendationService` & `PathwayEngine`  
**Regulatory Standards:** DoH Abu Dhabi / ADPHC Screening Guidelines  
**Status:** FULLY COMPLIANT

---

## 1. Architectural Overview

The recommendation engine translates empirical longitudinal care gaps into individualized, clinically actionable preventive recommendations. It serves as the primary clinical decision support (CDS) interface for healthcare providers and care coordinators.

```
┌─────────────────────────────────┐      ┌───────────────────────────────┐
│     Phase 6: Pathway Engine     │      │ Phase 7: Preventive Assessment│
│  (Evaluates DUE/OVERDUE Gaps)   │      │   (Clinical Risk & Factors)   │
└────────────────┬────────────────┘      └──────────────┬────────────────┘
                 │                                      │
                 └───────────────────┬──────────────────┘
                                     ▼
                   ┌───────────────────────────────────┐
                   │    Recommendation Engine Core     │
                   │  - Schema Validation              │
                   │  - Duplicate Rule ID Prevention   │
                   │  - Epistemic Abstention Pass      │
                   │  - Genomic Counselor Verification │
                   └─────────────────┬─────────────────┘
                                     ▼
                   ┌───────────────────────────────────┐
                   │     PreventiveRecommendation      │
                   │   Status: PENDING_CLINICAL_REVIEW │
                   └─────────────────┬─────────────────┘
                                     ▼
                   ┌───────────────────────────────────┐
                   │     Human Clinical Oversight      │
                   │   Licensed Provider Attestation   │
                   └───────────────────────────────────┘
```

---

## 2. Invariants & Clinical Safeguards

### 2.1 Rule Schema Governance
- Rules are defined across modular registries (`pediatric_rules.json`, `adult_rules.json`, `genomic_rules.json`).
- Schema validation enforces `approval_status == "APPROVED"`. Any draft or unapproved rule is rejected at engine boot.
- Global rule ID uniqueness is verified during startup; duplicate rule IDs across files raise a critical `ValueError`.

### 2.2 Integration with Upstream Phases
- **Pathway Gaps (Phase 6):** Evaluates `PathwayEngine.evaluate_care_gaps` against versioned clinical pathways (`configs/pathways/pediatric_well_child.json`, `configs/pathways/adult_ifhas.json`).
- **Clinical Assessments (Phase 7):** Consumes `PreventiveAssessment` records. Contributing factors (e.g. `ELEVATED_BLOOD_PRESSURE`) trigger targeted monitoring recommendations.
- **Genomic Guardrails (Phase 8):** Only findings reviewed and approved by a licensed genetic counselor (`GENETIC_COUNSELOR_VERIFIED`) with `PATHOGENIC` or `LIKELY_PATHOGENIC` tiering can trigger genomic recommendations. VUS variants are strictly prevented from generating recommendations.

### 2.3 Epistemic Uncertainty & Abstention
If baseline vitals or historical screening records are missing, the engine refrains from asserting definitive disease risk tiers. It marks recommendations as `uncertainty_status = "INSUFFICIENT_DATA_ABSTAINED"` and recommends a baseline vitals intake appointment (`REC-ADULT-004`).

### 2.4 Autonomous Choice & Non-Harassment
When a recommendation is declined with voluntary reason documentation (`RecommendationLifecycleStatus.DECLINED`), the recommendation engine registers the status in `existing_recs` and suppresses duplicate generation during subsequent evaluations.

---

## 3. Empirical Verification Summary
- Evaluated with pediatric (`test_generate_pediatric_recommendations_well_child`) and adult IFHAS (`test_generate_adult_recommendations_ifhas_and_cardiometabolic`) journeys.
- Human clinical oversight verified: only `HEALTHCARE_PROVIDER` can approve or reject recommendations.
- Rule governance verified: duplicate rule detection and unapproved rule rejection confirmed via `test_recommendation_rule_validation_rejects_duplicates_and_unapproved`.
