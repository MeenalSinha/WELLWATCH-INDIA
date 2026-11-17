# WELLWATCH PREVENT — Recommendation Rule Governance & Schema

## 1. Schema Definition & Validation
To prevent unvalidated logic or hardcoded strings from entering clinical pathways, all recommendation rules are defined as declarative JSON files adhering to `configs/schemas/recommendation_rules.schema.json`.

### Schema Structure:
- `rule_id`: Unique identifier (e.g., `PED_WELL_CHILD_001`, `ADULT_IFHAS_SCREENING_001`).
- `rule_version`: Semantic version string (`1.0.0`).
- `category`: Category enum matching `RecommendationCategory`.
- `priority`: Priority enum matching `RecommendationPriority`.
- `pathway_id`: Associated clinical pathway.
- `target_population`: Population criteria (`min_age`, `max_age`, `gender`).
- `trigger_conditions`: Deterministic evaluation triggers (e.g. `care_gap_code`, `risk_tier_threshold`, `loinc_codes`).
- `clinical_action`: Action title and description.
- `clinical_rationale`: Evidence and clinical guidelines (ADPHC, DoH Abu Dhabi, ACMG).
- `requires_human_review`: Boolean constraint.
- `target_completion_days`: Expected turnaround for care gap closure.

---

## 2. Governed Rule Catalogs
1. **Pediatric Rules (`configs/recommendation_rules/pediatric_rules.json`):**
   - `PED_WELL_CHILD_001`: Mandatory ADPHC Well-Child Visits under Wadeema's Law.
   - `PED_VACCINATION_002`: Mandatory UAE National Immunization Schedule vaccines.
2. **Adult Rules (`configs/recommendation_rules/adult_rules.json`):**
   - `ADULT_IFHAS_SCREENING_001`: Annual ADPHC IFHAS Comprehensive Screening for UAE citizens and residents.
   - `ADULT_HYPERTENSION_RECALL_002`: Blood pressure verification for Stage 1/2 HTN.
   - `ADULT_DIABETES_GLUCOSE_WORKUP_003`: Diagnostic OGTT/HbA1c for elevated random glucose or metabolic syndrome.
   - `ADULT_EPISTEMIC_INTAKE_004`: Intake recommendation when epistemic status is `INSUFFICIENT_DATA`.
3. **Genomic Rules (`configs/recommendation_rules/genomic_rules.json`):**
   - `GENOMIC_ACMG_SECONDARY_001`: Specialist genetics referral for ACMG SF v3.2 pathogenic/likely pathogenic variants.
   - `GENOMIC_FAMILIAL_HYPERCHOLESTEROLEMIA_002`: Advanced lipid workup and cascade testing for LDLR/APOB/PCSK9 variants.
