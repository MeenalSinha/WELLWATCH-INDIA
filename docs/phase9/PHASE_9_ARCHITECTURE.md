# WELLWATCH PREVENT — Phase 9 Architecture Specification
**Personalized Preventive Recommendations Engine**

## 1. System Overview & Mandate
Phase 9 establishes the intelligence core that converts patient longitudinal observations, population health risk tiers, identified care gaps, and genomic intelligence into personalized, actionable, evidence-backed preventive interventions.

In compliance with the Department of Health (DoH) Abu Dhabi and Abu Dhabi Public Health Centre (ADPHC) guidelines, the Recommendation Engine acts strictly as a clinical decision-support intelligence platform:
- **Zero Hallucination:** Every recommendation is generated through schema-validated, version-governed rule matrices (`configs/recommendation_rules/*.json`).
- **Human Clinical Oversight:** Recommendations are advisory. System administrators and unverified personnel cannot approve recommendations; licensed clinical providers review and approve them before they are active.
- **Explainable Traceability:** Every recommendation persists explicit clinical rationales, source guidelines, rule identifiers, rule versions, and epistemic uncertainty statuses.

---

## 2. Component Architecture

```
+--------------------------------------------------------------------------------+
|                            Phase 9 Architecture                                |
+--------------------------------------------------------------------------------+
|  [ Longitudinal Clinical Database ]      [ Preventive Assessments & Gaps ]    |
|                     |                                     |                    |
|                     v                                     v                    |
|       +---------------------------------------------------------+              |
|       |               RecommendationService                     |              |
|       +---------------------------------------------------------+              |
|         | Loads external JSON rule definitions                  |              |
|         | Evaluates demographic eligibility (Pediatric vs Adult)|              |
|         | Cross-checks care gaps & clinical triggers             |              |
|         | Epistemic safety abstention under incomplete data     |              |
|         | Enforces deduplication across existing active items    |              |
|         +-------------------------------------------------------+              |
|                                    |                                           |
|                     +--------------+--------------+                            |
|                     v                             v                            |
|        [ PreventiveRecommendation ]      [ PreventiveCarePlan ]                |
|        - Lifecycle Statuses              - Multi-item aggregation              |
|        - Clinical Rationale              - Target milestone dates              |
|        - Epistemic Uncertainty           - Patient / Clinical Goals            |
|                     |                                                          |
|                     v                                                          |
|        [ RecommendationReview ]                                                |
|        - Clinical Provider Signoff                                             |
|        - Human Review Audit Trail                                              |
+--------------------------------------------------------------------------------+
```

---

## 3. Data Models & Relational Persistence
All models are declared using SQLAlchemy 2.0 and persisted in the relational database (`005_recommendations_outreach_navigation.py`):
1. **`PreventiveRecommendation` (`preventive_recommendations` table):**
   - `id`: UUID primary key.
   - `organization_id`: Multi-tenant boundary.
   - `person_id`: Reference to `persons.id`.
   - `assessment_id`: Reference to originating `preventive_assessments.id`.
   - `pathway_id`: Associated clinical pathway (e.g. `WELL_CHILD_VISIT`, `IFHAS_COMPREHENSIVE_SCREENING`, `CARDIOMETABOLIC_INTERVENTION`, `ACMG_ACTIONABLE_GENOMIC`).
   - `care_gap_id`: Link to closed-loop care gap.
   - `category`: Enum (`SCREENING`, `LIFESTYLE`, `VACCINATION`, `CLINICAL_FOLLOW_UP`, `SPECIALIST_REFERRAL`, `DIAGNOSTIC_WORKUP`, `DATA_COLLECTION`).
   - `priority`: Enum (`EMERGENT`, `URGENT`, `ROUTINE`, `OPTIONAL`).
   - `lifecycle_status`: Enum (`GENERATED`, `PENDING_REVIEW`, `ACTIVE`, `OFFERED`, `ACCEPTED`, `SCHEDULED`, `IN_PROGRESS`, `COMPLETED`, `DECLINED`, `CANCELLED`, `SUPERSEDED`).
   - `recommended_action`: Plain text clinical direction.
   - `clinical_rationale`: Evidence-backed reasoning.
   - `supporting_evidence`: JSON payload containing LOINC codes, risk percentiles, and thresholds.
   - `source_guideline`: Authority (e.g., `ADPHC Well-Child Standard 2024`, `DoH IFHAS Protocol v2.1`, `ACMG SF v3.2`).
   - `uncertainty_status`: Epistemic status (`ROBUST`, `PROVISIONAL`, `INSUFFICIENT_DATA_ABSTAINED`).
   - `rule_id` & `rule_version`: Strict governance lineage.
   - `requires_human_review`: Boolean flag.
   - `declined_reason`: Reason logged if declined by patient.

2. **`RecommendationReview` (`recommendation_reviews` table):**
   - Captures human clinical sign-off, modifications, provider ID, review decision (`APPROVED`, `MODIFIED`, `REJECTED`, `INFO_REQUESTED`), and timestamped rationale.

3. **`PreventiveCarePlan` & `CarePlanItem` (`preventive_care_plans` & `care_plan_items` tables):**
   - Aggregates multiple recommendations into structured clinical care programs with overall goals, status, and milestone dates.

---

## 4. Key Security & Compliance Controls
- **Organization Isolation:** Multi-tenancy strictly enforced on every recommendation query.
- **Role-Based Access Control:** Clinical reviews restricted to `PHYSICIAN`, `CLINICIAN`, `NURSE`, and `HEALTHCARE_PROVIDER`.
- **Wadeema's Law (UAE Federal Law No. 3 of 2016):** Pediatric recommendations respect legal guardian authority.
