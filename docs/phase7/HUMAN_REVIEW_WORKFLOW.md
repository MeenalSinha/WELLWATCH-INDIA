# Phase 7 — Human Clinician Review Workflow

**Project:** WELLWATCH PREVENT — Life-Course Preventive Care Intelligence  
**Document:** Clinical Oversight & Human-in-the-Loop Workflow Specification  
**Compliance:** ADHICS v2, DoH Abu Dhabi Clinical Governance Regulations  

---

## 1. Clinician Oversight Architecture

WELLWATCH PREVENT is a clinical intelligence platform designed to augment, not replace, licensed healthcare providers. To guarantee safety and regulatory adherence, automated assessments cannot trigger high-priority patient recalls or intervention protocols without qualified medical review.

### Review Triggers

An assessment is automatically placed in the `AWAITING_REVIEW` queue when:
1. `clinical_risk_tier == ClinicalRiskTier.ELEVATED_RISK` (e.g., stage 2 hypertension, diabetes range glucose, active chronic conditions).
2. `assessment_status == AssessmentStatus.REQUIRES_REVIEW` (e.g., conflicting observations detected within 48 hours).
3. Complex multi-morbidity or contradictory care gaps are detected.

---

## 2. Review Lifecycle State Machine

```
                   +-------------------+
                   | Automated Engine  |
                   | Generates Record  |
                   +---------+---------+
                             |
         +-------------------+-------------------+
         | (Low Risk / Clean)                    | (Elevated Risk / Conflict)
         v                                       v
+-------------------+                   +-------------------+
|   NOT_REQUIRED    |                   |  AWAITING_REVIEW  |
|   (Direct Release)|                   +---------+---------+
+-------------------+                             |
                                                  | Clinician opens chart
                                                  v
                                        +-------------------+
                                        |   UNDER_REVIEW    |
                                        +---------+---------+
                                                  |
                         +------------------------+------------------------+
                         |                        |                        |
                         v                        v                        v
               +-------------------+    +-------------------+    +-------------------+
               |     APPROVED      |    |     MODIFIED      |    |     REJECTED      |
               | (Action Released) |    | (Tier / Plan Mod) |    | (Data Invalidated)|
               +-------------------+    +-------------------+    +-------------------+
```

---

## 3. RBAC & Separation of Duties

- **Authorized Reviewers:** Only users with `HEALTHCARE_PROVIDER` or `CLINICAL_SUPERVISOR` roles may review assessments.
- **Administrative Isolation:** `SYSTEM_ADMIN`, `DATA_GOVERNANCE_OFFICER`, and `COMMUNITY_HEALTH_WORKER` roles are strictly prohibited from approving or modifying clinical risk tiers.
- **Organization Scoping:** A provider may only review assessments for patients registered within their authorized healthcare organization.

---

## 4. API Endpoints for Clinical Review

- `POST /api/v1/assessments/{id}/review`:
  - Request Body:
    ```json
    {
      "new_status": "APPROVED",
      "notes": "Confirmed elevated blood pressure and glucose. Scheduled for primary care hypertension and diabetes workup."
    }
    ```
  - Validates reviewer permissions and updates `human_review_status`, `reviewed_by_user_id`, `reviewed_at`, and `reviewer_notes`.
  - Emits structured audit log recording the clinical sign-off event with correlation ID.
