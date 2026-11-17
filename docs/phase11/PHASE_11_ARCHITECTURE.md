# WELLWATCH PREVENT — Phase 11 Architecture Specification
**Closed-Loop Care Navigation & Care Gap Closure Engine**

## 1. System Overview & Mandate
Phase 11 implements the clinical navigation and closed-loop care-gap closure system.
In clinical preventive medicine, an intervention is not complete merely because an outreach message was delivered or an appointment was booked.

Phase 11 enforces:
- **Closed-Loop Scheduling:** Appointments and specialist referrals linked directly to `PreventiveRecommendation` and `CarePlanItem` records.
- **Missed-Care Protocol:** Automated creation of `FollowUpTask` whenever an appointment transitions to `NO_SHOW` or `CANCELLED`.
- **Governed Clinical Referrals:** Clinical referrals require licensed clinician role authorization.
- **Evidence-Backed Care Gap Closure (Module G):** Care gaps and recommendations can **only** transition to `COMPLETED` when accompanied by persistent, verified clinical evidence (`CareGapClosureEvidence`).

---

## 2. Component Architecture

```
+--------------------------------------------------------------------------------+
|                           Phase 11 Architecture                                |
+--------------------------------------------------------------------------------+
|  [ PreventiveRecommendation ]             [ PreventiveCarePlan ]               |
|                 |                                      |                       |
|                 v                                      v                       |
|       +---------------------------------------------------------+              |
|       |                   NavigationService                     |              |
|       +---------------------------------------------------------+              |
|         | 1. Books Appointments & Issues Clinical Referrals     |              |
|         | 2. Tracks Attendance Lifecycle                        |              |
|         |    - If NO_SHOW: Auto-creates FollowUpTask            |              |
|         | 3. Manages Periodic Clinical RecallTasks              |              |
|         | 4. Module G: Enforces Verified Care-Gap Closure       |              |
|         |    - Requires Evidence Type & Diagnostic Data         |              |
|         |    - Rejects Fake / Mock / Unverified Closures        |              |
|         +-------------------------------------------------------+              |
|                                    |                                           |
|                     +--------------+--------------+                            |
|                     v                             v                            |
|             [ Appointment ]                  [ Referral ]                      |
|             - BOOKED, ATTENDED,              - PENDING, ACCEPTED,              |
|               NO_SHOW, CANCELLED               COMPLETED, REJECTED             |
|                     |                             |                            |
|                     v                             v                            |
|             [ FollowUpTask ]               [ RecallTask ]                      |
|             - Missed Care Tracking         - Future screening recall           |
|                                    |                                           |
|                                    v                                           |
|                     [ CareGapClosureEvidence ]                                 |
|                     - Verified Diagnostic / Clinical Resolution                |
|                     - Marks Recommendation as COMPLETED                        |
+--------------------------------------------------------------------------------+
```

---

## 3. Data Models
1. **`Appointment` (`appointments` table):**
   - Linked to `person_id`, `recommendation_id`, `care_plan_id`.
   - `scheduled_time`, `service_type`, `location`, `status` (`BOOKED`, `ATTENDED`, `NO_SHOW`, `CANCELLED`).
2. **`Referral` (`referrals` table):**
   - Specialist clinical referral (`specialty`, `reason`, `referring_provider_id`, `receiving_facility`, `status`).
3. **`FollowUpTask` (`follow_up_tasks` table):**
   - Active outreach / follow-up item generated automatically upon missed care.
4. **`RecallTask` (`recall_tasks` table):**
   - Longitudinal recall engine tracking future screening dates.
5. **`CareGapClosureEvidence` (`care_gap_closure_evidences` table):**
   - Persists evidence type (`LAB_RESULT`, `CLINICAL_NOTE`, `IMAGING_REPORT`, `VACCINATION_RECORD`), evidence metadata, closing provider ID, and clinical notes.
