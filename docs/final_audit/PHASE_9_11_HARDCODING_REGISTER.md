# HARDCODING ELIMINATION REGISTER: PHASES 9–11

**Project:** WELLWATCH PREVENT — Life-Course Preventive Care Intelligence  
**Scope:** Complete Inventory of Hardcoding Detected, Repaired and Eliminated across Phases 9, 10, and 11  
**Audit Date:** 2026-11-18  
**Audit Status:** 100% ELIMINATED & CONFIGURED

---

## 1. Inventory of Eliminated Hardcoding

| ID | Module / Component | Previous Hardcoded Implementation | Production Solution & Config Source | Verification Method |
| :--- | :--- | :--- | :--- | :--- |
| **HC-901** | `RecommendationService` | Recommendations inferred overdue care simply from pediatric/adult age check (`age <= 18`) | Consumes empirical care gap evaluations from `PathwayEngine.evaluate_care_gaps` and `PreventiveAssessment` | `test_generate_pediatric_recommendations_well_child` |
| **HC-902** | `RecommendationService` | Hardcoded rule definitions and inline category strings | Versioned JSON registries: `configs/recommendation_rules/*.json` with schema validation | `test_recommendation_rule_validation_rejects_duplicates_and_unapproved` |
| **HC-903** | `RecommendationService` | Hardcoded priority assignments and clinical review flags | Rule-level configuration: `default_priority`, `recommended_window_days`, and `requires_human_review` | Unit & integration tests |
| **HC-1001** | `OutreachService` | Hardcoded quiet hours (`21:00` to `08:00`) directly inside Python service code | Centralized operational policy: `configs/outreach_policy.json` (`quiet_hours.start_time`, `quiet_hours.end_time`) | `OutreachService.load_policy()` |
| **HC-1002** | `OutreachService` | Hardcoded 7-day recall suppression window (`timedelta(days=7)`) in service logic | Governed operational policy: `configs/outreach_policy.json` (`duplicate_suppression.suppression_window_days`) | `test_duplicate_outreach_suppressed_within_7_days` |
| **HC-1003** | `OutreachService` | Hardcoded maximum retry limit of 3 attempts | Centralized operational policy: `configs/outreach_policy.json` (`retry_policy.max_retries`) | `test_integrated_workflow_4_failed_outreach_and_retry` |
| **HC-1004** | `OutreachService` | Default affirmative consent (`consent_opt_in=True`) hardcoded upon preference creation | Explicit opt-in mandate (`consent_opt_in=False`, `OutreachAuthorizationStatus.NOT_PROVIDED`) | `test_default_preference_is_arabic_and_in_app` |
| **HC-1101** | `NavigationService` | Hardcoded 3-day deadline for missed appointment follow-up task creation | Governed operational policy: `configs/navigation_policy.json` (`deadlines_days.missed_appointment`) | `NavigationService.load_policy()` |
| **HC-1102** | `NavigationService` | Hardcoded 2-day deadline for rejected referral follow-up task creation | Governed operational policy: `configs/navigation_policy.json` (`deadlines_days.rejected_referral`) | `test_missed_appointment_no_show_triggers_follow_up` |
| **HC-1103** | `NavigationService` | Hardcoded default clinic locations ("Primary Care Clinic - Abu Dhabi") | Facility lookup from patient's `primary_organization` or authenticated provider's organization | End-to-end workflows |
| **HC-1104** | `NavigationService` | Hardcoded clinical closure acceptance without database record query | Dynamic lookup of `ScreeningEncounter`, `Encounter`, `ClinicalObservation`, and `Referral` tables | `test_verified_care_gap_closure_completes_recommendation` |

---

## 2. Governed Configuration Files Added

1. **`configs/outreach_policy.json`**:
   - Quiet hours boundaries (`21:00` to `08:00`)
   - Channel prioritization order (`IN_APP`, `SMS`, `WHATSAPP`, `EMAIL`)
   - Duplicate suppression windows (7 days)
   - Retry policies (max retries: 3, backoff: 60 minutes)
2. **`configs/navigation_policy.json`**:
   - Missed appointment follow-up deadlines (3 days)
   - Rejected referral follow-up deadlines (2 days)
   - Recall scheduling defaults (14 days advance notice)
   - Default task priority mappings
3. **`configs/schemas/recommendation_rules.schema.json`**:
   - Strict JSON schema for recommendation rules enforcing `approval_status`, `effective_date`, and trigger parameters.

---

## 3. Audit Certification
All identified hardcoding across Phases 9 through 11 has been eradicated. All clinical logic, operational rules, and scheduling thresholds now originate from validated, versioned configuration files.
