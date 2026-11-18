# POPULATION HEALTH INTELLIGENCE ARCHITECTURE
## Phase 12 Architecture Specification

**Project:** WELLWATCH PREVENT — Life-Course Preventive Care Intelligence  
**Target Market:** Abu Dhabi, United Arab Emirates  
**Governing Standard:** Abu Dhabi Public Health Center (ADPHC) Standard 2026, UAE Health Data Law  
**Status:** Production-Ready  

---

## 1. Executive Summary

Phase 12 implements an epidemiological and population-scale intelligence layer on top of WELLWATCH PREVENT's longitudinal clinical data core. It transforms individual clinical encounters, screening observations, risk assessments, and care navigation events into governed, cohort-level epidemiological surveillance metrics.

The architecture is built upon three foundational tenets:
1. **Mathematical Denominator Integrity:** Distinct, governed denominators that prevent epidemiological misinterpretation.
2. **Small-Cell Privacy Suppression:** Automatic suppression of geographic cells with fewer than 5 individuals (`min_cell_size: 5`) to prevent re-identification.
3. **Clinical Ground-Truth Requirement:** Machine learning and algorithmic equity audits require verified clinical labels, strictly barring synthetic claims.

---

## 2. Architectural Diagram

```
+-----------------------------------------------------------------------------------+
|                           LONGITUDINAL CLINICAL CORE                              |
|   [Persons]      [ClinicalObservation]    [ScreeningRecord]    [PreventiveAssessment]  |
|   [CarePlan]     [Appointment]            [Referral]           [CareGapClosureEvidence] |
+-----------------------------------------------------------------------------------+
                                         |
                                         v
+-----------------------------------------------------------------------------------+
|                     POPULATION ANALYTICS ENGINE (Service Layer)                   |
|  - Denominator Engine (Registered vs Eligible vs Screened vs Closed)               |
|  - Life-Course Cohort Segmenter (Pediatric / Adult / Geriatric)                    |
|  - Small-Cell Privacy Filter (< 5 records suppressed)                             |
|  - Disparity Ratio & ML Fairness Auditor (AD_CENTRAL Benchmark)                   |
+-----------------------------------------------------------------------------------+
                                         |
                                         v
+-----------------------------------------------------------------------------------+
|                        GOVERNED REST API & ACCESS LOGGING                         |
|  GET /api/v1/population/overview        -> Comprehensive Population Overview       |
|  GET /api/v1/population/pediatric       -> Pediatric Surveillance & Wadeema's Law  |
|  GET /api/v1/population/adult           -> Adult IFHAS Cardiometabolic Intelligence|
|  GET /api/v1/population/geographic      -> Small-Cell Protected Regional Metrics   |
|  GET /api/v1/population/equity          -> Disparity Ratios & ML Fairness Audit    |
|  GET /api/v1/population/trends          -> Longitudinal Temporal Trends            |
|  POST /api/v1/population/reports/generate -> Governed Audit-Logged Report Exports  |
+-----------------------------------------------------------------------------------+
                                         |
                                         v
+-----------------------------------------------------------------------------------+
|                          AUDIT TRAIL & METRIC SNAPSHOTS                           |
|  - PopulationMetricSnapshot (Category, Value, Denominator, Snapshot Time)          |
|  - AnalyticsAccessLog (User ID, Endpoint, Parameters, Access Time)                |
+-----------------------------------------------------------------------------------+
```

---

## 3. Core Database Models

### 3.1 `PopulationMetricSnapshot` (`population_metric_snapshots`)
Stores periodic and on-demand calculated aggregate metrics for historical trend analysis and regulatory submissions:
- `organization_id`: Organization scope or `None` for Emirate-wide surveillance.
- `metric_category`: Category (`OVERVIEW`, `PEDIATRIC`, `ADULT_IFHAS`, `GEOGRAPHIC`, `EQUITY`).
- `metric_name`: Metric identifier (e.g. `screening_coverage_percentage`, `confirmed_overdue`).
- `metric_value`: Numerator value.
- `denominator_value`: Governed denominator.
- `rate_percentage`: Calculated rate percentage (`metric_value / denominator_value * 100`).
- `dimension_name` & `dimension_value`: Grouping dimensions (e.g. `region: AD_CENTRAL`, `age_stage: TODDLER`).
- `is_suppressed`: Boolean flag indicating whether small-cell suppression was enforced.
- `snapshot_timestamp`: UTC timestamp of metric calculation.

### 3.2 `AnalyticsAccessLog` (`analytics_access_logs`)
Records every population analytics access for information governance compliance:
- `user_id`: Accessing user ID.
- `endpoint`: API endpoint accessed.
- `filter_criteria`: Query parameters applied (e.g. `time_window_days`, `organization_id`).
- `accessed_at`: UTC timestamp.

---

## 4. Denominator Engine Specifications

In epidemiological surveillance, confusing registered individuals with eligible populations or completed screenings severely distorts public health insight. The Denominator Engine strictly computes:

$$\text{Coverage Rate} = \frac{\text{Completed Screenings}}{\text{Eligible Population}} \times 100$$

$$\text{No-Show Rate} = \frac{\text{No-Show Appointments}}{\text{Total Booked Appointments}} \times 100$$

$$\text{Referral Completion Rate} = \frac{\text{Completed Referrals}}{\text{Total Created Referrals}} \times 100$$

$$\text{Care Gap Closure Rate} = \frac{\text{Closed Care Gaps (Evidence-Backed)}}{\text{Identified Care Gaps}} \times 100$$

---

## 5. Security and Role-Based Access Control (RBAC)

Population intelligence endpoints are restricted by role:
- `POPULATION_HEALTH_ADMIN`: Full access to overview, pediatric, adult, geographic, equity, and report generation across organizations.
- `SYSTEM_ADMIN`: Technical monitoring and system audit access.
- `HEALTHCARE_PROVIDER` / `CARE_COORDINATOR`: Filtered access restricted to their licensed organization (`primary_organization_id`).
- `INDIVIDUAL` / `PARENT_GUARDIAN` / `COMMUNITY_HEALTH_WORKER`: Denied access with HTTP 403 Forbidden.
