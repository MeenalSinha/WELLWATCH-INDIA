# WELLWATCH PREVENT — Preventive Care Plan Specification

## 1. Concept & Scope
A `PreventiveCarePlan` represents a structured, patient-centered clinical care program spanning multiple interconnected interventions over a designated period (e.g., 6–12 months).

While individual `PreventiveRecommendation` objects address discrete clinical gaps (e.g., specific vaccines, blood pressure check), a Care Plan aggregates them into a coherent management protocol.

---

## 2. Relational Hierarchy
```
[ Patient / Person ]
         |
         v
[ PreventiveCarePlan ]
   - plan_name: "Annual Preventive Care Plan - 2026"
   - overall_goals: ["Blood pressure control < 130/80", "Completion of IFHAS screening"]
   - status: "DRAFT" | "ACTIVE" | "COMPLETED" | "SUPERSEDED"
   - target_completion_date: 2026-12-31
         |
         +---> [ CarePlanItem ]
         |        - item_title: "Attend IFHAS Comprehensive Screening"
         |        - recommendation_id: <UUID>
         |        - status: "PENDING" | "IN_PROGRESS" | "COMPLETED"
         |
         +---> [ CarePlanItem ]
                  - item_title: "Primary Care Blood Pressure Re-check"
                  - recommendation_id: <UUID>
                  - status: "PENDING"
```

---

## 3. API Endpoints
- `POST /api/v1/care-plans`: Create a care plan for a patient.
- `GET /api/v1/care-plans/patient/{person_id}`: Retrieve all care plans for a patient.
- `POST /api/v1/care-plans/{care_plan_id}/items`: Attach a recommendation or clinical item to a care plan.
- `PATCH /api/v1/care-plans/{care_plan_id}/status`: Update overall care plan lifecycle status.
