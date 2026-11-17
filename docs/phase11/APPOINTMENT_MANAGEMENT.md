# WELLWATCH PREVENT — Appointment Management Specification

## 1. Relational Scheduling Mechanics
Appointments in WELLWATCH PREVENT are not isolated calendar events; they are linked to preventive recommendations and patient care plans.

### Fields:
- `id`: UUID.
- `organization_id`: Multi-tenant boundary.
- `person_id`: Patient link.
- `recommendation_id`: Linked preventive recommendation.
- `care_plan_id`: Linked care plan.
- `service_type`: Standard clinical service (e.g. `WELL_CHILD_VISIT`, `IFHAS_COMPREHENSIVE_SCREENING`, `CARDIOMETABOLIC_INTERVENTION`, `DIABETES_GLUCOSE_WORKUP`).
- `scheduled_time`: Timestamp of appointment.
- `location`: Facility or clinic name.
- `provider_id`: Assigned healthcare provider.
- `status`: `BOOKED`, `ATTENDED`, `NO_SHOW`, `CANCELLED`.
- `attendance_notes`: Provider / coordinator notes.

---

## 2. Missed Appointment Trigger Mechanics
When an appointment status transitions to `NO_SHOW` (via `PATCH /api/v1/navigation/appointments/{appointment_id}/status`):
- `NavigationService` automatically detects the missed care event.
- It instantiates a new `FollowUpTask` with:
  - `priority = URGENT`.
  - `task_type = "MISSED_CARE_RE_ENGAGEMENT"`.
  - `notes = "Patient missed appointment on {scheduled_time}. Follow up required to reschedule."`.
- If a linked `recommendation_id` exists, its status remains `SCHEDULED` or returns to `ACTIVE` so the care gap is not lost.
