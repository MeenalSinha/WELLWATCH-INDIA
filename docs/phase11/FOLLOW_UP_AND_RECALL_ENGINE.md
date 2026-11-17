# WELLWATCH PREVENT — Follow-Up & Longitudinal Recall Engine

## 1. Follow-Up Task Management
The `follow_up_tasks` table provides operational tracking for care coordinators to re-engage patients:
- **Triggers:**
  - Missed appointment (`NO_SHOW`).
  - Provider requested additional diagnostic information.
  - Outreach failure after maximum retries.
- **Workflow:**
  - Coordinators filter active tasks by priority (`EMERGENT`, `URGENT`, `ROUTINE`).
  - Upon contacting the patient or guardian, notes are updated and the task is marked `RESOLVED`.

---

## 2. Longitudinal Recall Engine
Preventive medicine requires recurring health maintenance at defined intervals:
- **ADPHC IFHAS Screening:** 12-month recall for all eligible adults.
- **Pediatric Well-Child Schedule:** Age-based recall (2 months, 4 months, 6 months, 9 months, 12 months, 18 months, 24 months, annual).
- **Hypertension Re-measurement:** 30-day recall for borderline/elevated readings.
- **Colorectal Cancer Screening:** 24-month FIT recall or 10-year colonoscopy recall.

The `recall_tasks` table tracks `due_date`, `recall_type`, and `status` (`SCHEDULED`, `NOTIFIED`, `OVERDUE`, `COMPLETED`), automatically initiating Phase 10 outreach when `due_date` is reached.
