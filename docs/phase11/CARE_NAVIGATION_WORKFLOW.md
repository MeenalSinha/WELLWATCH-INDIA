# WELLWATCH PREVENT — Care Navigation Workflow Specification

## 1. End-to-End Navigation Journey
The Care Navigation engine bridges clinical intelligence and actual community healthcare delivery:

1. **Recommendation Accepted:** When a patient or clinician accepts a preventive recommendation, it enters `ACCEPTED` status.
2. **Scheduling:** The patient/coordinator books an appointment (`POST /api/v1/navigation/appointments`), transitioning recommendation to `SCHEDULED`.
3. **Attendance Verification:**
   - If the patient attends (`ATTENDED`), the recommendation enters `IN_PROGRESS`.
   - If the patient fails to attend (`NO_SHOW`), the system triggers a missed-care protocol.
4. **Diagnostic / Clinical Follow-Up:** If specialized evaluation is required, the clinician issues a governed `Referral`.
5. **Care Gap Verification & Closure:** Upon receiving verified lab/imaging/clinical encounter results, a clinician records `CareGapClosureEvidence`, successfully transitioning the recommendation to `COMPLETED`.

---

## 2. Multi-Role Navigation Coordination
- **Care Coordinators & Navigators:** Manage appointment queues, coordinate transport/accessibility, and address follow-up tasks for missed care.
- **Clinicians & Physicians:** Authorized to issue specialist referrals and execute verified care-gap closures.
- **Patients & Guardians:** View scheduled appointments, review care plan goals, and request rescheduling.
