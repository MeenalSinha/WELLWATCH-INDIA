# WELLWATCH PREVENT — Outreach Delivery Lifecycle & Suppression Rules

## 1. Delivery Lifecycle States
An `OutreachTask` progresses through the following states:

- `PENDING`: Task created, queued for dispatch.
- `DISPATCHED`: Forwarded to delivery provider.
- `DELIVERED`: Confirmed delivered by provider (e.g. In-App).
- `READ`: Read by recipient in portal.
- `FAILED`: Delivery attempt rejected by provider.
- `SUPPRESSED_OPT_OUT`: Suppressed due to patient consent withdrawal.
- `SUPPRESSED_QUIET_HOURS`: Held or suppressed because current local time is within quiet hours (e.g. 21:00 to 08:00).
- `NOT_CONFIGURED`: External carrier credentials missing (Honest boundary).

---

## 2. Anti-Fatigue 7-Day Duplicate Suppression
To prevent citizen alert fatigue, `OutreachService` enforces an anti-duplicate rule:
- Before creating a new `OutreachTask`, the service queries existing tasks for the same `person_id` and `template_id`.
- If an outreach task was already created within the last 7 calendar days, the redundant outreach is suppressed, returning the existing task record without re-messaging the patient.

---

## 3. Wadeema's Law Minor Routing Logic
When the target patient is under 18:
1. `OutreachService` inspects `guardianships` table for a record linking the child to an adult guardian where `verification_status == VERIFIED`.
2. If verified guardian exists:
   - Sets `recipient_role = "GUARDIAN"`.
   - Routes message to guardian contact/portal.
3. If no verified guardian is recorded:
   - Suppresses direct minor messaging.
   - Raises clinical escalation task for case coordinator.
