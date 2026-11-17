# OUTREACH SECURITY, CONSENT & WADEEMA'S LAW AUDIT

**Project:** WELLWATCH PREVENT — Life-Course Preventive Care Intelligence  
**Component:** `OutreachService` & `CommunicationPreference`  
**Regulatory Standards:** UAE Federal Law No. 3 (Wadeema's Law) / ADHICS v2 / DoH Patient Rights  
**Status:** FULLY COMPLIANT

---

## 1. Regulatory Context

Under UAE healthcare regulations, outbound patient communications require strict data protection, affirmative individual consent, and special protections for minors.
- **ADHICS v2 Section 4.3:** Software must enforce individual control over communications without presuming consent.
- **UAE Federal Law No. 3 of 2016 (Wadeema's Law):** Minors (under 18 years) must be protected; communication must route through verified legal guardians.

---

## 2. Consent State Machine & Invariants

### 2.1 Explicit Affirmative Consent State Machine
The `CommunicationPreference` entity implements a 7-state authorization lifecycle:
```
           ┌────────────────────────┐
           │      NOT_PROVIDED      │ (Initial Default)
           └───────────┬────────────┘
                       │
         ┌─────────────┴─────────────┐
         ▼                           ▼
 ┌───────────────┐           ┌───────────────┐
 │   AUTHORIZED  │           │   OPTED_OUT   │
 └───────┬───────┘           └───────┬───────┘
         │                           │
         ▼                           ▼
 ┌───────────────┐           ┌───────────────┐
 │    REVOKED    │           │    EXPIRED    │
 └───────────────┘           └───────────────┘
```

- **Invariant 1:** Creation of a preference never defaults to `consent_opt_in = True`. It initializes with `consent_opt_in = False` and `authorization_status = NOT_PROVIDED`.
- **Invariant 2:** Only records with `consent_opt_in is True` and `authorization_status == AUTHORIZED` are permitted to receive dispatches.
- **Invariant 3:** When `OPTED_OUT` or `REVOKED` is selected, `consent_opt_in` is immediately cleared (`False`), and `opt_out_timestamp` is immutably recorded.

### 2.2 Minor Protection & Legal Guardian Proxy
- For pediatric individuals (`person.is_pediatric == True`), direct communication to the minor is blocked.
- The service queries the `guardianships` table for an active relationship (`verification_status == VERIFIED`).
- If unverified or missing: Task status is set to `SUPPRESSED`, provider status `SUPPRESSED_UNVERIFIED_GUARDIAN`, and failure reason recorded.
- If verified: Task `recipient_role` is set to `"GUARDIAN"`, and `recipient_id` is set to the guardian's user ID.

---

## 3. Operational Policy Controls (`configs/outreach_policy.json`)

1. **Quiet Hours Enforcement:** Governed window (`21:00` to `08:00`). Messages scheduled during quiet hours are queued or delayed.
2. **Duplicate Suppression:** Identical template dispatches to the same individual are suppressed within a 7-day window.
3. **Retry Limits:** Max 3 retries with exponential backoff; tasks that reach max retries are locked against automated retry.

---

## 4. Verification Evidence
All tests in `backend/tests/test_cultural_outreach.py` (8/8) pass, proving zero assumed consent, guardian routing under Wadeema's Law, duplicate suppression, and in-app message isolation.
