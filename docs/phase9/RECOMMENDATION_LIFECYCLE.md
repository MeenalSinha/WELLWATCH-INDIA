# WELLWATCH PREVENT — Recommendation Lifecycle State Machine

## 1. Lifecycle States
A recommendation transitions through strictly defined lifecycle states:

```
  [ GENERATED ]
        |
        v
  [ PENDING_REVIEW ]  ---(Clinician Approves)--->  [ ACTIVE ]
        |                                              |
        +---(Clinician Rejects)---> [ CANCELLED ]      v
                                                  [ OFFERED ]
                                                       |
                             +-------------------------+-------------------------+
                             |                         |                         |
                             v                         v                         v
                       [ ACCEPTED ]              [ DECLINED ]              [ SUPERSEDED ]
                             |                  (Reason logged)            (New rule/age)
                             v
                       [ SCHEDULED ]
                             |
                             v
                      [ IN_PROGRESS ]
                             |
                             v
                       [ COMPLETED ]  <--- (Requires Verified Evidence via Module G)
```

---

## 2. Transition Guardrails
1. **`GENERATED` -> `PENDING_REVIEW`**: Recommendations requiring human sign-off (`requires_human_review = True`) are held in `PENDING_REVIEW`.
2. **`PENDING_REVIEW` -> `ACTIVE`**: Only licensed clinical roles (`PHYSICIAN`, `CLINICIAN`, `NURSE`, `HEALTHCARE_PROVIDER`) can approve. Admin role alone cannot bypass.
3. **`ACTIVE` -> `OFFERED`**: Set when outreach is successfully dispatched to the patient or guardian.
4. **`OFFERED` -> `DECLINED`**: Requires explicit patient refusal reason (`declined_reason` mandatory in payload) to support patient autonomy and avoid harassment.
5. **`SCHEDULED` -> `COMPLETED`**: **CANNOT** be transitioned solely by appointment attendance or message read status. It requires persistent `CareGapClosureEvidence` verifying clinical diagnostic or therapeutic resolution (Module G).
