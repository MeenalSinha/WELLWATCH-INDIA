# WELLWATCH PREVENT — Module G: Care-Gap Closure Verification Engine

## 1. Zero-Mock Evidence Mandate
A core architectural requirement of WELLWATCH PREVENT is the strict separation between:
1. Recommending a care intervention.
2. Delivering an outreach message.
3. Scheduling an appointment.
4. Actually closing the care gap with clinical proof.

**Under no circumstances does outreach delivery or appointment booking mark a care gap as closed.**

---

## 2. Evidence Verification Schema
Care gap closure requires calling `POST /api/v1/navigation/care-gaps/verify-closure` with a structured payload:
```json
{
  "person_id": "c138b584-...",
  "recommendation_id": "7fa49051-...",
  "care_gap_id": "ADULT_HTN_GAP",
  "evidence_type": "CLINICAL_NOTE",
  "evidence_data": {
    "loinc": "85354-9",
    "value": "122/78 mmHg",
    "verified_encounter_id": "enc-99213"
  },
  "clinical_notes": "Patient attended clinic; repeated blood pressure normal. Care gap closed."
}
```

### Verification Rules:
- If `evidence_type` or `evidence_data` is missing or empty, the API strictly rejects the request with HTTP 400.
- Only upon valid persistence of `CareGapClosureEvidence` does the linked `PreventiveRecommendation` transition to `COMPLETED`.
- A tamper-evident audit log event is recorded documenting the provider ID, timestamp, and clinical evidence summary.
