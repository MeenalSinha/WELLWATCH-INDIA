# WELLWATCH PREVENT
## Pediatric Profile Specification (Phase 5)

**Document ID:** SPEC-PH5-01  
**Target Market:** Abu Dhabi, United Arab Emirates  
**Status:** IMPLEMENTED & VERIFIED  

---

## 1. Domain Overview
The pediatric profile provides specialized preventive intelligence for children from birth to 17 years, with particular focus on the first 6 years of life:
- **Child Identity Reference:** Age calculated in both full years and elapsed months.
- **Guardianship Linkage:** Wadeema-compliant legal parent/guardian proxy with verification status.
- **Developmental Observations:** Motor, cognitive, speech, and anthropometric percentiles (height, weight, head circumference).
- **Well-Child Visit Milestone Tracker:** Complete history of visits and preventive immunizations.
- **Family Context:** Documented hereditary risks (e.g. familial hypercholesterolemia, early CVD, Type 1/2 diabetes).

## 2. API Schema
Endpoint: `GET /api/v1/patients/{id}/pediatric`
Response structure:
```json
{
  "person": {
    "id": "person-uuid",
    "full_name": "Zayed Al Nahyan",
    "date_of_birth": "2023-02-10",
    "age_years": 3,
    "age_months": 36,
    "gender": "MALE",
    "population_classification": "PEDIATRIC"
  },
  "guardians": [
    {
      "id": "guardian-id",
      "relationship": "MOTHER",
      "scope": "FULL_PREVENTIVE_HEALTHCARE_PROXY",
      "verified": true
    }
  ],
  "well_child_visits": [
    {
      "screening_type": "WELL_CHILD_18M",
      "date": "2024-08-10",
      "status": "COMPLETED",
      "findings": "Normal developmental milestones observed"
    }
  ],
  "developmental_milestones": [],
  "family_history": [],
  "completeness": {
    "score": 60.0,
    "status": "PARTIALLY_COMPLETE"
  }
}
```
