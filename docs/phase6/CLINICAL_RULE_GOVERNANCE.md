# WELLWATCH PREVENT
## Clinical Rule Governance & Lifecycle (Phase 6)

**Document ID:** GOV-PH6-01  
**Target Market:** Abu Dhabi, United Arab Emirates  
**Status:** IMPLEMENTED & VERIFIED  

---

## 1. Rule Governance Lifecycle

To guarantee clinical safety and prevent untested rules from reaching active assessments, all pathway definitions follow a strict 6-stage lifecycle:

```text
  [1. DRAFT]
      |
      | (Authored by Clinical Working Group with verified source citation)
      v
  [2. UNDER_REVIEW]
      |
      | (Reviewed by Healthcare Governance Board & Chief Medical Officer)
      v
  [3. APPROVED]
      |
      | (Scheduled activation matching DoH policy effective date)
      v
  [4. ACTIVE] ---> [Engine evaluates live patient cohorts]
      |
      +---> [5. SUSPENDED] (Temporarily halted during clinical re-evaluation)
      |
      v
  [6. RETIRED] (Superseded by newer version; retained for historical audit)
```

## 2. Immutability & Audit Tracking
- Historical assessments permanently retain the exact `pathway_id` and `pathway_version` under which they were generated.
- A rule definition cannot be edited in place once marked `ACTIVE`. Any modification requires generating a new semantic version (e.g. `1.2.0` -> `1.3.0`).
