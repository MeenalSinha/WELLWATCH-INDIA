# WELLWATCH PREVENT — Clinical & Operational User Training Program
**Target Audience:** Physicians, Triage Nurses, Care Coordinators, Community Health Workers  
**Curriculum Format:** Modular Interactive Workshops & Quick-Reference SOPs (Arabic & English)  
**Certification Requirement:** Mandatory Pre-Pilot Assessment ($\ge 85\%$ Pass Mark)

---

## 1. Modular Curriculum Overview

```
┌────────────────────────────────────────────────────────────────────────┐
│                        CORE TRAINING MODULES                           │
├────────────────────────────────────────────────────────────────────────┤
│ Module 1: System Foundations & ADHICS Data Protection (All Staff)      │
│ Module 2: Pediatric Well-Child Life-Course Workflows (Pediatrics/Nurses│
│ Module 3: Adult IFHAS Cardiometabolic Decision Support (Physicians)    │
│ Module 4: High-Integrity Care Gap Closure & Evidence (Clinicians)      │
│ Module 5: Field Operations & Offline Tablet Workspace (CHWs)           │
│ Module 6: Care Navigation, Recall Queues & Cultural Outreach (Coord.) │
└────────────────────────────────────────────────────────────────────────┘
```

---

## 2. Role-Specific Training Pathways

### Pathway A: Primary Care Physicians (Modules 1, 3, 4)
- Understanding CDS risk scoring algorithms (MOD-CVD-01, MOD-DIAB-02) and clinical explainability drivers.
- Executing legal attestation for care-gap closure via `verify_and_close_care_gap`.
- Overriding recommendations and documenting clinical rationale.

### Pathway B: Community Health Workers (Modules 1, 5)
- Device registration and offline lease authentication.
- Offline screening intake in `/chw` without cellular network coverage.
- Handling sync conflicts and verifying successful upload.

### Pathway C: Care Coordinators (Modules 1, 6)
- Managing missed-appointment follow-up tasks (`FollowUpTask`).
- Dispatching culturally validated Arabic/English recall messages.
- Closing follow-up tasks with documented resolution notes ($\ge 5$ characters).

---

## 3. Training Collateral & Verification
- **Quick-Reference Pocket Cards:** Laminated dual-language pocket cards detailing key error codes, sync steps, and emergency helpdesk contacts.
- **Competency Verification:** 10 simulated clinical encounters in the Staging sandbox; participants must achieve 100% correct workflow completion.
