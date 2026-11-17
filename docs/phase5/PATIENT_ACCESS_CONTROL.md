# WELLWATCH PREVENT
## Patient-Level Access Control Specification (Phase 5)

**Document ID:** SEC-PH5-01  
**Target Market:** Abu Dhabi, United Arab Emirates  
**Status:** IMPLEMENTED & VERIFIED  

---

## 1. Access Control Matrix

| Persona / Role | Own Profile | Dependent Minor | Facility Patient | Cross-Facility Patient | System Administration |
|:---|:---:|:---:|:---:|:---:|:---:|
| **Individual** | ALLOWED | DENIED | DENIED | DENIED | DENIED |
| **Parent/Guardian** | ALLOWED | ALLOWED (Verified only) | DENIED | DENIED | DENIED |
| **Healthcare Provider** | N/A | N/A | ALLOWED | DENIED | DENIED |
| **Community Health Worker**| N/A | N/A | ALLOWED | DENIED | DENIED |
| **Care Coordinator** | N/A | N/A | ALLOWED | DENIED | DENIED |
| **Population Health Admin**| N/A | N/A | ALLOWED (Aggregated/Org) | DENIED | DENIED |
| **System Administrator** | DENIED | DENIED | DENIED (Zero PHI) | DENIED (Zero PHI) | ALLOWED (Infra only) |
| **Data Governance Officer**| DENIED | DENIED | DENIED (Zero PHI) | DENIED (Zero PHI) | ALLOWED (Audit logs) |

## 2. Enforcement Mechanisms
- Implemented in `PersonService.get_person_by_id()` and route handlers in `backend/app/api/v1/endpoints/patients.py`.
- Verified empirically through automated tests in `test_longitudinal_data.py` and `test_patient_intelligence.py`.
