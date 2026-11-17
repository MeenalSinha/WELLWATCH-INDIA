# WELLWATCH PREVENT
## Longitudinal Healthcare Data Architecture (Phase 4)

**Document ID:** ARCH-PH4-01  
**Target Market:** Abu Dhabi, United Arab Emirates  
**Compliance Standards:** DoH Abu Dhabi, ADPHC, ADHICS v2, UAE Health Data Law No. 2/2019, Wadeema's Law  
**Status:** IMPLEMENTED & VERIFIED  

---

## 1. Architectural Separation: Person vs. User

A critical flaw in early healthcare prototypes is the conflation of application login accounts (`User`) with clinical healthcare subjects (`Person`).

In WELLWATCH PREVENT:
- **`Person` (Healthcare Subject):** Represents the biological human whose longitudinal health observations, family histories, conditions, and screening records are maintained. A newborn child or elder receiving preventive care has a `Person` record but typically does not possess a direct system login account.
- **`User` (Identity Principal):** Represents an authenticated principal (e.g. Healthcare Provider, Community Health Worker, Parent/Guardian, System Administrator) with an email address, password hash, and RBAC permissions.
- **Relational Linkage:** A `Person` optionally links to a `User` via a nullable foreign key (`user_id`). A parent `User` can be linked to a child `Person` through a legally verified `Guardianship` record.

```
       +----------------------------+
       |            User            | (Authentication Principal)
       | - email                    |
       | - role (8 RBAC Roles)      |
       | - password_hash            |
       +--------------+-------------+
                      | 0..1 (Optional link)
                      v
       +----------------------------+
       |           Person           | (Healthcare Subject)
       | - first_name, last_name    |
       | - date_of_birth, gender    |
       | - record_status            |
       +--------------+-------------+
                      |
        +-------------+-------------+-------------+
        |                           |             |
        v                           v             v
+----------------+          +---------------+  +------------------+
| PatientProfile |          | ClinicalObs   |  | ScreeningRecord  |
| - population   |          | - LOINC code  |  | - Well-Child /   |
| - completeness |          | - value, unit |  |   IFHAS packages |
+----------------+          +---------------+  +------------------+
```

---

## 2. Core Entities in Phase 4 Schema

1. **`Person`:** Foundational demographic entity with data minimization.
2. **`PersonIdentifier`:** Namespace-scoped identifiers (`EMIRATES_ID`, `MRN`, `MALAFFI_RECIPROCAL_ID`, `PASSPORT`, `INTERNAL_UUID`) with SHA-256 zero-knowledge query hashing.
3. **`PatientProfile`:** Population classification (`PEDIATRIC`, `ADULT`, `GERIATRIC`), primary healthcare facility, and completeness indicators.
4. **`Household` & `HouseholdMember`:** Legitimate care-coordination groupings for community health workers and family-based preventive follow-up.
5. **`DataSource`:** Registry of authorized clinical data systems (EMR, Malaffi HIE, Lab, Community Screening, Patient Portal) with trust levels and connection states.
6. **`DataProvenance`:** Complete audit traceability on collection method, import timestamp, source external ID, transformation history, and responsible processor.
7. **`Encounter`:** Clinical consultations and preventive screening events.
8. **`ClinicalObservation`:** LOINC/SNOMED CT structured measurements with raw vs. canonical units, collection vs. recording timestamps, and 8-state epistemic classifications.
9. **`ClinicalCondition`:** Documented chronic diseases (ICD-10 / SNOMED CT) with clinical and verification status.
10. **`FamilyHistory`:** Inherited disease history across degrees of relationship.
11. **`ScreeningRecord`:** Structured history of Well-Child visit milestones and IFHAS screening packages.
12. **`ClinicalDocument`:** Secure metadata references and SHA-256 checksums for external clinical PDFs and lab results.

---

## 3. Data Protection and Tenant Isolation

- **Facility Boundaries:** All clinical encounters, patient profiles, and observations are scoped to an authorized `organization_id`. Cross-facility access attempts by clinical personnel are rejected with `PermissionError`.
- **Administrative Separation of Duties:** In compliance with ADHICS v2, `SYSTEM_ADMIN` users possess privileges to configure infrastructure but are strictly barred from querying individual patient clinical records.
- **UAE Wadeema's Law Compliance:** Access to pediatric dependent records requires active, verified `Guardianship` records.
