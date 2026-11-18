# IDENTITY INTEGRATION & DEMOGRAPHIC MATCHING
## Phase 14 Emirates ID & UAE Pass Identity Protocol

**Project:** WELLWATCH PREVENT — Life-Course Preventive Care Intelligence  
**Target Market:** Abu Dhabi, United Arab Emirates  

---

## 1. Identity Hierarchy & Namespaces

WELLWATCH PREVENT implements a multi-tier identity resolution pipeline:

1. **Tier 1 (Authoritative National ID):** Emirates ID (`EMIRATES_ID`)
   - Regex validation: `^784-[0-9]{4}-[0-9]{7}-[0-9]{1}$`.
   - Guaranteed unique national identifier across the UAE.
   - Zero-knowledge SHA-256 identifier hashing (`identifier_hash`) for privacy-preserving lookup.
2. **Tier 2 (Health System Identifier):** Medical Record Number (`MRN`)
   - Scoped to issuing organization (`Organization.id` or `issuing_authority`).
3. **Tier 3 (HIE Identifier):** Malaffi Reciprocal ID (`MALAFFI_RECIPROCAL_ID`).
4. **Tier 4 (Internal UUID):** Unique surrogate primary key (`Person.id`).

---

## 2. Demographic Matching Algorithm

When an external FHIR Patient resource arrives without an Emirates ID:
1. Exact match on normalized Last Name + Date of Birth + Gender.
2. If single candidate matches with 100% confidence, links incoming data to existing `Person`.
3. If multiple candidates or zero candidates match, quarantines record for manual identity reconciliation by health informatics staff, preventing wrongful chart merge.
