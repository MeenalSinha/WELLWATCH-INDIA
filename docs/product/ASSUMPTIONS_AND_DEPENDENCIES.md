# WELLWATCH PREVENT — Assumptions & Technical Dependencies
## Architectural Assumptions, External Dependencies, and Risk Register

**Target Jurisdiction:** Abu Dhabi, United Arab Emirates  
**Status:** SPECIFICATION BASELINE  

---

## 1. Product Assumptions

1. **Healthcare Context Assumption:** The primary clinical user base operates within the Abu Dhabi healthcare network (e.g. SEHA, Mubadala Health, private medical groups licensed by DoH).
2. **Connectivity Assumption:** While frontline CHW mobile devices may experience intermittent connectivity in remote Al Dhafra regions, primary healthcare centers and hospitals maintain high-speed, stable internet access.
3. **Data Availability Assumption:** Patients or clinicians possess reliable baseline vitals (BP, glucose, BMI) during screening encounters. Where data is absent, the platform operates under explicit epistemic uncertainty rather than heuristic fabrication.
4. **Legal Basis Assumption:** Processing of adult health screening data is conducted under legitimate preventive healthcare mandates (DoH public health authority) supplemented by explicit patient consent for navigation outreach.
5. **Guardian Legal Competence:** Guardians accessing child records possess legitimate legal custody recognized under UAE law.

---

## 2. External Technical & Integration Dependencies

```mermaid
graph LR
    System["WellWatch Prevent Core"]
    System -.->|Phase 3 (Auth)| IdP["UAE Pass / OIDC Provider"]
    System -.->|Phase 4 (EHR)| Malaffi["Malaffi HIE (FHIR R4)"]
    System -.->|Phase 8 (Genomics)| EGP["Emirati Genome Program"]
    System -.->|Phase 9 (Outreach)| Telco["UAE Telecom SMS / WhatsApp"]
```

| Dependency | Provider / Integration Type | Role in Platform | Current Status (Phases 0–3) | Failure Mode & Fallback |
|---|---|---|---|---|
| **Identity Provider** | UAE Pass / OpenID Connect | User authentication and national identity verification | Standard OIDC Mock / Local JWT provider | Local credential authentication with bcrypt hashing fallback. |
| **Malaffi HIE** | DoH / InHealth (FHIR R4 API) | Longitudinal clinical record ingestion & encounter verification | Contract fixture / Mock schema | Direct entry of screening encounters via web UI / CHW interface. |
| **UAE National Genome** | Emirati Genome Program | Secondary variant clinical reports | Specification stub | Engine operates with genomic dimension marked `NOT_APPLICABLE`. |
| **Telecom Gateway** | Etisalat (e&) / du SMS API | Appointment reminders & outreach delivery | Deferred (Phase 9) | In-app notification center fallback. |
| **Relational Database** | PostgreSQL 16+ | Primary transactional and audit store | Active in Phase 2 | Local SQLite dual-support for local tests / development. |
| **Frontend Framework** | Next.js 14 / React / Tailwind | User interface & portal | Active in Phase 2 | Server-side rendering and static export capability. |

---

## 3. Risk Register & Mitigation Strategy

| Risk ID | Risk Description | Severity | Probability | Mitigation Strategy |
|---|---|---|---|---|
| **RSK-01** | Clinician skepticism toward AI-generated risk scores | High | Moderate | Strictly separate deterministic rules from ML scores; provide local SHAP feature attribution and clinical source citations for every recommendation. |
| **RSK-02** | Data sovereignty violation due to cloud misconfiguration | Critical | Low | Hardcode infrastructure deployment configs to UAE cloud regions; integrate automated CSP policy checks in CI pipeline. |
| **RSK-03** | Unauthorized access to pediatric records by non-custodial party | Critical | Low | Enforce multi-factor verification of guardianship documentation; support immediate guardianship dispute revocation flags. |
| **RSK-04** | Model performance drift across diverse expatriate demographics | High | Moderate | Implement continuous Kolmogorov-Smirnov distribution monitoring; conduct quarterly subgroup fairness evaluations. |
| **RSK-05** | CHW data entry errors in field conditions | Moderate | High | Enforce strict Pydantic range validation (e.g. physiological bounds check) on all form submissions. |
