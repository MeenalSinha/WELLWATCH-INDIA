# WELLWATCH PREVENT — UAE Data Residency & Regulatory Assessment
**Applicable Legal Instruments:**
1. UAE Federal Law No. 2 of 2019 on the Use of Information and Communication Technology in Healthcare
2. Abu Dhabi Department of Health (DoH) ADHICS Standard (Version 2.0)
3. UAE Federal Law No. 3 of 2016 on Child Rights ("Wadeema's Law")
4. UAE Federal Decree-Law No. 45 of 2021 on Personal Data Protection (PDPL)

---

## 1. Executive Summary & Legal Mandate
In the United Arab Emirates, health data is classified as highly sensitive sovereign personal information. Under Article 13 of Federal Law No. 2 of 2019, health data generated, processed, or stored within the UAE must **not be transferred, stored, or processed outside the State** except in cases defined by ministerial resolution.

WELLWATCH PREVENT adheres strictly to complete in-country data residency across all application tiers.

---

## 2. Regulatory Compliance Matrix

| Regulatory Requirement | Legal Citation | Technical Implementation in WELLWATCH PREVENT | Compliance Status |
| :--- | :--- | :--- | :--- |
| **In-Country Physical Data Residency** | Federal Law No. 2/2019, Art. 13 | All PostgreSQL instances, backups, and app servers deployed strictly to UAE Sovereign Cloud regions (Abu Dhabi / Dubai). | **FULLY COMPLIANT** |
| **Prohibition of Offshore Cloud Egress** | ADHICS v2, Section 8 (Data Protection) | Outbound network egress blocked at WAF/VNet level; no telemetry, crash logs, or analytics exported to foreign SaaS. | **FULLY COMPLIANT** |
| **Pediatric Health & Minor Rights Protection**| Wadeema's Law (Law No. 3/2016) | Explicit parental/guardian identity linkage for patients < 18 years; consent tracking in `ConsentRecord` table. | **FULLY COMPLIANT** |
| **Immutable Audit Trails** | ADHICS v2, Section 11 (Audit Logging) | Tamper-evident `AuditLog` captures user identity, role, IP address, timestamp, and accessed resource. Retained for 25 years. | **FULLY COMPLIANT** |
| **Role-Based Access Control & Licensure**| DoH Healthcare Provider Regulations | Clinical care-gap closures, referral creation, and diagnostic verifications restricted to licensed providers. | **FULLY COMPLIANT** |
| **Cryptographic Protection** | ADHICS v2, Section 9 (Cryptography) | AES-256 for data at rest; TLS 1.3 for data in transit; SHA-256 for database snapshot integrity verification. | **FULLY COMPLIANT** |

---

## 3. Pediatric Safeguards (Wadeema's Law Compliance)
Pediatric preventive health records (Well-Child visits, developmental milestones, metabolic screenings) are subject to heightened statutory privacy:
1. **Guardian Representation:** A minor's clinical record is bound to a verified adult guardian (`guardian_person_id` in `PersonIdentifier`).
2. **Access Isolation:** Only authorized pediatric clinicians and legally verified guardians can view developmental assessments.
3. **Audit Immutability:** Any administrative or clinical inspection of a minor's record generates a high-priority audit event.
