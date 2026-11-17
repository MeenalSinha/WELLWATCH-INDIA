# DATABASE SCHEMA, MIGRATION & TRANSACTION INTEGRITY AUDIT

**Project:** WELLWATCH PREVENT — Life-Course Preventive Care Intelligence  
**Component:** PostgreSQL / SQLite Engine, SQLAlchemy 2.0, Alembic Migrations  
**Migration Head:** `006_in_app_notifications_and_audit`  
**Status:** FULLY COMPLIANT

---

## 1. Migration History & Schema Evolution

The database schema has evolved through 6 managed Alembic migrations without data loss or schema drift:
1. `001_initial_schema`: Base enterprise entities (`organizations`, `users`, `audit_logs`, `screening_encounters`, `guardianships`, `consents`).
2. `002_longitudinal_clinical_data`: Persons, patient profiles, clinical encounters, observations, documents, and data provenance.
3. `003_pathways_and_milestones`: Screening records, care gap tracking, and pathway configurations.
4. `004_assessments_and_genomics`: Preventive assessments, genomic findings, variant tiering, and counselor reviews.
5. `005_recommendations_outreach_navigation`: Preventive recommendations, care plans, communication preferences, outreach tasks, appointments, referrals, follow-up tasks, recalls, and care-gap closure evidence.
6. `006_in_app_notifications_and_audit`: Dedicated `in_app_notifications` table and `authorization_status` column on `communication_preferences`.

---

## 2. Table Integrity & Foreign Key Safeguards

### 2.1 In-App Notifications (`in_app_notifications`)
- `id`: Primary Key (UUIDv4 string)
- `organization_id`: Foreign Key to `organizations.id` (RESTRICT)
- `person_id`: Foreign Key to `persons.id` (CASCADE)
- `recipient_user_id`: Foreign Key to `users.id` (CASCADE)
- `title`, `body`, `direction`, `is_read`, `read_at`, `created_at`: Indexed and non-nullable where appropriate.

### 2.2 Communication Preferences (`communication_preferences`)
- Added column: `authorization_status` (Enum: `NOT_PROVIDED`, `PENDING`, `AUTHORIZED`, `OPTED_OUT`, `REVOKED`, `EXPIRED`, `NOT_APPLICABLE`).
- Default: `consent_opt_in = False`, `authorization_status = NOT_PROVIDED`.

---

## 3. Transaction Boundaries & Idempotency
- All service methods operate within explicit SQLAlchemy sessions.
- Multi-step operations (e.g., care-gap closure + recommendation completion + audit log generation) are encapsulated in single commit boundaries. If any validation fails, the entire transaction rolls back cleanly.
- Alembic upgrade executed cleanly to head without warnings or schema errors.
