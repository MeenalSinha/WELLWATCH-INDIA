# WELLWATCH PREVENT — Outreach Template Specification

## 1. Governed Template Definitions
All outreach messages are governed under `configs/templates/outreach_templates.json`.
Each template is pre-approved for specific clinical pathways, target populations, and communication channels.

### Available Templates:
1. **`TEMPLATE_PED_WELL_CHILD_AR_01` & `TEMPLATE_PED_WELL_CHILD_EN_01`:**
   - **Target:** Pediatric patients (dispatched to legal guardians).
   - **Context:** Wadeema's Law child developmental checkup.
   - **Placeholders:** `{child_name}`, `{guardian_name}`, `{booking_link}`.
2. **`TEMPLATE_ADULT_IFHAS_AR_01` & `TEMPLATE_ADULT_IFHAS_EN_01`:**
   - **Target:** Adult citizens & residents.
   - **Context:** Annual DoH IFHAS Comprehensive Screening.
   - **Placeholders:** `{patient_name}`, `{booking_link}`.
3. **`TEMPLATE_HTN_RECALL_AR_01` & `TEMPLATE_HTN_RECALL_EN_01`:**
   - **Target:** Adults with elevated blood pressure.
   - **Context:** Recall for clinical BP re-measurement.
   - **Placeholders:** `{patient_name}`, `{latest_reading}`, `{booking_link}`.
4. **`TEMPLATE_APPOINTMENT_REMINDER_AR_01` & `TEMPLATE_APPOINTMENT_REMINDER_EN_01`:**
   - **Target:** Scheduled patients / guardians.
   - **Context:** 24-hour appointment reminder to minimize no-shows.
   - **Placeholders:** `{recipient_name}`, `{appointment_time}`, `{location}`.

---

## 2. Dynamic Variable Interpolation
Templates use standard Python `{key}` interpolation. The `OutreachService` strictly checks required parameters before rendering. If missing placeholders occur, the service safely preserves token placeholders or raises validation errors rather than generating truncated messages.
