# COMMUNICATION PROVIDER INTEGRATION & HONEST BOUNDARY AUDIT

**Project:** WELLWATCH PREVENT — Life-Course Preventive Care Intelligence  
**Component:** `CommunicationProviderService` & `OutreachService`  
**Regulatory Standards:** ADHICS v2 / UAE Telecommunications Standards  
**Status:** FULLY COMPLIANT

---

## 1. Scope & Objective

This audit verifies the production integrity of multi-channel patient communication across four supported channels:
1. `IN_APP` (Secure Patient & Guardian Portal)
2. `SMS` (Transactional Preventive Alerts)
3. `WHATSAPP` (Interactive Screening Coordination)
4. `EMAIL` (Periodic Preventive Bulletins & Summaries)

The audit confirms the complete elimination of synthetic delivery statuses and fake UUIDs, and verifies genuine database persistence for in-app messaging.

---

## 2. Channel Verification

### 2.1 In-App Portal Notifications (`IN_APP`)
- **Persistence:** Genuinely creates and commits records to the `in_app_notifications` table.
- **Payload:** Stores `title`, `body`, `direction` (RTL/LTR), `metadata_json`, `person_id`, and `recipient_user_id`.
- **Read Receipts:** Tracks `is_read = False` on dispatch; updates `is_read = True` and records `read_at = now()` when marked read via `PATCH /api/v1/outreach/inbox/{id}/read`.
- **Isolation:** Scoped by `recipient_user_id` and verified against patient linkages.

### 2.2 SMS Gateway (`SMS`)
- **Config Key:** `SMS_GATEWAY_URL` & `SMS_GATEWAY_API_KEY`.
- **Configured Path:** Dispatches genuine HTTP POST via `httpx` client with JSON payload and bearer authorization.
- **Unconfigured Path:** When credentials are absent, the provider returns:
  - `status: OutreachDeliveryStatus.FAILED`
  - `provider_status: "NOT_CONFIGURED"`
  - `failure_reason: "SMS Gateway API key not configured (Set SMS_GATEWAY_API_KEY environment variable)."`
  - `provider_message_id: None` (Never invents synthetic UUIDs).

### 2.3 WhatsApp Cloud API (`WHATSAPP`)
- **Config Key:** `WHATSAPP_API_TOKEN` & `WHATSAPP_PHONE_NUMBER_ID`.
- **Configured Path:** Dispatches genuine HTTP POST to `https://graph.facebook.com/v18.0/{id}/messages`.
- **Unconfigured Path:** Returns status `FAILED`, provider status `"NOT_CONFIGURED"`, failure reason stating token absence, and `provider_message_id = None`.

### 2.4 Email (`EMAIL`)
- **Config Key:** `SMTP_HOST`, `SMTP_PORT`, `SMTP_USER`, `SMTP_PASSWORD`.
- **Configured Path:** Dispatches genuine MIME multipart email via standard `smtplib`.
- **Unconfigured Path:** Returns status `FAILED`, provider status `"NOT_CONFIGURED"`, failure reason stating SMTP host absence, and `provider_message_id = None`.

---

## 3. Audit Findings & Empirical Proof

1. `test_honest_provider_boundary_reports_not_configured_for_sms`: Verified that missing SMS gateway keys return `FAILED` with `NOT_CONFIGURED`.
2. `test_in_app_notification_inbox_and_read_lifecycle`: Verified that in-app dispatches write to `in_app_notifications` and support inbox retrieval.
3. Code inspection confirms zero instances of `uuid.uuid4()` used to fabricate provider confirmation IDs.
