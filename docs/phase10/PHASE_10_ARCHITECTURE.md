# WELLWATCH PREVENT — Phase 10 Architecture Specification
**Culturally-Competent Multilingual Outreach Engine**

## 1. System Overview & Mandate
Phase 10 provides culturally-congruent, multilingual outreach across the Emirate of Abu Dhabi. In alignment with DoH Abu Dhabi and ADHICS v2 regulations:
- **Arabic as Primary Default:** Arabic (`ar`, RTL) is the native first-class language for citizen communication, with English (`en`, LTR) available as an explicit preference.
- **Wadeema's Law Guardian Proxy:** For pediatric patients (<18 years), outreach is strictly dispatched to verified legal guardians rather than directly to minors.
- **Consent Governance:** Patients who opt out of communication channels or revoke consent have all outbound messages automatically suppressed.
- **Honest Integration Boundaries:** The communication engine integrates via a strict `ICommunicationProvider` boundary. In-app notifications deliver directly; external channels (SMS, WhatsApp, Email) accurately record `NOT_CONFIGURED` without mock deliveries when credentials are not active.

---

## 2. Component Architecture

```
+--------------------------------------------------------------------------------+
|                           Phase 10 Architecture                                |
+--------------------------------------------------------------------------------+
|  [ PreventiveRecommendation ]             [ CommunicationPreference ]          |
|                 |                                      |                       |
|                 v                                      v                       |
|       +---------------------------------------------------------+              |
|       |                     OutreachService                     |              |
|       +---------------------------------------------------------+              |
|         | 1. Checks Consent & Opt-Out Status                    |              |
|         | 2. Evaluates Minor Status under Wadeema's Law         |              |
|         |    -> Resolves Verified Guardian Proxy if Minor       |              |
|         | 3. Enforces 7-Day Anti-Duplicate Fatigue Suppression  |              |
|         | 4. Selects Approved Template (Arabic / English)       |              |
|         | 5. Interpolates Demographic & Clinical Placeholders   |              |
|         | 6. Dispatches to Honest Provider Boundary             |              |
|         +-------------------------------------------------------+              |
|                                    |                                           |
|                     +--------------+--------------+                            |
|                     v                             v                            |
|           [ InAppProvider ]            [ ExternalChannels ]                    |
|           - Delivers to inbox          - SMS / WhatsApp / Email                |
|           - Status: DELIVERED          - Status: NOT_CONFIGURED (Honest)       |
|                     |                             |                            |
|                     +--------------+--------------+                            |
|                                    |                                           |
|                                    v                                           |
|                           [ OutreachTask ]                                     |
|                           - Tamper-Evident Delivery Audit                      |
|                           - Timestamped Status Tracking                        |
+--------------------------------------------------------------------------------+
```

---

## 3. Data Models
1. **`CommunicationPreference` (`communication_preferences` table):**
   - `person_id`: Patient link.
   - `preferred_language`: `ar` (default) or `en`.
   - `preferred_channel`: `IN_APP` (default), `SMS`, `WHATSAPP`, `EMAIL`.
   - `contact_window_start` / `contact_window_end`: Active calling/messaging hours.
   - `quiet_hours_start` / `quiet_hours_end`: Suppressed hours (e.g. 21:00 to 08:00).
   - `consent_opt_in`: Consent flag. If `False`, messages are strictly suppressed.
   - `guardian_proxy_required`: Enforced for pediatric patients.
2. **`OutreachTask` (`outreach_tasks` table):**
   - `id`: Unique task identifier.
   - `person_id`: Target patient.
   - `recipient_role`: `PATIENT` or `GUARDIAN`.
   - `recipient_contact`: Recipient phone/identifier.
   - `channel`: Communication channel.
   - `template_id`: Template used.
   - `language`: `ar` or `en`.
   - `message_body`: Rendered bilingual content.
   - `delivery_status`: `PENDING`, `DISPATCHED`, `DELIVERED`, `READ`, `FAILED`, `SUPPRESSED_OPT_OUT`, `SUPPRESSED_QUIET_HOURS`, `NOT_CONFIGURED`.
