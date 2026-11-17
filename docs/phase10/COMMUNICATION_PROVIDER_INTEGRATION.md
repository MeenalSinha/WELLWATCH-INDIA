# WELLWATCH PREVENT — Communication Provider Integration & Honest Boundary

## 1. Zero-Mock Provider Architecture
In compliance with Rule 2 ("No Mock Functionality"), WELLWATCH PREVENT implements an honest communication provider interface (`ICommunicationProvider` in `backend/app/services/communication_provider.py`).

The platform never generates fake SMS receipts, fake WhatsApp blue ticks, or simulated carrier responses.

---

## 2. Channel Implementations

### 2.1 In-App Notifications (`InAppNotificationProvider`)
- Fully implemented and active.
- Delivers message directly to the patient's and provider's in-app portal database table.
- Returns status: `DELIVERED`.

### 2.2 External Providers (SMS, WhatsApp, Email)
- Implemented as pluggable gateway drivers (`SMSGatewayProvider`, `WhatsAppGatewayProvider`, `EmailGatewayProvider`).
- Detects whether enterprise gateway credentials (e.g. Twilio SID, WhatsApp Business API Key, SMTP host) are present in environment variables.
- **When credentials are not present:**
  - Accurately reports `status: "NOT_CONFIGURED"`.
  - Logs explanatory diagnostic message: `"SMS gateway not configured in current deployment environment"`.
  - Does **not** simulate network transmission or return fake delivery timestamps.

---

## 3. Provider Boundary Interface
```python
class ICommunicationProvider(ABC):
    @abstractmethod
    def send(
        self,
        recipient: str,
        message: str,
        subject: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Dispatches message or reports true integration status.
        Returns:
            {"status": "DELIVERED" | "NOT_CONFIGURED" | "FAILED", "provider_id": "...", "notes": "..."}
        """
        pass
```
