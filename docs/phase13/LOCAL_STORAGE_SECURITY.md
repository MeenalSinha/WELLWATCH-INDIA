# LOCAL STORAGE SECURITY & ENCRYPTION SPECIFICATION
## Phase 13 Client-Side Data Protection

**Project:** WELLWATCH PREVENT — Life-Course Preventive Care Intelligence  
**Target Market:** Abu Dhabi, United Arab Emirates  
**Statutory Authority:** UAE Health Data Law, ADHICS Data Protection Standards  

---

## 1. Threat Scenarios for Field Devices

Field operations carry elevated physical security risks:
- Device loss or theft during community rounds.
- Forensic extraction of unencrypted browser databases.
- Device transfer between field staff without cache purge.
- In-transit intercept when re-connecting to untrusted public cellular networks.

---

## 2. Encryption at Rest Specification

All patient identifiable data cached locally on CHW devices must be encrypted using AES-GCM (256-bit key):
1. **Key Derivation:** Client device key derived via PBKDF2 (SHA-256, 100,000 iterations) combining user credentials and hardware device fingerprint.
2. **Encrypted Storage:** Only encrypted payloads, nonces, and authentication tags are persisted in IndexedDB or localStorage.
3. **Memory Cleansing:** Decrypted demographic data is maintained strictly in transient application memory and cleared on session timeout or tab closure.

---

## 3. Cache Eviction & Remote Wipe

- **Automatic Expiration:** Cached assignment packages enforce strict expiration timestamps (`expires_at`). Upon reaching expiration, client-side logic immediately purges all cached patient records.
- **Revocation Handshake:** Every synchronization request checks device registration status (`DeviceRegistration.status`). If marked `REVOKED`, the server returns HTTP 403 Forbidden with a header instructing immediate purge of client data stores.
