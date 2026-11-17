# WELLWATCH PREVENT — Security & Dependency Audit
## Comprehensive Vulnerability, Credential, and Risk Assessment

**Repository:** `WELLWATCH-INDIA`  
**Audit Date:** 2026-11-17  
**Scope:** Dependencies, Static Code, Authentication, Data Protection, Network  

---

## 1. Dependency Analysis & Package Vulnerabilities

The legacy project specifies dependencies in `requirements.txt`:
```text
scikit-learn
xgboost
pandas
numpy
matplotlib
seaborn
plotly
shap
fpdf
streamlit
joblib
pyngrok
```

### Critical Dependency Findings:
1. **Unpinned Versions:** None of the packages have version pins or hash locks (`==`). In modern Python environments (e.g. Python 3.12 / 3.14), unpinned installs lead to breaking changes and silent ABI incompatibilities.
2. **`pyngrok` in Production Requirements:** `pyngrok` exposes local ports to public internet tunnels. Including tunneling tools in a healthcare repository creates serious security risks if run in hospital or clinical environments.
3. **`fpdf` Deprecation:** The `fpdf` package is unmaintained (last release 2012) and has known formatting and text encoding issues. Modern applications require `fpdf2` or `weasyprint`.
4. **`joblib` / `pickle` Deserialization:** The application uses `joblib.load()` on `.pkl` files. In Python, unpickling untrusted serialized files is vulnerable to Arbitrary Code Execution (CWE-502).

---

## 2. Secrets & Credential Exposure Scan

A recursive scan across all files, git history, and commit logs was performed:
- **API Keys / Passwords:** None discovered in plaintext.
- **Git Commit Inspection:** No committed `.env` files or API secrets found in git commits.
- **Hardcoded Endpoints:** In `DEPLOYMENT.md`, placeholders `support@wellwatch.in`, `devops@wellwatch.in`, and `https://api.wellwatch.in/predict` are listed.

---

## 3. Authentication & Authorization Gaps

1. **Zero Access Control:** `streamlit_app.py` has no authentication mechanism. Any user who reaches the URL has unrestricted access to the patient screening form, community analytics, judge test panel, and admin database viewer.
2. **Missing Multi-Tenancy:** No concept of organizational or facility boundaries exists.
3. **Missing Role-Based Access Control (RBAC):** No distinction exists between a Community Health Worker, a Physician, a System Administrator, or an unauthorized visitor.

---

## 4. Insecure Data Handling & Privacy Deficiencies

1. **Plaintext SQLite Storage:** `wellwatch.db` stores biometric, lifestyle, and clinical health indicators unencrypted on disk.
2. **Absence of Data Minimization:** Forms collect sensitive lifestyle attributes (e.g. alcohol, smoking, symptoms) without consent documentation, purpose specification, or data retention schedules.
3. **No Audit Trails:** The application does not log who accessed, created, or queried patient records. Under UAE Federal Law No. 2 of 2019 and Abu Dhabi ADHICS standards, audit logging of health record access is mandatory.
4. **Insecure Offline Mode:** The "Offline Mode" is a purely cosmetic toggle that sets a CSS banner; it provides no cryptographic offline storage, encrypted local caching, or authenticated synchronization protocol.

---

## 5. Input Validation & Error Handling

1. **Insecure Error Suppressions:** In `streamlit_app.py:229`, a blanket `except:` block masks all exceptions during model loading.
2. **No Backend API Validation:** The application directly binds Streamlit form inputs to Python dictionaries without Pydantic schema validation or range constraints.
3. **HTML / Script Injection:** Inline styling and HTML rendering via `st.markdown(..., unsafe_allow_html=True)` is used extensively across `streamlit_app.py`. If user inputs are rendered unsanitized into these templates, Cross-Site Scripting (XSS) vulnerabilities occur.

---

## 6. Security Baseline Recommendations for WellWatch Prevent
1. **Remove Unsafe Tools:** Eliminate `pyngrok` from requirements.
2. **Implement OIDC / OAuth2 with PKCE:** Secure all endpoints via standard JWT tokens with short expiry and refresh rotation.
3. **Enforce Backend RBAC:** Implement strict role-to-permission mapping on every API route.
4. **Audit Logging & Encryption:** Implement tamper-evident database audit logs for all security and PHI access events; enforce TLS 1.3 in transit and AES-256 at rest.
5. **Secure Middleware:** Add CSP, HSTS, X-Content-Type-Options, CORS origin validation, and rate limiting.
