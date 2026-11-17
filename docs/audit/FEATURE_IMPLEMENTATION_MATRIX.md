# WELLWATCH PREVENT — Feature Implementation Matrix
## Detailed Audit of Existing Application Features

**Application:** `streamlit_app.py`  
**Evaluation Standard:** Production & Clinical Readiness  
**Date:** 2026-11-17  

---

## 1. Classification Categories
- **Fully Implemented and Verified:** Works end-to-end with real persistence, validation, and verified backend logic.
- **Implemented with Limitations:** Works, but constrained by design flaws, incomplete validation, or edge-case failures.
- **Partially Implemented:** Some logic exists, but key components (persistence, pipeline, or output) are absent.
- **Mocked or Simulated:** Appears functional on UI but backed by fake delays, static mocks, or dummy responders.
- **Hardcoded:** Logic or data relies on fixed constants, static dictionaries, or arbitrary heuristics.
- **Broken:** Throws runtime exceptions, fails to load, or fails to execute.
- **Missing:** Specified in requirements/README but completely absent from the codebase.

---

## 2. Feature-by-Feature Matrix

| Feature / Sub-Feature | Claimed Status (README) | Actual Status | Code Location | Finding & Evidence |
|---|---|---|---|---|
| **Home Dashboard** | Operational | **Hardcoded** | `streamlit_app.py:415-460` | Displays static metrics (e.g., "72.8% Accuracy", "28 Features", "1,600 Train"). Charts use fixed static layouts. |
| **Patient Screening Form** | Real-time prediction | **Implemented with Limitations** | `streamlit_app.py:590-730` | Form gathers 21 inputs with validation, but does not execute the trained XGBoost model. |
| **Risk Assessment Engine** | XGBoost AI Inference | **Hardcoded** | `streamlit_app.py:232-330`, `741` | Calls `predict_risk_simple()` instead of `load_model_artifacts()`. Uses hardcoded rule arithmetic and fixed probability dictionaries. |
| **Risk Visualization (Gauge)** | Interactive Plotly | **Fully Implemented and Verified** | `streamlit_app.py:331-365`, `770-800` | Gauge chart properly renders the calculated `risk_score` (0-100) using Plotly `go.Indicator`. |
| **Risk Breakdown Chart** | Real-time Breakdown | **Implemented with Limitations** | `streamlit_app.py:805-835` | Plots probabilities from `predict_risk_simple`, but values are hardcoded constants (e.g. 0.75, 0.20, 0.05). |
| **Save to Database** | DB Persistence | **Mocked or Simulated** | `streamlit_app.py:860-862` | `st.button("💾 Save to Database")` directly calls `st.success(...)` with no SQL execution. Zero rows saved. |
| **Send to Patient (SMS/Email)** | Multi-channel Alerts | **Mocked or Simulated** | `streamlit_app.py:856-858` | `st.button("📱 Send to Patient")` directly calls `st.success(...)` with no network, Twilio, or SMTP call. |
| **PDF Report Generation** | PDF Generation | **Mocked or Simulated** | `streamlit_app.py:852-854` | `st.button("📄 Generate Report")` directly calls `st.success(...)`. Notebook has FPDF code in Cell 128, but it was never wired to the UI button. |
| **Community Health Dashboard** | Real-time Analytics | **Hardcoded** | `streamlit_app.py:870-1015` | Calls `np.random.seed(42)` and generates synthetic samples using `np.random.choice(...)` on every page load. Reads no actual DB data. |
| **Model Explainability (SHAP)** | Explainability | **Implemented with Limitations** | `streamlit_app.py:1020-1120` | Renders pre-generated static PNG images from `outputs/shap_summary.png` and `outputs/feature_importance.png`. No dynamic patient-level SHAP explanation. |
| **Subgroup Fairness Analysis** | Fairness Evaluation | **Implemented with Limitations** | `streamlit_app.py:1125-1160` | Displays pre-calculated static subgroup bar charts. Does not compute live fairness metrics across patient records. |
| **Judge / Demo Quick Test** | 5 Test Cases | **Hardcoded** | `streamlit_app.py:1260-1420` | Evaluates 5 pre-canned patient cases using `predict_risk_simple()`. Results are deterministic outputs of hardcoded formula. |
| **Admin Panel - View Records** | DB Explorer | **Hardcoded** | `streamlit_app.py:1435-1460` | Creates a DataFrame using `np.random.randint(25, 75, 20)` and random strings. Fails to query `data/wellwatch.db`. |
| **Admin Panel - Export Data** | Multi-format Export | **Implemented with Limitations** | `streamlit_app.py:1510-1555` | Generates downloadable CSV/Excel of the dynamically generated random records, not real database records. |
| **Admin Panel - Sync Now** | Synchronization | **Mocked or Simulated** | `streamlit_app.py:1590-1596` | Shows `time.sleep(1.5)` spinner followed by `st.success("✅ 5 records synced successfully!")`. Zero synchronization logic. |
| **Admin Panel - Create Backup** | Automated Backups | **Mocked or Simulated** | `streamlit_app.py:1614-1618` | Shows `time.sleep(2)` spinner followed by `st.success("✅ Backup created successfully!")`. No file copied. |
| **Admin Panel - Restore Backup** | Backup Restoration | **Broken** | `streamlit_app.py:1620-1623` | Shows `st.warning("⚠️ Restore will overwrite current data. Confirm?")` but provides no confirmation handler or file restore code. |
| **Offline Mode Toggle** | Offline Capability | **Partially Implemented** | `streamlit_app.py:398-410`, `1582-1589` | Toggle toggles a CSS banner `offline-indicator`. Does not configure local storage, IndexedDB, SQLite sync queue, or offline caching. |
| **Voice Alerts** | Low-literacy Support | **Missing** | `WELLWATCH_INDIA.ipynb:Cell 132` | Cell 132 in notebook defines `generate_voice_alert()`, but feature is completely omitted from `streamlit_app.py`. |
| **User Authentication & RBAC** | Enterprise Security | **Missing** | Nowhere in repo | No login screen, session auth, API tokens, password hashes, or role checks exist. Anyone with access to the URL has full admin access. |

---

## 3. Detailed Evidence of Critical Deficiencies

### 3.1 Model Integration Defect
- **File:** `streamlit_app.py`
- **Line:** 220–230
- **Code:**
  ```python
  @st.cache_resource
  def load_model_artifacts():
      try:
          model = joblib.load('models/chronic_risk_model.pkl')
          scaler = joblib.load('models/scaler.pkl')
          label_encoder = joblib.load('models/label_encoder.pkl')
          with open('models/feature_list.json', 'r') as f:
              features = json.load(f)['features']
          return model, scaler, label_encoder, features
      except:
          return None, None, None, None
  ```
- **Finding:** The function `load_model_artifacts()` is defined but never invoked in `streamlit_app.py`. The model pipeline is dormant.

### 3.2 Mocked Database Operations
- **File:** `streamlit_app.py`
- **Line:** 860–862
- **Code:**
  ```python
  if st.button("💾 Save to Database", use_container_width=True):
      st.success("✅ Record saved to database!")
  ```
- **Finding:** Displays success without opening a database connection or executing an SQL INSERT statement.

### 3.3 Dynamic Randomness in Analytics
- **File:** `streamlit_app.py`
- **Line:** 870–883
- **Code:**
  ```python
  np.random.seed(42)
  data = {
      'Location': np.random.choice(['Rural', 'Semi-Urban', 'Urban'], sample_size, p=[0.60, 0.25, 0.15]),
      'Risk_Level': np.random.choice(['Low', 'Medium', 'High'], sample_size, p=[0.55, 0.30, 0.15]),
      'Age_Group': np.random.choice(['18-30', '31-45', '46-60', '60+'], sample_size),
  ...
  ```
- **Finding:** The community analytics page creates synthetic random numbers on every render rather than reading persisted data.

---

## 4. Remediation Strategy for WellWatch Prevent
1. Retire the monolithic Streamlit script from production routing.
2. Build an API-driven backend where all risk predictions pass through verified, calibrated, versioned models.
3. Replace all cosmetic UI buttons with real RESTful POST/PUT operations backed by persistent PostgreSQL storage.
4. Establish true RBAC, preventing unauthorized role transitions and securing patient records.
