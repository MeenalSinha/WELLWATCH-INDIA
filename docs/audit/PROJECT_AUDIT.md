# WELLWATCH PREVENT — Phase 0 Project Audit
## Comprehensive Repository and Architecture Audit

**Repository:** `WELLWATCH-INDIA`  
**Evolution Target:** `WELLWATCH PREVENT — Life-Course Preventive Care Intelligence`  
**Date of Audit:** 2026-11-17  
**Auditor Role:** Senior Healthcare Technology Architect, Principal Full-Stack Engineer, ML Engineer  

---

## 1. Executive Summary

An exhaustive technical, architectural, and clinical integrity audit of the `WELLWATCH-INDIA` repository was executed on 2026-11-17. The repository represents a prototype submitted for the "AI4Bharat Hackathon 2025" aimed at chronic lifestyle disease risk detection in rural India.

While the repository contains an initial exploratory Jupyter Notebook (`WELLWATCH_INDIA.ipynb`), a serialized XGBoost model pipeline (`models/`), and a 1,906-line monolithic Streamlit web application (`streamlit_app.py`), the audit revealed **critical foundational defects, architectural anti-patterns, and misleading claims** that disqualify the legacy codebase from immediate production or clinical use:
1. **Disconnected ML Model:** The Streamlit web application defines a function `load_model_artifacts()`, but never invokes it anywhere in the application. Instead, all patient screenings and judge demonstration tests invoke `predict_risk_simple()`, which computes risk scores using arbitrary hand-crafted arithmetic rules and returns hardcoded probability distributions (`{'Low': 0.75, 'Medium': 0.20, 'High': 0.05}`, etc.).
2. **Methodological ML Flaws & Leakage:** The model was trained entirely on synthetic data generated via an ad-hoc scoring formula plus Gaussian noise. Feature scaling (`StandardScaler`) and missing-value median imputation were fitted across the entire 1,982 records **prior to train/test splitting**, introducing data leakage.
3. **Cosmetic Stubs & Fake Functionality:** UI action buttons including "Save to Database", "Send to Patient", "Generate Report", "Sync Now", and "Create Backup" are cosmetic illusions; clicking them displays an instant `st.success(...)` banner without performing SQL writes, SMS transmissions, or backup creations.
4. **Fabricated Dashboard & Admin Data:** The "Community Dashboard" and "Admin Panel - View Records" do not read from `wellwatch.db` or CSV datasets; they invoke `np.random` to fabricate patient distributions, ages, and risk levels dynamically.
5. **No Authentication, Authorization, or Privacy Controls:** The legacy application contains zero authentication, zero role-based access control, no data minimization, and stores unencrypted health metrics in SQLite without access controls.

Consequently, `WELLWATCH-INDIA` cannot be directly extended. It must be carefully preserved as a legacy research baseline while constructing a secure, modular, API-first production architecture for `WELLWATCH PREVENT`.

---

## 2. Complete Repository Inventory

The repository comprises 33 files across 4 directories totaling ~19.5 MB. Every file was inspected, measured, and classified:

| Path | Size (Bytes) | Category | Description & Role | Audit Assessment |
|---|---|---|---|---|
| `README.md` | 2,865 | Documentation | High-level overview, quick start, claimed metrics (72.8% acc) | Claims 72.8% accuracy; omits that UI uses hardcoded heuristics. |
| `DEPLOYMENT.md` | 2,080 | Documentation | Deployment guidelines for Streamlit, Heroku, AWS | Contains outdated, insecure deployment suggestions. |
| `LICENSE` | 1,089 | Legal | MIT License text | Standard open-source license. |
| `requirements.txt` | 141 | Dependency Spec | 12 unpinned Python libraries (`scikit-learn`, `xgboost`, etc.) | Unpinned versions; missing web framework, auth, test suites. |
| `streamlit_app.py` | 74,883 | Application | 1,906 lines single-file Streamlit web app | Monolithic; uncalled ML model; fake actions; hardcoded visuals. |
| `WELLWATCH_INDIA.ipynb` | 2,084,561 | Notebook | 168 cells covering synthetic data generation, EDA, training | Complete workflow; data leakage present; synthetic data origin. |
| `data/raw_screening_data.csv` | 377,884 | Dataset | 2,000 synthetic patient rows, 24 columns | 100% synthetic; contains missing values (weight, glucose, pulse). |
| `data/cleaned_data.csv` | 393,111 | Dataset | 1,982 rows, 25 columns | Filtered data; still has 700 missing `physical_activity` rows. |
| `data/wellwatch.db` | 24,576 | Database | SQLite 3 database with 3 tables | Holds 5 dummy patient records; `interventions` table empty. |
| `models/chronic_risk_model.pkl` | 1,802,718 | ML Artifact | Pickled `XGBClassifier` (200 estimators, max_depth 6) | Functional when tested via custom script; uncalled in web app. |
| `models/scaler.pkl` | 1,079 | ML Artifact | Pickled `StandardScaler` (fitted on 6 numerical features) | Version mismatch (sklearn 1.6.1 vs 1.8.0); fitted pre-split. |
| `models/label_encoder.pkl` | 495 | ML Artifact | Pickled `LabelEncoder` (classes: High, Low, Medium) | Alphabetical order (0: High, 1: Low, 2: Medium). |
| `models/feature_list.json` | 641 | Metadata | JSON list of 28 feature names in expected order | Accurate to model expectations. |
| `models/model_metadata.json` | 348 | Metadata | JSON metadata: training date, metrics, sample counts | Matches notebook output metrics (72.80% accuracy). |
| `outputs/evaluation_metrics.csv` | 427 | Report | CSV of overall and per-class precision, recall, F1 | Accurate to notebook test split; test split suffered from leakage. |
| `outputs/feature_importance.csv` | 772 | Report | XGBoost feature importance values | High BP, age, glucose ranked top. |
| `outputs/*.png` (12 files) | ~4.5 MB | Visualizations | Confusion matrices, SHAP plots, calibration, fairness plots | Generated from notebook execution. |
| `outputs/*.html` (3 files) | ~13.9 MB | Visualizations | Standalone Plotly interactive HTML plots | Large static plots embedded in outputs. |
| `outputs/health_report.pdf` | 2,416 | Generated Doc | Sample PDF health report generated by FPDF | Minimal static prototype PDF. |

---

## 3. Architecture & Code Structure Audit

### 3.1 Monolithic Streamlit Architecture (`streamlit_app.py`)
- **Structure:** 1,906 lines of procedural Python with only 4 function definitions (`load_model_artifacts`, `predict_risk_simple`, `create_gauge_chart`, `highlight_risk`). All page routing is handled via a top-level sidebar radio button and session state overrides.
- **State Management:** Heavy reliance on global Streamlit execution flow and `st.session_state`.
- **Presentation Logic vs Business Logic:** Tightly coupled. HTML CSS styles are injected inline via `st.markdown(..., unsafe_allow_html=True)`. Database connection strings, SQL statements, and chart definitions are interwoven directly into UI components.
- **Error Handling:** Minimal to non-existent. In `load_model_artifacts()`:
  ```python
  except:
      return None, None, None, None
  ```
  This bare `except` masks file path errors, version incompatibilities, and deserialization failures.

### 3.2 Machine Learning Disconnect
The model artifact loading function:
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
Searches for invocations of `load_model_artifacts` across the entire codebase yield exactly **zero calls**.
When a screening form is submitted (line 741):
```python
result = predict_risk_simple(patient_data)
```
And in Judge Mode (line 1303):
```python
results[case_name] = predict_risk_simple(patient_data)
```
The actual trained XGBoost model (`models/chronic_risk_model.pkl`) is completely ignored in production runtime.

### 3.3 Rule-Based Inference Heuristics (`predict_risk_simple`)
`predict_risk_simple` calculates an ad-hoc score:
```python
risk_score = 0
if patient_data['age'] > 60: risk_score += 20
elif patient_data['age'] > 45: risk_score += 10
# ...
if risk_score < 35:
    risk_level = 'Low'
    prob = {'Low': 0.75, 'Medium': 0.20, 'High': 0.05}
elif risk_score < 65:
    risk_level = 'Medium'
    prob = {'Low': 0.25, 'Medium': 0.60, 'High': 0.15}
else:
    risk_level = 'High'
    prob = {'Low': 0.10, 'Medium': 0.25, 'High': 0.65}
```
The returned "probabilities" are static dictionary literals, not model softmax or logistic outputs.

### 3.4 Cosmetic Stubs & False Success Actions
Multiple user actions display deceptive success notifications without performing any backend operations:
- **Save to Database (Line 861):**
  ```python
  if st.button("💾 Save to Database", use_container_width=True):
      st.success("✅ Record saved to database!")
  ```
  No SQLite connection, no `cursor.execute("INSERT ...")`, no persistence.
- **Send to Patient (Line 857):**
  ```python
  if st.button("📱 Send to Patient", use_container_width=True):
      st.success("✅ SMS/Email sent to patient!")
  ```
  No telephony or SMTP integration.
- **Generate Report (Line 853):**
  ```python
  if st.button("📄 Generate Report", use_container_width=True):
      st.success("✅ PDF report generated successfully!")
  ```
  Does not call `generate_health_report_pdf()` from the notebook.
- **Sync Now (Line 1591-1595):**
  ```python
  if st.button("🔄 Sync Now", use_container_width=True):
      with st.spinner("Syncing data..."):
          time.sleep(1.5)
          st.success("✅ 5 records synced successfully!")
  ```
  Sleeps for 1.5 seconds and returns a hardcoded success message.
- **Create Backup (Line 1614-1618):**
  ```python
  if st.button("💾 Create Backup", use_container_width=True):
      with st.spinner("Creating backup..."):
          time.sleep(2)
          st.success("✅ Backup created successfully!")
  ```
  Sleeps for 2 seconds with zero file operations.

### 3.5 Fabricated Dashboard Analytics
- **Community Health Analytics (Lines 872-882):**
  ```python
  np.random.seed(42)
  'Location': np.random.choice(['Rural', 'Semi-Urban', 'Urban'], sample_size, p=[0.60, 0.25, 0.15]),
  'Risk_Level': np.random.choice(['Low', 'Medium', 'High'], sample_size, p=[0.55, 0.30, 0.15]),
  ```
- **Admin Panel Record Table (Lines 1439-1447):**
  ```python
  'Age': np.random.randint(25, 75, 20),
  'Risk_Level': np.random.choice(['Low', 'Medium', 'High'], 20, p=[0.55, 0.30, 0.15]),
  ```
  The dashboard and admin tables fabricate random numbers on the fly rather than querying the 5 rows in `data/wellwatch.db` or the 1,982 rows in `data/cleaned_data.csv`.

---

## 4. Technical Debt & Safety Risks

1. **Clinical Safety Risk:** Displaying arbitrary rule-based points as "AI Clinical Probabilities" poses severe patient safety hazards if relied upon for triage or screening recommendations.
2. **Data Integrity Risk:** False database save notifications mislead clinicians into believing records are persisted when they are lost upon browser refresh.
3. **Security Deficit:** Total absence of authentication allows unvetted access to any endpoint or record. SQLite database lacks file encryption and table-level access controls.
4. **Serialization Vulnerability:** Model artifacts are stored as Python `.pkl` files without signature verification or checksum validation.

---

## 5. Architectural Verdict

The existing repository demonstrates exploratory data science work and a rapid hackathon demo script, but possesses zero production engineering or clinical software foundation. 

**Recommendation:**
- Preserve all existing files in place as legacy historical references under `docs/audit/` and `ml/artifacts/`.
- Transition the core architecture to a modular, API-first architecture using **FastAPI (Backend)** and **Next.js / TypeScript (Frontend)** with **PostgreSQL** and **Alembic**, establishing strict clinical governance, true model pipeline execution, real database persistence, and comprehensive RBAC security.
