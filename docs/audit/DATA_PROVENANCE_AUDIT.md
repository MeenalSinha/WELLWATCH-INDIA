# WELLWATCH PREVENT — Data Provenance & Suitability Audit
## Dataset Origins, Synthetic Artifacts, and Clinical Suitability Assessment

**Audited Files:** `data/raw_screening_data.csv`, `data/cleaned_data.csv`  
**Execution Script:** `scratch/audit_datasets_models.py`  
**Audit Date:** 2026-11-17  

---

## 1. Dataset Dimensions & Schema

Two tabular datasets reside in the `data/` directory:

| Property | `data/raw_screening_data.csv` | `data/cleaned_data.csv` |
|---|---|---|
| **Row Count** | 2,000 | 1,982 (18 rows dropped due to outlier/cleaning filter) |
| **Column Count** | 24 | 25 (added `age_group`) |
| **File Size** | 377,884 bytes | 393,111 bytes |
| **Duplicates** | 0 | 0 |
| **Target Column** | `risk_label` (`Low`, `Medium`, `High`) | `risk_label` (`Low`, `Medium`, `High`) |

### Column Catalog & Data Types:
- Demographics: `patient_id` (string), `age` (int), `gender` (string: Male, Female), `location` (string: Rural, Urban, Semi-Urban)
- Biometric & Physical: `height_cm` (float), `weight_kg` (float), `bmi` (float)
- Clinical Vitals: `systolic_bp` (float), `diastolic_bp` (float), `pulse_rate` (float), `fasting_glucose` (float)
- Lifestyle: `smoking` (binary), `alcohol` (binary), `physical_activity` (string: None, Low, Moderate, High), `diet_quality` (string: Poor, Average, Good)
- Family History: `family_diabetes` (binary), `family_hypertension` (binary), `family_heart_disease` (binary)
- Reported Symptoms: `fatigue` (binary), `breathlessness` (binary), `chest_pain` (binary), `frequent_urination` (binary), `blurred_vision` (binary)
- Engineered in Cleaned: `age_group` (categorical: Young, Middle, Senior)

---

## 2. Missing Value Analysis

Audit script execution revealed:

### `raw_screening_data.csv`:
- `weight_kg`: 100 missing (5.0%)
- `pulse_rate`: 100 missing (5.0%)
- `fasting_glucose`: 100 missing (5.0%)
- `physical_activity`: 705 missing (35.25%) — artificially injected in notebook or raw generator

### `cleaned_data.csv`:
- `weight_kg`: 0 missing (imputed with median: 65.0 kg)
- `pulse_rate`: 0 missing (imputed with median: 75.0 bpm)
- `fasting_glucose`: 0 missing (imputed with median: 104.7 mg/dL)
- `physical_activity`: **700 missing values remain uncleaned!** (35.3% missingness). In notebook cell 34, `numerical_cols` were imputed with median, but categorical `physical_activity` was mapped to integer `activity_encoded` via `.map({'None': 0, 'Low': 1, 'Moderate': 2, 'High': 3})`, converting missing values directly to `NaN`!

---

## 3. Data-Source Provenance & Synthetic Nature

### 3.1 Provenance Determination
- **Source:** 100% synthetically generated in Python using `numpy.random` inside `WELLWATCH_INDIA.ipynb` (Cell 17: `generate_synthetic_data()`).
- **External Data:** None. Zero external records (e.g. NHANES, NFHS, or hospital records) were incorporated.
- **Generation Logic:** 
  - Demographics generated from flat distributions: `age ~ Uniform(18, 80)`, `gender ~ Bernoulli(0.52)`
  - Continuous measurements generated from Gaussian distributions: `height_cm ~ Normal(160, 10)`, `systolic_bp ~ Normal(125, 18)`
  - Target labels constructed using an explicit arithmetic formula with Gaussian noise $\epsilon \sim \mathcal{N}(0, 10)$.

### 3.2 Target Class Distribution
- **Raw Data:**
  - Medium: 49.35% (987 / 2,000)
  - Low: 25.65% (513 / 2,000)
  - High: 25.00% (500 / 2,000)
- **Cleaned Data:**
  - Medium: 49.04% (972 / 1,982)
  - Low: 25.73% (510 / 1,982)
  - High: 25.23% (500 / 1,982)

---

## 4. Assessment of Suitability for Target Market (Abu Dhabi, UAE)

| Dimension | Legacy Dataset Characteristics | Abu Dhabi / UAE Healthcare Requirements | Suitability Verdict |
|---|---|---|---|
| **Geographic Context** | Rural / Semi-Urban / Urban Indian villages | Abu Dhabi emirate (Abu Dhabi City, Al Ain, Al Dhafra) | **Unsuitable** |
| **Demographic Pyramid** | General rural Indian population, age 18-80 | Diverse UAE population (UAE Nationals & expatriates, pediatric to geriatric) | **Unsuitable** |
| **Clinical Focus** | Coarse adult lifestyle chronic risk | Comprehensive Life-Course (Pediatric Well-Child Visits + IFHAS adult screening) | **Unsuitable** |
| **Genomic Risk** | None | UAE National Genome Strategy integration readiness | **Missing** |
| **Pediatric Coverage** | None (Min age: 18) | Pediatric Well-Child intervals (Birth to 6 years) | **Completely Missing** |
| **Regulatory Standing** | Synthetic data without consent or IRB | Subject to ADHICS, Federal Law No. 2/2019, DoH standards | **Unsuitable for clinical use** |

---

## 5. Summary & Recommendation

The legacy datasets `raw_screening_data.csv` and `cleaned_data.csv` are purely synthetic artifacts designed for a hackathon demonstration of adult rural Indian screening. They possess zero clinical validity, lack pediatric screening data, and cannot be utilized for preventive care decision-making in Abu Dhabi.

**Recommendation:**
1. Isolate legacy datasets in `data/legacy/` strictly as regression test fixtures.
2. Label all records explicitly as synthetic (`is_synthetic: true`).
3. For Phase 2–3 development, establish a formal synthetic data generator following reproducible clinical protocols and strict metadata tagging, completely separated from future production clinical repositories.
