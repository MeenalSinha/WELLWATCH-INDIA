# WELLWATCH PREVENT — Post-Phase-17 Future Research & Development Roadmap
**Strategic Focus:** Multi-Omics Integration & Middle-Eastern Calibrated Polygenic Risk Scores (PRS)  
**Research Collaboration:** Emirati Genome Program (EGP) & M42 Life Sciences  
**Status:** Decoupled Future Research Initiative (Post-Phase-17 Horizon)

---

## 1. Rationale for Architectural & Operational Decoupling
Earlier preliminary documentation contemplated incorporating polygenic risk scoring (PRS) into the core operational phases. During Phase 15 clinical safety appraisal (ISO 14971), the Clinical Safety Officer mandated formal **decoupling of experimental multi-omics from the active clinical screening software**:
1. **Clinical Regulatory Focus:** The current production deployment (Phases 0–17) is tightly aligned with approved, evidence-backed public health guidelines (ADPHC 2026 IFHAS and Well-Child).
2. **Lack of Arab Ancestry Calibration:** Most publicly available PRS models (e.g. from UK Biobank) exhibit severe calibration degradation when applied to Arab and Emirati ancestries, introducing potential health disparity risks.
3. **Safety Isolation:** Experimental genomics must undergo dedicated multi-center prospective validation trials before being allowed to influence preventive clinical recommendations in primary care.

---

## 2. Multi-Year Future Research Horizons

```
[Phases 0–17: Core Production & Pilot] ──> [Horizon 1: Emirati Genome Program Linkage]
                                                        │
                                                        ▼
[Horizon 3: Multi-Omic Metabolic Integration] <── [Horizon 2: Arab-Specific PRS Calibration]
```

---

## 3. Detailed Future Horizons

### Horizon 1: Secure Data Linkage with Emirati Genome Program (Months 12–18)
- **Objective:** Establish sovereign federated data exchange between WELLWATCH PREVENT and the Emirati Genome Program (EGP) under DoH Research Ethics oversight.
- **Privacy Architecture:** Privacy-preserving record linkage (PPRL) using encrypted homomorphic hashes to link clinical phenotypes with genomic biobank IDs without exposing raw genomic sequence data.

### Horizon 2: Middle-Eastern Calibrated Polygenic Risk Scores (Months 18–30)
- **Target Conditions:**
  1. Early-Onset Type 2 Diabetes Mellitus.
  2. Premature Coronary Artery Disease (CAD).
  3. Familial Hypercholesterolemia and Early Colorectal Carcinoma.
- **Ancestry Calibration:** Re-weighting variant effect sizes using Emirati Genome Program reference panels to eliminate European ancestry bias.
- **Regulatory Gate:** Clinical trial authorization from the DoH Human Research Committee prior to any patient-facing risk display.

### Horizon 3: Multi-Omics Precision Prevention (Months 30–42)
- **Scope:** Integration of targeted lipidomics, metabolomic markers (e.g., branched-chain amino acids for insulin resistance), and epigenetic age acceleration markers.
- **Clinical Integration:** Serving as second-tier risk stratification for individuals with borderline clinical risk scores.
