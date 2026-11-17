# Phase 8 — Genomic Intelligence Architecture

**Project:** WELLWATCH PREVENT — Life-Course Preventive Care Intelligence  
**Document:** Genomic Intelligence System & Clinical Pathway Integration  
**Alignment:** Emirati Genome Program (EGP), Abu Dhabi Department of Health (DoH)  

---

## 1. Role of Genomics in Preventive Healthcare

The Emirati Genome Program (EGP) is a cornerstone of Abu Dhabi's healthcare transformation, enabling precision medicine and population-wide disease prevention. Genomic risk factors provide vital early indicators for hereditary conditions—such as Familial Hypercholesterolemia (FH) and Hereditary Breast and Ovarian Cancer (HBOC)—often decades before clinical symptoms manifest.

**Phase 8 introduces the Genomic Intelligence Subsystem:**
- Authorized ingestion of sequencing findings from accredited laboratories (e.g. G42 Healthcare / EGP).
- Clinical variant classification using standardized ACMG guidelines (`PATHOGENIC`, `LIKELY_PATHOGENIC`, `VUS`, `LIKELY_BENIGN`, `BENIGN`).
- Mapping of pathogenic variants to targeted preventive screening pathways (e.g., early lipid monitoring, accelerated breast MRI surveillance).
- Mandatory enforcement of the **VUS Safety Rule**.

---

## 2. Genomic Data Model Schema

Implemented in `backend/app/models/genomics.py` (`GenomicFinding`) and migrated via Alembic `004_genomic_intelligence.py`:

```
+---------------------------------------------------------------------------------+
|                                 genomic_findings                                |
+---------------------------------------------------------------------------------+
| id                       | VARCHAR(36)  | Primary Key (UUID)                    |
| patient_id               | VARCHAR(36)  | FK to persons.id                      |
| gene_symbol              | VARCHAR(64)  | Gene symbol (e.g. LDLR, BRCA1)        |
| variant_id               | VARCHAR(128) | dbSNP rsID or HGVS nomenclature       |
| genomic_coordinates      | VARCHAR(128) | GRCh38 genomic coordinates            |
| variant_classification   | ENUM         | PATHOGENIC, LIKELY_PATHOGENIC, VUS    |
| clinical_interpretation  | TEXT         | Laboratory diagnostic interpretation  |
| actionable_guidance      | TEXT         | Preventive surveillance protocol      |
| source_laboratory        | VARCHAR(255) | Sequencing facility                   |
| report_reference         | VARCHAR(128) | Accession number                      |
| provenance_id            | VARCHAR(36)  | FK to data_provenances.id             |
| review_status            | ENUM         | PENDING_REVIEW, VERIFIED, REJECTED    |
| reviewed_by_user_id      | VARCHAR(36)  | FK to users.id                        |
| reviewed_at              | DATETIME     | Timestamp of clinical sign-off        |
| reviewer_notes           | TEXT         | Genetic counselor rationale           |
| created_at               | DATETIME     | Ingestion audit timestamp             |
+---------------------------------------------------------------------------------+
```

---

## 3. High-Confidence Actionable Pathways

| Gene | Condition | Actionable Preventive Pathway | Clinical Surveillance Protocol |
|---|---|---|---|
| **LDLR** | Familial Hypercholesterolemia (FH) | `EARLY_CARDIOVASCULAR_LIPID_SCREENING` | Annual fasting lipid profile, ApoB, Lp(a) starting in early adulthood; lifestyle intervention. |
| **APOB** | Familial Hypercholesterolemia (FH) | `EARLY_CARDIOVASCULAR_LIPID_SCREENING` | Annual lipid evaluation; candidate for early preventive lipid-lowering therapy. |
| **BRCA1** | Hereditary Breast & Ovarian Cancer | `HIGH_RISK_BREAST_SCREENING` | Annual contrast-enhanced breast MRI beginning at age 25–30, alternating with mammography. |
| **BRCA2** | Hereditary Breast & Ovarian Cancer | `HIGH_RISK_BREAST_SCREENING` | Annual breast MRI and mammography starting age 25–30. |
| **MLH1** | Lynch Syndrome (Colorectal) | `EARLY_COLONOSCOPY_SURVEILLANCE` | High-resolution colonoscopy every 1–2 years beginning at age 20–25. |
| **MSH2** | Lynch Syndrome (Colorectal) | `EARLY_COLONOSCOPY_SURVEILLANCE` | High-resolution colonoscopy every 1–2 years beginning at age 20–25. |
