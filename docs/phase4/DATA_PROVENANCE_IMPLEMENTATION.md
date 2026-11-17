# WELLWATCH PREVENT
## Data Provenance Implementation (Phase 4)

**Document ID:** GOV-PH4-01  
**Target Market:** Abu Dhabi, United Arab Emirates  
**Status:** IMPLEMENTED & VERIFIED  

---

## 1. Provenance Objectives
In healthcare intelligence platforms, data cannot be trusted without answering:
1. *Where did the observation originate?* (Source system & facility)
2. *When was the biological measurement collected vs. when was it recorded?*
3. *What transformations (e.g. unit normalization, OCR, HL7 conversion) were applied?*
4. *Who was the clinical actor or automated system responsible?*

## 2. Ingestion Provenance Workflow
```
[External Source: Clinic EMR / CHW Tablet / Lab]
                    |
                    v
          [DataSource Registry]
          - source_type: CLINIC_EMR
          - trust_level: ACCREDITED_CLINICAL
                    |
                    v
          [Normalization & Ingestion Pipeline]
          - Validates physiological bounds
          - Converts raw units (e.g. mmol/L -> mg/dL)
                    |
                    v
          [DataProvenance Generated]
          - import_timestamp: UTC ISO-8601
          - external_record_id: "SEHA-ENC-2026-991"
          - transformation_history: "Normalized 6.0 mmol/L -> 108.1 mg/dL"
          - responsible_processor_id: User UUID
                    |
                    v
          [ClinicalObservation / Condition / Screening]
          - Links to provenance_id
```

## 3. Trust Levels
- `OFFICIAL_REGULATORY`: Data directly received from Department of Health Abu Dhabi or Malaffi reciprocal queries.
- `ACCREDITED_CLINICAL`: Data ingested from certified hospital or outpatient clinic EHR systems.
- `COMMUNITY_SCREENING`: In-person screenings performed by credentialed Community Health Workers (CHWs).
- `SELF_REPORTED`: Subjective observations entered directly by patients via web or mobile portals.
