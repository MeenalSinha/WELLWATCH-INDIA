# PRIVACY & SMALL-CELL SUPPRESSION FRAMEWORK
## De-Identification & Re-Identification Mitigation

**Project:** WELLWATCH PREVENT — Life-Course Preventive Care Intelligence  
**Target Market:** Abu Dhabi, United Arab Emirates  
**Statutory Authority:** UAE Health Data Law, ADPHC Information Governance Policy  

---

## 1. Regulatory Background

Under UAE Federal Law No. 2 of 2019 Concerning the Use of Information and Communication Technology in Health Fields (UAE Health Data Law) and the Abu Dhabi Department of Health Information and Cyber Security Standard (ADHICS), health data published or shared for secondary analysis must prevent re-identification of data subjects.

When aggregating data by sensitive demographics (such as rare conditions, genomic variants, or localized geographic sectors), small cell counts ($N < 5$) pose an elevated risk of deductive re-identification ("singling out").

---

## 2. Suppression Policy Specification

Governed in `configs/population_policy.json`:

```json
{
  "small_cell_suppression": {
    "enabled": true,
    "min_cell_size": 5,
    "suppression_mask": "SUPPRESSED",
    "apply_to": [
      "geographic_analytics",
      "demographic_subgroups",
      "rare_disease_aggregates"
    ]
  }
}
```

---

## 3. Implementation Rules

1. **Primary Cell Suppression:**
   - If `registered_population < 5`, then `registered_population = "SUPPRESSED"`, `eligible_population = "SUPPRESSED"`, `completed_screenings = "SUPPRESSED"`, and `screening_coverage_percentage = null`.
   - The cell response sets `is_suppressed = true`.

2. **Secondary (Complementary) Suppression:**
   - If an attacker could derive the suppressed cell value by subtracting published cells from a known marginal total, the engine suppresses the next smallest non-zero cell or masks the marginal total.

3. **Rate Suppression:**
   - Percentages and rates are never published for suppressed cells to prevent reverse-engineering the count from a percentage.
