# GEOGRAPHIC ANALYTICS & REGIONAL HIERARCHY
## Abu Dhabi Public Health Spatial Intelligence

**Project:** WELLWATCH PREVENT — Life-Course Preventive Care Intelligence  
**Target Market:** Abu Dhabi, United Arab Emirates  

---

## 1. Abu Dhabi Geographic Hierarchy

Spatial epidemiology is structured according to the official administrative hierarchy of the Emirate of Abu Dhabi:

1. **Emirate Level:** Abu Dhabi (`ARE-AZ`)
2. **Municipal Region Level:**
   - **Abu Dhabi Central / Capital Region (`AD_CENTRAL` / أبوظبي العاصمة):** High-density urban population; primary academic medical centers (CCAD, Sheikh Shakhbout Medical City).
   - **Al Ain Region (`AL_AIN` / العين - المنطقة الشرقية):** Eastern municipal region; suburban and agricultural communities (Tawam Hospital, Al Ain Hospital).
   - **Al Dhafra Region (`AL_DHAFRA` / الظفرة - المنطقة الغربية):** Western region; geographically dispersed communities, desert settlements, oil & gas industrial zones (Madinat Zayed, Ghayathi, Ruwais, Liwa).
3. **District Level:** Mapped via `Person.district` (e.g. Al Bateen, Al Reem Island, Al Jimi, Madinat Zayed).

---

## 2. Spatial Aggregation & Small-Cell Privacy Protection

Epidemiological calculations aggregate at the region level with automatic privacy gating:

$$\text{Cell Metric} = \begin{cases} \text{Numeric Count \& Rate}, & \text{if } N \ge 5 \\ \text{"SUPPRESSED" \& null}, & \text{if } N < 5 \end{cases}$$

This guarantees that sparse populations in remote Al Dhafra districts cannot be singled out through public health dashboards.
