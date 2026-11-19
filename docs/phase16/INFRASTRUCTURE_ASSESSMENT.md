# WELLWATCH PREVENT — Infrastructure Assessment & Hosting Evaluation
**Target Jurisdiction:** Abu Dhabi, United Arab Emirates  
**Compliance Standard:** ADHICS v2 Infrastructure Tiering & ISO 22301 Business Continuity  
**Version:** 1.0.0

---

## 1. Hosting Provider Evaluation (UAE Sovereign Data Zones)

Under the UAE Federal Law No. 2 of 2019 (Health Data Law) and DoH Abu Dhabi policy, all patient identifiable data and health system infrastructure must reside physically within the borders of the UAE.

| Provider / Region | Data Center Locations | ADHICS v2 Readiness | Sovereign Cloud Suitability | Recommendation |
| :--- | :--- | :--- | :--- | :--- |
| **G42 Cloud / Core42** | Abu Dhabi (Al Ain & Abu Dhabi City) | High (Primary UAE National Cloud) | Tier IV sovereign infrastructure; direct private dark fiber interconnect to DoH & SEHA | **PRIMARY (Recommended for Gov / SEHA)** |
| **Microsoft Azure (UAE Central / North)** | Abu Dhabi (Central) & Dubai (North) | High (DoH & Malaffi approved tenant tier) | Multi-AZ High Availability, ISO 27001, HIPAA & ADHICS compliance framework | **SECONDARY / HYBRID ALTERNATIVE** |
| **Oracle Cloud (Abu Dhabi Region)** | Abu Dhabi sovereign cluster | Medium-High | Excellent database performance; enterprise support | **VIABLE ALTERNATIVE** |
| **Offshore Clouds (AWS eu-west-1, etc.)** | Frankfurt, Dublin, Virginia | **PROHIBITED** | **STRICTLY PROHIBITED** by UAE Health Data Law for PHI storage | **DISQUALIFIED** |

---

## 2. Capacity Sizing & Scalability Metrics

### 2.1 Population Scale Parameters
- **Target Catchment:** 3.0 Million residents across Abu Dhabi Emirate.
- **Active Monthly Screenings:** ~150,000 adult IFHAS & pediatric well-child encounters / month.
- **Peak Hourly Transaction Rate:** ~2,500 encounters / hour (peak morning clinic hours 08:00 – 12:00 GST).
- **Network Bandwidth:** Peak ingress: 45 Mbps; Peak egress: 75 Mbps.

### 2.2 Sizing Specifications
- **FastAPI Backend Nodes:** 3 replicas (minimum), autoscaling up to 8 nodes during mass screening campaigns.
- **Frontend Next.js Nodes:** 3 replicas (minimum), autoscaling up to 6 nodes.
- **PostgreSQL Database:** 4 vCPUs, 16 GB RAM primary; 1 standby replica with automatic failover (< 30s recovery).
- **Storage Growth Estimate:** ~180 GB / year of relational records, audit trails, and FHIR transaction logs.
