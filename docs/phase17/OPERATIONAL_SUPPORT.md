# WELLWATCH PREVENT — Operational Support & Maintenance Structure
**Scope:** 24/7 Clinical Pilot & Production Support Model  
**Support Tiers:** Tier 1 (Clinic Helpdesk), Tier 2 (Application Support), Tier 3 (Core Engineering)  
**Standard:** ITIL v4 Incident Management & ADHICS v2 Operational Continuity

---

## 1. Multi-Tier Support Hierarchy

```
[Clinic Staff / CHW]
         │
         ▼
┌─────────────────────────────────┐
│     TIER 1: CLINIC HELPDESK     │  ── Account unlock, tablet connectivity, browser cache clearing
└────────────────┬────────────────┘
                 │ (Unresolved in 15m)
                 ▼
┌─────────────────────────────────┐
│  TIER 2: HEALTHCARE APP SUPPORT │  ── Sync conflict triage, patient ID matching, care gap anomalies
└────────────────┬────────────────┘
                 │ (Defect / Bug)
                 ▼
┌─────────────────────────────────┐
│    TIER 3: CORE ENGINEERING    │  ── FastEngine bugs, algorithm patches, database failover
└─────────────────────────────────┘
```

---

## 2. Service Level Agreements (SLAs)

| Incident Classification | Response SLA | Target Workaround SLA | Target Resolution SLA |
| :--- | :--- | :--- | :--- |
| **P1 - System Outage / CDS Failure** | $< 15$ Minutes | $< 45$ Minutes | $< 2$ Hours |
| **P2 - Module Impairment (e.g. Sync Blocked)** | $< 30$ Minutes | $< 2$ Hours | $< 6$ Hours |
| **P3 - Individual Workflow Anomaly** | $< 2$ Hours | $< 8$ Hours | $< 24$ Hours |
| **P4 - General Query / Minor UI Request** | $< 8$ Hours | N/A | Next Sprint |

---

## 3. Dedicated Clinical Pilot Helpdesk Channels
- **Toll-Free Emergency Hotline:** 800-WELLWATCH (Toll-free inside UAE).
- **Secure Clinical Chat:** Dedicated channel in DoH secure messaging environment for attending clinic leads.
- **On-Call Engineering Roster:** 24/7 primary and secondary on-call software architects during all pilot operational hours.
