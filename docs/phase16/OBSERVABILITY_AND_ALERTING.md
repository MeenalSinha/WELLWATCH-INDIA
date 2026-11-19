# WELLWATCH PREVENT — Observability & Alerting Architecture
**Standards:** OpenTelemetry, Prometheus / Grafana, ADHICS v2 Audit Monitoring  
**Target SLAs:** 99.95% Availability, API p95 Latency < 200 ms, Zero Unmonitored Failures

---

## 1. Observability Framework Architecture

```
┌─────────────────────────────────┐     ┌─────────────────────────────────┐
│     FASTAPI BACKEND METRICS     │     │      NEXT.JS FRONTEND METRICS   │
│   (Prometheus Middleware / OTel)│     │     (Web Vitals / Server Timing)│
└────────────────┬────────────────┘     └────────────────┬────────────────┘
                 │                                       │
                 ▼                                       ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                   METRICS COLLECTOR & AGGREGATOR                        │
│                       (Prometheus Server / Cortex)                      │
└────────────────────────────────────┬────────────────────────────────────┘
                                     │
                 ┌───────────────────┴───────────────────┐
                 ▼                                       ▼
┌─────────────────────────────────┐     ┌─────────────────────────────────┐
│      GRAFANA DASHBOARDS         │     │         ALERTMANAGER            │
│  (Clinical & Operational Views) │     │ (PagerDuty / OpsGenie / Webhook)│
└─────────────────────────────────┘     └─────────────────────────────────┘
```

---

## 2. Four Golden Signals Monitoring

| Signal | Target Metric | Warning Threshold | Critical Alert Threshold | Action Plan |
| :--- | :--- | :--- | :--- | :--- |
| **Latency** | p95 < 200 ms | p95 > 350 ms for 5m | p95 > 500 ms for 2m | Auto-scale backend workers; inspect slow queries |
| **Traffic** | 50 – 500 req/s | > 1,000 req/s | > 2,500 req/s (spike) | Activate WAF rate limiting; inspect DDoS or sync loop |
| **Errors** | HTTP 5xx < 0.05% | 5xx > 0.5% for 3m | 5xx > 2.0% for 1m | Page on-call engineer; check DB connectivity |
| **Saturation** | CPU < 60%, RAM < 70% | CPU > 75%, RAM > 80% | CPU > 90%, RAM > 90% | Scale container replicas; recycle idle connections |

---

## 3. Clinical & Operational KPI Dashboards
1. **Clinic Intake Throughput:** Real-time counter of active screenings, adult IFHAS reviews, and pediatric well-child encounters.
2. **Offline CHW Sync Health:** Monitors `sync_conflicts` rate and batch sync response times. A spike in conflicts (> 10 / hr) triggers clinical coordinator alert.
3. **Care-Gap Closure Velocity:** Daily count of closed care gaps verified with clinical evidence.
4. **Audit Trail Completeness:** Verifies continuous ingestion into `AuditLog` table with zero dropped security events.
