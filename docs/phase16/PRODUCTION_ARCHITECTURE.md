# WELLWATCH PREVENT — Production Deployment Architecture
**Target Region:** Abu Dhabi, United Arab Emirates (UAE Sovereign Cloud)  
**Standard Compliance:** ADHICS v2, UAE Health Data Law, ISO 27001  
**Version:** 1.0.0 (Phase 16 Production Baseline)

---

## 1. High-Level System Architecture

```
                                 ┌─────────────────────────────────┐
                                 │      Internet / MoHAP / DoH     │
                                 │        Encrypted Clients        │
                                 └────────────────┬────────────────┘
                                                  │ HTTPS / TLS 1.3
                                                  ▼
                                 ┌─────────────────────────────────┐
                                 │     WAF & Ingress Controller    │
                                 │ (Azure App Gateway / Cloudflare)│
                                 └────────┬───────────────┬────────┘
                                          │               │
                 ┌────────────────────────┘               └────────────────────────┐
                 ▼ (Port 3000)                                                     ▼ (Port 8000)
    ┌───────────────────────────┐                                     ┌───────────────────────────┐
    │  Frontend Next.js Cluster │                                     │  Backend FastAPI Cluster  │
    │  (Unprivileged user 1001) │                                     │  (Unprivileged user 10001)│
    │   Auto-scaling Replicas   │                                     │   Auto-scaling Replicas   │
    └───────────────────────────┘                                     └─────────────┬─────────────┘
                                                                                    │
                                                                                    │ Internal Network
                                                                                    ▼ (Port 5432)
                                                                      ┌───────────────────────────┐
                                                                      │ PostgreSQL 16 HA Primary  │
                                                                      │ (AES-256 Storage, Pool 20)│
                                                                      └─────────────┬─────────────┘
                                                                                    │ Streaming Replication
                                                                                    ▼
                                                                      ┌───────────────────────────┐
                                                                      │ PostgreSQL Read Replica   │
                                                                      │   (Disaster Recovery)     │
                                                                      └───────────────────────────┘
```

---

## 2. Infrastructure Components & Specifications

### 2.1 Compute Tier
- **Container Runtime:** Hardened OCI-compliant containers orchestrated via Kubernetes (AKS / OCI Engine in UAE North) or Docker Swarm/Compose for dedicated private nodes.
- **Backend Service:** FastAPI with ASGI Uvicorn workers (4 workers per container, memory limit 2048 MB, CPU limit 2.0 cores).
- **Frontend Service:** Next.js 14 standalone SSR/Static runtime running unprivileged under user `nextjs` (UID 1001), memory limit 2048 MB, CPU limit 2.0 cores.

### 2.2 Data Persistence Tier
- **Primary Database:** Managed PostgreSQL 16 with High Availability (HA) Multi-AZ deployment.
- **Connection Pooling:** SQLAlchemy engine configured with `pool_size=20`, `max_overflow=10`, `pool_pre_ping=True`, and `pool_recycle=1800`.
- **Storage Encryption:** Volume-level AES-256 encryption at rest; TLS 1.3 for all database connections.

### 2.3 Network Isolation & Security Zones
- **Public / Ingress Zone:** Terminated at WAF; only ports 443 (HTTPS) exposed to external users.
- **Application Zone:** Microservices communicate via internal virtual network (`internal-net`).
- **Data Tier Zone:** Database cluster resides in private subnets with strict security group filtering; direct public internet ingress is physically blocked.
