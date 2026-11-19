# WELLWATCH PREVENT — Container Security & Hardening Standards
**Standard:** CIS Docker Benchmark & ADHICS v2 Infrastructure Security  
**Runtime:** OCI Compliant Container Images (Docker / Containerd / Kubernetes)

---

## 1. Executive Summary
Containerized deployment in healthcare infrastructure requires comprehensive defense-in-depth to eliminate privilege escalation vectors, malicious dependency injection, and resource exhaustion.

WELLWATCH PREVENT implements container hardening across both backend and frontend layers.

---

## 2. Engineered Security Controls

### 2.1 Unprivileged User Execution
- **Backend Container (`Dockerfile.backend`):**
  - Explicit non-root user and group created: `wellwatch:wellwatch` (UID 10001 / GID 10001).
  - Application files owned by `wellwatch`; root privileges explicitly dropped prior to CMD.
- **Frontend Container (`Dockerfile.frontend`):**
  - Explicit non-root user created: `nextjs:nodejs` (UID 1001 / GID 1001).
  - Static runtime executed strictly under unprivileged context.

### 2.2 Multi-Stage Build & Minimal Runtime Footprint
- **Build Isolation:** Compilers, development header packages (`build-essential`), and package management caches are discarded during builder stages.
- **Runtime Images:**
  - Backend utilizes `python:3.11-slim` (runtime dependencies only; wheels copied from `/install`).
  - Frontend utilizes `node:20-alpine` (standalone Next.js output; dev dependencies stripped).

### 2.3 Runtime Protection & Orchestration Flags
In `docker-compose.prod.yml`:
1. `security_opt: ["no-new-privileges:true"]`: Prevents child processes from gaining elevated capabilities (e.g. via setuid binaries).
2. **Resource Constraints:**
   - Backend: Limit `cpus: 2.0`, `memory: 2048M`; Reservation `cpus: 0.5`, `memory: 512M`.
   - Frontend: Limit `cpus: 2.0`, `memory: 2048M`; Reservation `cpus: 0.25`, `memory: 256M`.
   - Prevents compute exhaustion and denial-of-service to adjacent containers.
3. **Log Rotation:** `json-file` driver capped at `max-size: 10m` and `max-file: 5` to prevent host disk exhaustion.
4. **Health Checks:** Built-in unprivileged curl probes test `/api/v1/health` every 30s.

---

## 3. Vulnerability Scanning Protocol
- Automated container scanning integrated via GitHub Actions (Trivy / Docker Scout).
- Images with Critical or High severity CVEs are blocked from deployment.
