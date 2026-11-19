# Google Cloud Deployment Guide — WELLWATCH PREVENT
**Platform:** Google Cloud Platform (Cloud Run, Cloud Build, Artifact Registry, Cloud SQL)  
**Region Default:** `asia-south1` (or UAE sovereign cloud equivalent `me-central1` / `me-west1`)

---

## 1. Overview & Architecture

WELLWATCH PREVENT is architected as a cloud-native, decoupled service topology on Google Cloud:

```
                                  +----------------------------+
                                  | Google Cloud Load Balancer |
                                  |     (Custom Domain / SSL)  |
                                  +--------------+-------------+
                                                 |
                         +-----------------------+-----------------------+
                         |                                               |
                         v                                               v
        +----------------------------------+            +----------------------------------+
        |   Frontend (Next.js 14 App)      |            |    Backend (FastAPI Engine)      |
        |   Service: wellwatch-frontend    |  REST API  |    Service: wellwatch-backend    |
        |   - Node.js 20 Alpine            +----------->+    - Python 3.11 Slim            |
        |   - Static + SSR (Port 3000)     |            |    - 136 Clinical Rules/Tests    |
        +----------------------------------+            +----------------+-----------------+
                                                                         |
                                                        +----------------+-----------------+
                                                        |                                  |
                                                        v (Staging Mode)                   v (Production Mode)
                                            +-----------------------+          +-----------------------+
                                            | Ephemeral SQLite DB   |          | Google Cloud SQL      |
                                            | Pre-seeded AD Cohorts |          | PostgreSQL 16 (Pool)  |
                                            +-----------------------+          +-----------------------+
```

---

## 2. Prerequisites

1. **Google Cloud SDK (`gcloud` CLI)** installed and authenticated:
   ```powershell
   gcloud auth login
   ```
2. **Google Cloud Project** with **Active Billing Enabled**:
   > [!IMPORTANT]
   > Google Cloud requires an active billing account linked to your project to run Cloud Build, Artifact Registry, and Cloud Run.
   > You can verify or enable billing at: [Google Cloud Console Billing](https://console.developers.google.com/billing)

---

## 3. One-Command Automated Deployment

We have provided ready-to-run deployment pipelines for both Windows PowerShell and Linux/macOS:

### Option A: Windows PowerShell (`deploy-gcp.ps1`)

```powershell
# Deploy to your GCP project (e.g., brandpr-512618 or your dedicated project)
.\deploy-gcp.ps1 -ProjectId <YOUR_GCP_PROJECT_ID> -Region asia-south1 -Environment staging
```

### Option B: Linux / macOS / Cloud Shell (`deploy-gcp.sh`)

```bash
chmod +x deploy-gcp.sh
./deploy-gcp.sh <YOUR_GCP_PROJECT_ID> asia-south1 staging
```

The script automatically:
1. Enables required Google Cloud APIs (`run.googleapis.com`, `cloudbuild.googleapis.com`, `artifactregistry.googleapis.com`).
2. Configures the Docker repository `wellwatch-repo` in Artifact Registry.
3. Builds and deploys `wellwatch-backend` to Cloud Run.
4. Retrieves the secure backend HTTPS URL.
5. Injects the backend URL into the Next.js frontend build.
6. Builds and deploys `wellwatch-frontend` to Cloud Run.
7. Automatically configures backend CORS to trust the frontend domain.
8. Performs live health verification on `/api/v1/health`.

---

## 4. Manual Step-by-Step Deployment (via `gcloud`)

If you prefer executing the steps manually:

### Step 1: Set Project & Enable Services
```bash
gcloud config set project <PROJECT_ID>
gcloud services enable run.googleapis.com cloudbuild.googleapis.com artifactregistry.googleapis.com
```

### Step 2: Create Artifact Registry Repository
```bash
gcloud artifacts repositories create wellwatch-repo \
    --repository-format=docker \
    --location=asia-south1 \
    --description="WELLWATCH PREVENT Images"
```

### Step 3: Build & Deploy Backend
```bash
# Build image with Cloud Build
gcloud builds submit --tag asia-south1-docker.pkg.dev/<PROJECT_ID>/wellwatch-repo/wellwatch-backend:latest \
    -f infrastructure/docker/Dockerfile.backend .

# Deploy to Cloud Run
gcloud run deploy wellwatch-backend \
    --image asia-south1-docker.pkg.dev/<PROJECT_ID>/wellwatch-repo/wellwatch-backend:latest \
    --region asia-south1 \
    --platform managed \
    --allow-unauthenticated \
    --port 8000 \
    --memory 2Gi \
    --cpu 2 \
    --set-env-vars "ENVIRONMENT=staging,SECRET_KEY=$(openssl rand -hex 24)"
```

### Step 4: Build & Deploy Frontend
```bash
BACKEND_URL=$(gcloud run services describe wellwatch-backend --region asia-south1 --format="value(status.url)")

# Deploy directly via source or Cloud Build
gcloud run deploy wellwatch-frontend \
    --source . \
    --region asia-south1 \
    --platform managed \
    --allow-unauthenticated \
    --port 3000 \
    --memory 2Gi \
    --cpu 2 \
    --set-env-vars "NODE_ENV=production,NEXT_PUBLIC_API_URL=${BACKEND_URL}/api/v1"
```

---

## 5. Production Database Integration (Google Cloud SQL)

For production deployments using managed PostgreSQL 16:

1. **Create Cloud SQL Instance:**
   ```bash
   gcloud sql instances create wellwatch-postgres \
       --database-version=POSTGRES_16 \
       --tier=db-custom-2-7680 \
       --region=asia-south1
   ```

2. **Create Database & User:**
   ```bash
   gcloud sql databases create wellwatch_prevent --instance=wellwatch-postgres
   gcloud sql users create wellwatch_user --instance=wellwatch-postgres --password=<SECURE_PASSWORD>
   ```

3. **Deploy Backend with Cloud SQL Connection:**
   ```bash
   CONNECTION_NAME=$(gcloud sql instances describe wellwatch-postgres --format="value(connectionName)")
   
   gcloud run deploy wellwatch-backend \
       --add-cloudsql-instances $CONNECTION_NAME \
       --update-env-vars "ENVIRONMENT=production,DATABASE_URL=postgresql+psycopg2://wellwatch_user:<PASSWORD>@/wellwatch_prevent?host=/cloudsql/$CONNECTION_NAME"
   ```

---

## 6. Verification & Health Monitoring

Once deployed, verify the endpoints:
- **Liveness Health Check:** `GET https://<BACKEND_URL>/api/v1/health`
  ```json
  {"status": "healthy", "service": "WELLWATCH PREVENT", "version": "1.0.0"}
  ```
- **Readiness Probe:** `GET https://<BACKEND_URL>/api/v1/readiness`
  ```json
  {"status": "ready", "database": "connected"}
  ```
- **Interactive Swagger Docs:** `https://<BACKEND_URL>/api/v1/docs`
- **Frontend Dashboard:** `https://<FRONTEND_URL>/dashboard`
