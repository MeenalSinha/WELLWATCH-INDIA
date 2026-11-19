# Render Deployment Guide — WELLWATCH PREVENT (100% Free Tier)
**Target Platform:** [Render.com](https://render.com)  
**Monthly Cost:** **$0.00 / month** (Render Free Tier)  
**Architecture:** Decoupled FastAPI Backend + Next.js 14 Frontend + Managed PostgreSQL 16 Database

---

## 1. Architecture Overview on Render

```
                                      +--------------------------+
                                      |   Render Cloud Router    |
                                      |  (Free SSL / HTTPS URLs) |
                                      +-------------+------------+
                                                    |
                           +------------------------+------------------------+
                           |                                                 |
                           v                                                 v
        +------------------------------------+             +------------------------------------+
        |   Frontend (Next.js 14 App Router) |             |    Backend (FastAPI Intelligence)  |
        |   Service: wellwatch-frontend      |  REST API   |    Service: wellwatch-backend      |
        |   - Node.js 20 (Free Web Service)  +------------>+    - Python 3.11 (Free Web Service)|
        |   - Port: $PORT (Free 512 MB RAM)  |             |    - Port: $PORT (1 Worker, 512 MB)|
        +------------------------------------+             +-----------------+------------------+
                                                                             |
                                                                             v
                                                           +------------------------------------+
                                                           |   wellwatch-db (Render PostgreSQL) |
                                                           |   - PostgreSQL 16 (Free Tier 1 GB) |
                                                           +------------------------------------+
```

---

## 2. Quick Deploy: Render Blueprint (Recommended — 1 Click)

The repository includes a pre-configured [render.yaml](file:///c:/Users/Meenal%20Sinha/Downloads/WELLWATCH-INDIA/render.yaml) Infrastructure-as-Code Blueprint.

### Step 1: Push latest changes to GitHub
Ensure all new deployment configurations are committed and pushed to your GitHub repository:
```bash
git add .
git commit -m "feat: add Render free tier blueprint and cloud configuration"
git push origin main
```

### Step 2: Create Blueprint Instance on Render
1. Navigate to the **[Render Dashboard](https://dashboard.render.com/)**.
2. Click the **"New +"** button in the top right navigation bar.
3. Select **"Blueprint"**.
4. Connect your GitHub account and select repository: **`MeenalSinha/WELLWATCH-INDIA`**.
5. Render will automatically detect `render.yaml` and display the deployment plan:
   - `wellwatch-backend` (Web Service, Python, Free)
   - `wellwatch-frontend` (Web Service, Node, Free)
   - `wellwatch-db` (PostgreSQL Database, Free)
6. Click **"Apply"**.

Render will now provision the database, build the backend, compile the Next.js frontend, and wire up all connection strings and environment variables automatically!

---

## 3. Manual Deployment Guide (Service-by-Service)

If you prefer deploying the services individually via the Render web UI:

### A. Create Free Managed PostgreSQL Database
1. Go to **New +** -> **PostgreSQL**.
2. **Name:** `wellwatch-db`
3. **Database:** `wellwatch_prevent`
4. **User:** `wellwatch_user`
5. **Region:** Oregon (US West) or Frankfurt (EU Central)
6. **Plan:** **Free**
7. Click **Create Database**.
8. Once provisioned, copy the **Internal Database URL** (e.g. `postgres://wellwatch_user:...@dpg-...:5432/wellwatch_prevent`).

---

### B. Create Backend Web Service (FastAPI)
1. Go to **New +** -> **Web Service**.
2. Select your repository: `MeenalSinha/WELLWATCH-INDIA`.
3. Configure the service settings:
   - **Name:** `wellwatch-backend`
   - **Region:** Same region as your database (e.g. `Oregon`)
   - **Branch:** `main`
   - **Root Directory:** *(leave blank — repository root)*
   - **Runtime:** `Python 3`
   - **Build Command:**
     ```bash
     pip install --upgrade pip && pip install -r backend/requirements.txt
     ```
   - **Start Command:**
     ```bash
     uvicorn backend.app.main:app --host 0.0.0.0 --port $PORT --workers 1
     ```
   - **Instance Type:** **Free** (512 MB RAM, 0.1 CPU)
4. Add **Environment Variables**:
   | Variable | Value | Notes |
   | :--- | :--- | :--- |
   | `PYTHON_VERSION` | `3.11.9` | Ensures Python 3.11 runtime |
   | `ENVIRONMENT` | `staging` | Enables auto-schema creation & demo cohorts |
   | `SECRET_KEY` | *(Click "Generate" or enter 32+ characters)* | JWT encryption key |
   | `DATABASE_URL` | *(Paste Internal Database URL from Step A)* | Connects to PostgreSQL |
   | `BACKEND_CORS_ORIGINS` | `["*"]` | Allows cross-origin requests from frontend |
5. Click **Create Web Service**.
6. Wait for the build to finish. Once live, note your backend URL:
   `https://wellwatch-backend-xxxx.onrender.com`

---

### C. Create Frontend Web Service (Next.js 14)
1. Go to **New +** -> **Web Service**.
2. Select your repository: `MeenalSinha/WELLWATCH-INDIA`.
3. Configure the service settings:
   - **Name:** `wellwatch-frontend`
   - **Region:** Same region as backend (e.g. `Oregon`)
   - **Branch:** `main`
   - **Root Directory:** `frontend`
   - **Runtime:** `Node`
   - **Build Command:**
     ```bash
     npm install && npm run build
     ```
   - **Start Command:**
     ```bash
     npm start
     ```
   - **Instance Type:** **Free** (512 MB RAM, 0.1 CPU)
4. Add **Environment Variables**:
   | Variable | Value | Notes |
   | :--- | :--- | :--- |
   | `NODE_ENV` | `production` | Production Next.js mode |
   | `NEXT_PUBLIC_API_URL` | `https://<YOUR_BACKEND_URL>/api/v1` | Points browser to backend |
   | `INTERNAL_BACKEND_URL`| `https://<YOUR_BACKEND_URL>` | Used by Next.js server rewrites |
5. Click **Create Web Service**.

---

## 4. Free Tier Operational Details

1. **Spin-Down on Inactivity (Cold Starts):**
   - Render Free Web Services automatically sleep after 15 minutes of inactivity.
   - When a user visits your app after it has spun down, the first request will take ~30–50 seconds to wake up the service.
   - *Tip:* You can use a free ping monitor (such as [cron-job.org](https://cron-job.org) or [UptimeRobot](https://uptimerobot.com)) to ping `https://<your-backend>.onrender.com/api/v1/health` every 14 minutes to keep both services permanently warm.

2. **Memory Footprint Optimization:**
   - The backend is configured with `--workers 1` to use ~120 MB RAM, well below the 512 MB free tier threshold.
   - Next.js 14 runs with optimized memory caching.

3. **Auto-Seeding & Database Initialization:**
   - In `ENVIRONMENT=staging`, on initial boot the backend automatically runs `Base.metadata.create_all()` and seeds default Abu Dhabi clinical cohorts, test users, and organizations.

---

## 5. Live Verification & Demo Credentials

Once both services show **"Live"** on Render:

1. **Verify Backend Liveness:**
   ```bash
   curl https://<your-backend>.onrender.com/api/v1/health
   # Expected: {"status": "healthy", "service": "WELLWATCH PREVENT", "version": "1.0.0"}
   ```

2. **Open Interactive Swagger Documentation:**
   `https://<your-backend>.onrender.com/api/v1/docs`

3. **Access Frontend Application:**
   `https://<your-frontend>.onrender.com/dashboard`

4. **Default Test Accounts:**
   - **Healthcare Provider:** `dr.layla@seha.ae` / `ProviderPass123!`
   - **Data Governance Officer:** `data_governance_officer@wellwatch.ae` / `GovernancePass123!`
   - **Community Health Worker:** `chw.fatima@adphc.gov.ae` / `ChwPass123!`
   - **Individual Patient:** `mariam@example.com` / `PatientPass123!`
