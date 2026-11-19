#!/usr/bin/env bash
# ==============================================================================
# WELLWATCH PREVENT — Google Cloud Run Automated Deployment Script
# ==============================================================================
set -euo pipefail

PROJECT_ID="${1:-$(gcloud config get-value project 2>/dev/null || true)}"
REGION="${2:-asia-south1}"
ENVIRONMENT="${3:-staging}"
REPO_NAME="wellwatch-repo"

echo "=========================================================="
echo "   WELLWATCH PREVENT — Google Cloud Deployment Pipeline   "
echo "=========================================================="

if [ -z "$PROJECT_ID" ]; then
    echo "ERROR: No GCP Project ID specified. Usage: ./deploy-gcp.sh <PROJECT_ID> [REGION] [ENVIRONMENT]"
    exit 1
fi

echo "  -> Target Project ID: $PROJECT_ID"
echo "  -> Target Region:     $REGION"
echo "  -> Target Env:        $ENVIRONMENT"

# 1. Enable Required Services
echo "[1/6] Enabling Google Cloud APIs..."
gcloud services enable \
    run.googleapis.com \
    cloudbuild.googleapis.com \
    artifactregistry.googleapis.com \
    storage-component.googleapis.com \
    --project="$PROJECT_ID"

# 2. Setup Artifact Registry Repository
echo "[2/6] Ensuring Artifact Registry repository..."
REPO_CHECK=$(gcloud artifacts repositories describe "$REPO_NAME" --location="$REGION" --project="$PROJECT_ID" 2>&1 || true)
if echo "$REPO_CHECK" | grep -q -E "requires billing to be enabled|BILLING_DISABLED"; then
    echo ""
    echo "ERROR: Google Cloud Billing is required to use Cloud Build & Artifact Registry."
    echo "Please link an active billing account at: https://console.developers.google.com/billing/enable?project=$PROJECT_ID"
    exit 1
fi

if ! echo "$REPO_CHECK" | grep -q "name:"; then
    gcloud artifacts repositories create "$REPO_NAME" \
        --repository-format=docker \
        --location="$REGION" \
        --description="WELLWATCH PREVENT Docker Images" \
        --project="$PROJECT_ID"
fi

BACKEND_IMAGE="$REGION-docker.pkg.dev/$PROJECT_ID/$REPO_NAME/wellwatch-backend:latest"
FRONTEND_IMAGE="$REGION-docker.pkg.dev/$PROJECT_ID/$REPO_NAME/wellwatch-frontend:latest"
SECRET_KEY="$(openssl rand -hex 24)"

# 3. Build & Deploy Backend
echo "[3/6] Building and deploying Backend (FastAPI)..."
gcloud builds submit \
    --tag "$BACKEND_IMAGE" \
    -f infrastructure/docker/Dockerfile.backend \
    . \
    --project="$PROJECT_ID"

gcloud run deploy wellwatch-backend \
    --image "$BACKEND_IMAGE" \
    --region "$REGION" \
    --platform managed \
    --allow-unauthenticated \
    --port 8000 \
    --memory 2Gi \
    --cpu 2 \
    --min-instances 0 \
    --max-instances 5 \
    --set-env-vars "ENVIRONMENT=$ENVIRONMENT,SECRET_KEY=$SECRET_KEY,PORT=8000" \
    --project="$PROJECT_ID"

BACKEND_URL="$(gcloud run services describe wellwatch-backend --region "$REGION" --project "$PROJECT_ID" --format="value(status.url)")"
echo "  -> Backend URL: $BACKEND_URL"

# 4. Build & Deploy Frontend
echo "[4/6] Building and deploying Frontend (Next.js 14)..."
cat <<EOF > /tmp/frontend-cloudbuild.yaml
steps:
  - name: 'gcr.io/cloud-builders/docker'
    args:
      - 'build'
      - '--build-arg'
      - 'NEXT_PUBLIC_API_URL=$BACKEND_URL/api/v1'
      - '-t'
      - '$FRONTEND_IMAGE'
      - '-f'
      - 'infrastructure/docker/Dockerfile.frontend'
      - '.'
  - name: 'gcr.io/cloud-builders/docker'
    args: ['push', '$FRONTEND_IMAGE']
images:
  - '$FRONTEND_IMAGE'
EOF

gcloud builds submit --config /tmp/frontend-cloudbuild.yaml . --project="$PROJECT_ID"
rm -f /tmp/frontend-cloudbuild.yaml

gcloud run deploy wellwatch-frontend \
    --image "$FRONTEND_IMAGE" \
    --region "$REGION" \
    --platform managed \
    --allow-unauthenticated \
    --port 3000 \
    --memory 2Gi \
    --cpu 2 \
    --min-instances 0 \
    --max-instances 5 \
    --set-env-vars "NODE_ENV=production,NEXT_PUBLIC_API_URL=$BACKEND_URL/api/v1,PORT=3000" \
    --project="$PROJECT_ID"

FRONTEND_URL="$(gcloud run services describe wellwatch-frontend --region "$REGION" --project "$PROJECT_ID" --format="value(status.url)")"
echo "  -> Frontend URL: $FRONTEND_URL"

# 5. Update Backend CORS
echo "[5/6] Updating Backend CORS configuration..."
gcloud run services update wellwatch-backend \
    --region "$REGION" \
    --update-env-vars "BACKEND_CORS_ORIGINS=['$FRONTEND_URL','http://localhost:3000']" \
    --project="$PROJECT_ID"

# 6. Verification
echo "[6/6] Verifying health..."
curl -s "$BACKEND_URL/api/v1/health" || true

echo ""
echo "=========================================================="
echo "   DEPLOYMENT TO GOOGLE CLOUD SUCCESSFUL!                 "
echo "=========================================================="
echo "  * Platform: Google Cloud Run (Managed)"
echo "  * Backend:  $BACKEND_URL"
echo "  * API Docs: $BACKEND_URL/api/v1/docs"
echo "  * Frontend: $FRONTEND_URL"
echo "=========================================================="
