<#
.SYNOPSIS
    Automated Google Cloud Run Deployment Script for WELLWATCH PREVENT.
.DESCRIPTION
    Builds and deploys the FastAPI backend and Next.js frontend to Google Cloud Run,
    configures environment variables, establishes CORS, and verifies health status.
.PARAMETER ProjectId
    The Google Cloud Project ID.
.PARAMETER Region
    The GCP region (default: 'asia-south1').
.PARAMETER Environment
    Deployment environment ('staging' or 'production', default: 'staging').
.PARAMETER DatabaseUrl
    Custom database connection URL (optional).
#>

[CmdletBinding()]
param(
    [Parameter(Mandatory = $false)]
    [string]$ProjectId = "",

    [Parameter(Mandatory = $false)]
    [string]$Region = "asia-south1",

    [Parameter(Mandatory = $false)]
    [ValidateSet("staging", "production", "development")]
    [string]$Environment = "staging",

    [Parameter(Mandatory = $false)]
    [string]$DatabaseUrl = "",

    [Parameter(Mandatory = $false)]
    [string]$SecretKey = ""
)

$ErrorActionPreference = "Continue"

Write-Host "==========================================================" -ForegroundColor Cyan
Write-Host "   WELLWATCH PREVENT — Google Cloud Deployment Pipeline   " -ForegroundColor Cyan
Write-Host "==========================================================" -ForegroundColor Cyan

# 1. Verify gcloud CLI
Write-Host "`n[1/7] Checking Google Cloud CLI installation..." -ForegroundColor Yellow
if (-not (Get-Command gcloud -ErrorAction SilentlyContinue)) {
    Write-Error "Google Cloud SDK (gcloud) is not found in PATH. Please install Google Cloud SDK."
    exit 1
}

# 2. Resolve Active Account & Project
$activeAccount = (gcloud config get-value account 2>$null).Trim()
if ([string]::IsNullOrWhiteSpace($activeAccount) -or $activeAccount -like "*unset*") {
    Write-Error "No active Google Cloud account detected. Please authenticate using 'gcloud auth login'."
    exit 1
}
Write-Host "  -> Authenticated as: $activeAccount" -ForegroundColor Green

if ([string]::IsNullOrWhiteSpace($ProjectId)) {
    $ProjectId = (gcloud config get-value project 2>$null).Trim()
}

if ([string]::IsNullOrWhiteSpace($ProjectId) -or $ProjectId -like "*unset*") {
    Write-Error "No GCP Project ID specified or active. Pass -ProjectId <YOUR_PROJECT_ID> or run 'gcloud config set project <ID>'."
    exit 1
}
Write-Host "  -> Target Project ID: $ProjectId" -ForegroundColor Green
Write-Host "  -> Target Region: $Region" -ForegroundColor Green
Write-Host "  -> Target Environment: $Environment" -ForegroundColor Green

# 3. Check / Generate Secret Key
if ([string]::IsNullOrWhiteSpace($SecretKey)) {
    $SecretKey = -join ((65..90) + (97..122) + (48..57) | Get-Random -Count 48 | ForEach-Object { [char]$_ })
}

# 4. Enable Google Cloud APIs
Write-Host "`n[2/7] Enabling required Google Cloud APIs..." -ForegroundColor Yellow
$requiredApis = @(
    "run.googleapis.com",
    "cloudbuild.googleapis.com",
    "artifactregistry.googleapis.com",
    "storage-component.googleapis.com"
)
foreach ($api in $requiredApis) {
    Write-Host "  -> Enabling $api on $ProjectId..."
    & gcloud services enable $api --project=$ProjectId
    if ($LASTEXITCODE -ne 0) {
        Write-Warning "Failed enabling $api. If billing is disabled, please enable billing at: https://console.developers.google.com/billing/enable?project=$ProjectId"
        exit $LASTEXITCODE
    }
}

# 5. Create Artifact Registry Repository if not present
Write-Host "`n[3/7] Setting up Artifact Registry repository..." -ForegroundColor Yellow
$repoName = "wellwatch-repo"
$repoOutput = & gcloud artifacts repositories describe $repoName --location=$Region --project=$ProjectId 2>&1
$repoText = "$repoOutput"

if ($repoText -like "*requires billing to be enabled*" -or $repoText -like "*BILLING_DISABLED*") {
    Write-Host "`n[!] GOOGLE CLOUD BILLING REQUIRED" -ForegroundColor Red
    Write-Host "Project '$ProjectId' does not currently have active billing linked." -ForegroundColor Red
    Write-Host "To deploy Cloud Run & Cloud Build, please enable billing at:" -ForegroundColor Yellow
    Write-Host "  -> https://console.developers.google.com/billing/enable?project=$ProjectId" -ForegroundColor Cyan
    exit 1
}

if ($LASTEXITCODE -ne 0 -or $repoText -like "*NOT_FOUND*") {
    Write-Host "  -> Creating repository '$repoName' in $Region..."
    & gcloud artifacts repositories create $repoName --repository-format=docker --location=$Region --description="WELLWATCH PREVENT Docker Images" --project=$ProjectId
} else {
    Write-Host "  -> Repository '$repoName' is ready in $Region." -ForegroundColor Green
}

$backendImage = "$Region-docker.pkg.dev/$ProjectId/$repoName/wellwatch-backend:latest"
$frontendImage = "$Region-docker.pkg.dev/$ProjectId/$repoName/wellwatch-frontend:latest"

# 6. Build and Deploy Backend Service
Write-Host "`n[4/7] Building and deploying Backend (FastAPI)..." -ForegroundColor Yellow

$backendEnvVars = "ENVIRONMENT=$Environment,SECRET_KEY=$SecretKey,PORT=8000"
if (-not [string]::IsNullOrWhiteSpace($DatabaseUrl)) {
    $backendEnvVars += ",DATABASE_URL=$DatabaseUrl"
}

Write-Host "  -> Submitting Cloud Build for Backend container..."
& gcloud builds submit --tag $backendImage -f infrastructure/docker/Dockerfile.backend . --project=$ProjectId
if ($LASTEXITCODE -ne 0) {
    Write-Error "Backend Cloud Build failed."
    exit $LASTEXITCODE
}

Write-Host "  -> Deploying 'wellwatch-backend' to Cloud Run..."
& gcloud run deploy wellwatch-backend `
    --image $backendImage `
    --region $Region `
    --platform managed `
    --allow-unauthenticated `
    --port 8000 `
    --memory 2Gi `
    --cpu 2 `
    --min-instances 0 `
    --max-instances 5 `
    --set-env-vars $backendEnvVars `
    --project=$ProjectId

if ($LASTEXITCODE -ne 0) {
    Write-Error "Backend deployment to Cloud Run failed."
    exit $LASTEXITCODE
}

$backendUrl = (& gcloud run services describe wellwatch-backend --region $Region --project $ProjectId --format="value(status.url)").Trim()
Write-Host "  -> Backend URL: $backendUrl" -ForegroundColor Green

# 7. Build and Deploy Frontend Service
Write-Host "`n[5/7] Building and deploying Frontend (Next.js 14)..." -ForegroundColor Yellow

Write-Host "  -> Submitting Cloud Build for Frontend container with backend URL: $backendUrl/api/v1..."
# Create a temporary Cloud Build config to pass build-arg for Next.js
$frontendBuildYaml = @"
steps:
  - name: 'gcr.io/cloud-builders/docker'
    args:
      - 'build'
      - '--build-arg'
      - 'NEXT_PUBLIC_API_URL=$backendUrl/api/v1'
      - '-t'
      - '$frontendImage'
      - '-f'
      - 'infrastructure/docker/Dockerfile.frontend'
      - '.'
  - name: 'gcr.io/cloud-builders/docker'
    args: ['push', '$frontendImage']
images:
  - '$frontendImage'
"@

$tempBuildFile = [System.IO.Path]::GetTempFileName() + ".yaml"
$frontendBuildYaml | Out-File -FilePath $tempBuildFile -Encoding utf8

& gcloud builds submit --config $tempBuildFile . --project=$ProjectId
Remove-Item $tempBuildFile -Force -ErrorAction SilentlyContinue

if ($LASTEXITCODE -ne 0) {
    Write-Error "Frontend Cloud Build failed."
    exit $LASTEXITCODE
}

Write-Host "  -> Deploying 'wellwatch-frontend' to Cloud Run..."
& gcloud run deploy wellwatch-frontend `
    --image $frontendImage `
    --region $Region `
    --platform managed `
    --allow-unauthenticated `
    --port 3000 `
    --memory 2Gi `
    --cpu 2 `
    --min-instances 0 `
    --max-instances 5 `
    --set-env-vars "NODE_ENV=production,NEXT_PUBLIC_API_URL=$backendUrl/api/v1,PORT=3000" `
    --project=$ProjectId

if ($LASTEXITCODE -ne 0) {
    Write-Error "Frontend deployment to Cloud Run failed."
    exit $LASTEXITCODE
}

$frontendUrl = (& gcloud run services describe wellwatch-frontend --region $Region --project $ProjectId --format="value(status.url)").Trim()
Write-Host "  -> Frontend URL: $frontendUrl" -ForegroundColor Green

# 8. Update Backend CORS with Frontend URL
Write-Host "`n[6/7] Updating Backend CORS configuration..." -ForegroundColor Yellow
$corsOrigins = "['$frontendUrl','http://localhost:3000']"
& gcloud run services update wellwatch-backend `
    --region $Region `
    --update-env-vars "BACKEND_CORS_ORIGINS=$corsOrigins" `
    --project=$ProjectId

# 9. Verify Live Health Probes
Write-Host "`n[7/7] Verifying live deployments..." -ForegroundColor Yellow
try {
    $healthCheck = Invoke-RestMethod -Uri "$backendUrl/api/v1/health" -Method Get -TimeoutSec 15
    Write-Host "  -> Backend Health Check: $($healthCheck.status)" -ForegroundColor Green
} catch {
    Write-Warning "Backend health probe returned: $_"
}

Write-Host "`n==========================================================" -ForegroundColor Green
Write-Host "   DEPLOYMENT TO GOOGLE CLOUD SUCCESSFUL!                 " -ForegroundColor Green
Write-Host "==========================================================" -ForegroundColor Green
Write-Host "  * Platform: Google Cloud Run (Managed)" -ForegroundColor Cyan
Write-Host "  * Region:   $Region" -ForegroundColor Cyan
Write-Host "  * Backend:  $backendUrl" -ForegroundColor Cyan
Write-Host "  * API Docs: $backendUrl/api/v1/docs" -ForegroundColor Cyan
Write-Host "  * Frontend: $frontendUrl" -ForegroundColor Cyan
Write-Host "==========================================================" -ForegroundColor Green
