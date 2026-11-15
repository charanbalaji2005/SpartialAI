# SpatialAI environment setup for Windows 11 + NVIDIA GPU.
# Usage:  powershell -ExecutionPolicy Bypass -File setup_environment.ps1 [-Cuda cu128] [-SkipModels] [-Training]
param(
    [string]$Cuda = "cu128",      # PyTorch CUDA wheel tag; RTX 40-series works with cu126/cu128
    [switch]$SkipModels,
    [switch]$Training             # also install dataset + training dependencies
)
$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $Root

function Step($n, $text) { Write-Host "`n[$n/10] $text" -ForegroundColor Cyan }
function Fail($text) { Write-Host "ERROR: $text" -ForegroundColor Red; exit 1 }

Step 1 "Checking Python 3.11"
$pyExe = $null; $pyArgs = @()
try { & py -3.11 --version *> $null; if ($LASTEXITCODE -eq 0) { $pyExe = "py"; $pyArgs = @("-3.11") } } catch {}
if (-not $pyExe) {
    try { $v = & python --version 2>&1; if ("$v" -match "3\.1[01]") { $pyExe = "python" } } catch {}
}
if (-not $pyExe) { Fail "Python 3.11 (64-bit) not found. Install it from python.org and tick 'Add to PATH'." }
& $pyExe @pyArgs --version

Step 2 "Checking Node.js"
try { $nodeV = & node --version } catch { Fail "Node.js not found. Install Node.js 20 LTS or newer from nodejs.org." }
Write-Host "Node $nodeV"

Step 3 "Checking NVIDIA GPU"
if (Get-Command nvidia-smi -ErrorAction SilentlyContinue) {
    nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader
} else {
    Write-Host "nvidia-smi not found. SpatialAI will run on CPU (slow). Install the NVIDIA driver for GPU use." -ForegroundColor Yellow
}

Step 4 "Creating virtual environment backend\.venv"
if (-not (Test-Path "backend\.venv")) { & $pyExe @pyArgs -m venv backend\.venv }
$vpy = Join-Path $Root "backend\.venv\Scripts\python.exe"
& $vpy -m pip install --upgrade pip wheel | Out-Null

Step 5 "Installing PyTorch with CUDA ($Cuda)"
& $vpy -m pip install torch torchvision --index-url "https://download.pytorch.org/whl/$Cuda"
if ($LASTEXITCODE -ne 0) {
    Write-Host "That CUDA build failed; trying cu126." -ForegroundColor Yellow
    & $vpy -m pip install torch torchvision --index-url "https://download.pytorch.org/whl/cu126"
    if ($LASTEXITCODE -ne 0) { Fail "PyTorch install failed. Pick a command at pytorch.org/get-started/locally and run it in backend\.venv." }
}

Step 6 "Installing backend dependencies"
& $vpy -m pip install -r backend\requirements.txt
if ($LASTEXITCODE -ne 0) { Fail "pip install -r backend\requirements.txt failed." }

if ($Training) {
    Write-Host "Installing training extras (requirements-training.txt)"
    & $vpy -m pip install -r requirements-training.txt
    if ($LASTEXITCODE -ne 0) { Fail "pip install -r requirements-training.txt failed." }
}

Step 7 "Installing frontend dependencies"
Push-Location frontend
npm install
if ($LASTEXITCODE -ne 0) { Pop-Location; Fail "npm install failed." }
Pop-Location

Step 8 "Preparing storage and configuration"
New-Item -ItemType Directory -Force -Path storage, models, datasets\evaluation | Out-Null
if (-not (Test-Path ".env")) { Copy-Item .env.example .env; Write-Host "Created .env from .env.example" }

Step 9 "Verifying GPU and packages"
& $vpy scripts\check_gpu.py
& $vpy scripts\check_dependencies.py

Step 10 "Downloading models"
if ($SkipModels) { Write-Host "Skipped. Run: backend\.venv\Scripts\python scripts\download_models.py" }
else { & $vpy scripts\download_models.py }

Write-Host "`nSetup finished. Start SpatialAI with start_spatialai.bat" -ForegroundColor Green
Write-Host "Optional: Stable Fast 3D for Best mode needs a separate setup (docs\models.md)."
