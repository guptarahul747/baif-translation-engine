# BAIF Offline Translation Engine - Windows One-Click Setup Script
[CmdletBinding()]
param(
    [string]$ModelRepoId = $env:BAIF_MODEL_REPO_ID,
    [string]$HfToken = $env:HF_TOKEN,
    [switch]$SkipModelVerification
)

$ErrorActionPreference = "Stop"
$env:PYTHONUTF8 = "1"
$env:PYTHONIOENCODING = "utf-8"

Write-Host "=================================================================" -ForegroundColor Cyan
Write-Host "  BAIF Offline Translation Engine - One-Click Windows Setup" -ForegroundColor Cyan
Write-Host "=================================================================" -ForegroundColor Cyan
Write-Host ""

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $ScriptDir

# ------------------------------------------------------------------------------
# 1. Detect & Configure FFmpeg
# ------------------------------------------------------------------------------
Write-Host "[1/6] Checking FFmpeg and FFprobe..." -ForegroundColor Yellow

function Test-FFmpeg {
    $ff = Get-Command ffmpeg -ErrorAction SilentlyContinue
    $pr = Get-Command ffprobe -ErrorAction SilentlyContinue
    return ($null -ne $ff -and $null -ne $pr)
}

if (-not (Test-FFmpeg)) {
    $CommonFFmpegPaths = @(
        "C:\ffmpeg\bin",
        "C:\Program Files\ffmpeg\bin",
        "C:\tools\ffmpeg\bin",
        "$env:LOCALAPPDATA\Programs\ffmpeg\bin"
    )

    $FoundFFmpeg = $false
    foreach ($p in $CommonFFmpegPaths) {
        if ((Test-Path (Join-Path $p "ffmpeg.exe")) -and (Test-Path (Join-Path $p "ffprobe.exe"))) {
            Write-Host "  Found FFmpeg at: $p" -ForegroundColor Green
            $userPath = [Environment]::GetEnvironmentVariable("Path", "User")
            if ($userPath -notlike "*$p*") {
                [Environment]::SetEnvironmentVariable("Path", "$userPath;$p", "User")
                Write-Host "  Added $p to User PATH." -ForegroundColor Green
            }
            $env:Path = "$env:Path;$p"
            $FoundFFmpeg = $true
            break
        }
    }

    if (-not $FoundFFmpeg) {
        Write-Host "  [WARNING] FFmpeg / FFprobe not found in PATH or standard folders." -ForegroundColor Yellow
        Write-Host "  Please install FFmpeg x64 from https://www.gyan.dev/ffmpeg/builds/ (e.g. extract to C:\ffmpeg\bin) or run: winget install Gyan.FFmpeg" -ForegroundColor Gray
        Write-Host "  Setup will continue, but audio/video processing requires FFmpeg." -ForegroundColor Gray
    } else {
        Write-Host "  [OK] FFmpeg & FFprobe configured successfully." -ForegroundColor Green
    }
} else {
    Write-Host "  [OK] FFmpeg & FFprobe are available on PATH." -ForegroundColor Green
}

# ------------------------------------------------------------------------------
# 2. Detect Python 3.12
# ------------------------------------------------------------------------------
Write-Host ""
Write-Host "[2/6] Detecting Python 3.12 x64..." -ForegroundColor Yellow

$PythonCandidates = @(
    "py.exe -3.12",
    "python.exe",
    "C:\Program Files\Python312\python.exe",
    "$env:LOCALAPPDATA\Programs\Python\Python312\python.exe",
    "C:\Python312\python.exe"
)

$SelectedPython = $null

foreach ($cand in $PythonCandidates) {
    try {
        if ($cand -eq "py.exe -3.12") {
            $ver = & py -3.12 -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')" 2>$null
            if ($ver -eq "3.12") {
                $SelectedPython = "py -3.12"
                break
            }
        } elseif (Test-Path $cand) {
            $ver = & $cand -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')" 2>$null
            if ($ver -eq "3.12") {
                $SelectedPython = $cand
                break
            }
        } else {
            $cmd = Get-Command $cand -ErrorAction SilentlyContinue
            if ($null -ne $cmd) {
                $ver = & $cand -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')" 2>$null
                if ($ver -eq "3.12") {
                    $SelectedPython = $cmd.Source
                    break
                }
            }
        }
    } catch {}
}

if ($null -eq $SelectedPython) {
    Write-Host "  [ERROR] Python 3.12 was not found on your system." -ForegroundColor Red
    Write-Host "  Please install Python 3.12 x64 from https://www.python.org/downloads/ (check 'Add python.exe to PATH')." -ForegroundColor Yellow
    exit 1
}

Write-Host "  [OK] Using Python interpreter: $SelectedPython" -ForegroundColor Green

# ------------------------------------------------------------------------------
# 3. Create / Verify Python Virtual Environment (.venv)
# ------------------------------------------------------------------------------
Write-Host ""
Write-Host "[3/6] Setting up virtual environment (.venv)..." -ForegroundColor Yellow

$VenvPython = Join-Path $ScriptDir ".venv\Scripts\python.exe"

if (-not (Test-Path $VenvPython)) {
    Write-Host "  Creating .venv with $SelectedPython..." -ForegroundColor Cyan
    if ($SelectedPython -eq "py -3.12") {
        & py -3.12 -m venv .venv
    } else {
        & "$SelectedPython" -m venv .venv
    }
    if (-not (Test-Path $VenvPython)) {
        Write-Host "  [ERROR] Failed to create .venv." -ForegroundColor Red
        exit 1
    }
}
Write-Host "  [OK] Virtual environment is ready: $VenvPython" -ForegroundColor Green

# ------------------------------------------------------------------------------
# 4. Install Python Dependencies
# ------------------------------------------------------------------------------
Write-Host ""
Write-Host "[4/6] Installing Python dependencies..." -ForegroundColor Yellow

Write-Host "  Upgrading installer tools..." -ForegroundColor Cyan
& $VenvPython -m pip install --upgrade pip "setuptools<70" "wheel==0.43.0" "packaging<24,>=16.8" --quiet

Write-Host "  Installing runtime requirements..." -ForegroundColor Cyan
& $VenvPython -m pip install -r requirements.txt --quiet

Write-Host "  Installing setup utilities..." -ForegroundColor Cyan
& $VenvPython -m pip install -r requirements-setup.txt --quiet

Write-Host "  Checking dependency compatibility..." -ForegroundColor Cyan
$checkResult = & $VenvPython -m pip check
if ($LASTEXITCODE -ne 0) {
    Write-Host "  [WARNING] Dependency warning: $checkResult" -ForegroundColor Yellow
} else {
    Write-Host "  [OK] All dependencies installed and validated (pip check passed)." -ForegroundColor Green
}

# ------------------------------------------------------------------------------
# 5. Download and Provision Models
# ------------------------------------------------------------------------------
Write-Host ""
Write-Host "[5/6] Checking / Provisioning local model vault..." -ForegroundColor Yellow

$VaultDir = Join-Path $ScriptDir "local_model_vault"
$RequiredModelChecks = @(
    (Join-Path $VaultDir "whisper\model.bin"),
    (Join-Path $VaultDir "indictrans2\en-indic\model.bin"),
    (Join-Path $VaultDir "indictrans2\indic-en\model.bin"),
    (Join-Path $VaultDir "tts\english\en_US-lessac-medium.onnx"),
    (Join-Path $VaultDir "tts\hindi\hi_IN-pratham-medium.onnx"),
    (Join-Path $VaultDir "tts\marathi\mr_IN-google-medium.onnx")
)

$AllModelsPresent = $true
foreach ($m in $RequiredModelChecks) {
    if (-not (Test-Path $m)) {
        $AllModelsPresent = $false
        break
    }
}

if ($AllModelsPresent) {
    Write-Host "  [OK] All required production models are already present in local_model_vault." -ForegroundColor Green
} else {
    Write-Host "  Missing models detected. Downloading required models..." -ForegroundColor Cyan
    
    if (-not [string]::IsNullOrEmpty($ModelRepoId) -and -not [string]::IsNullOrEmpty($HfToken)) {
        Write-Host "  Downloading from private vault: $ModelRepoId" -ForegroundColor Cyan
        $env:BAIF_MODEL_REPO_ID = $ModelRepoId
        $env:HF_TOKEN = $HfToken
        & $VenvPython download_production_models.py
        Remove-Item Env:HF_TOKEN -ErrorAction SilentlyContinue
    } else {
        Write-Host "  Downloading production models directly via download_models.py..." -ForegroundColor Cyan
        & $VenvPython download_models.py
    }
}

# ------------------------------------------------------------------------------
# 6. Run System Validation
# ------------------------------------------------------------------------------
Write-Host ""
Write-Host "[6/6] Validating installation..." -ForegroundColor Yellow

& $VenvPython verify_installation.py

if (-not $SkipModelVerification) {
    Write-Host ""
    Write-Host "  Running complete model verification across all 6 directions and 3 TTS voices..." -ForegroundColor Cyan
    & $VenvPython verify_all_models.py
}

Write-Host ""
Write-Host "=================================================================" -ForegroundColor Green
Write-Host "  BAIF SETUP COMPLETE AND READY FOR OFFLINE USE!" -ForegroundColor Green
Write-Host "=================================================================" -ForegroundColor Green
Write-Host ""
Write-Host "To start the application, simply run:" -ForegroundColor Cyan
Write-Host "   .\run_windows.bat" -ForegroundColor White
Write-Host "   or: .\.venv\Scripts\python.exe -m streamlit run app\web_ui.py" -ForegroundColor Gray
Write-Host ""
