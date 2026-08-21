# BAIF Offline Translation Engine - Windows System Validation Script
[CmdletBinding()]
param(
    [switch]$FastPreflightOnly
)

$ErrorActionPreference = "Stop"
$env:PYTHONUTF8 = "1"
$env:PYTHONIOENCODING = "utf-8"

Write-Host "=================================================================" -ForegroundColor Cyan
Write-Host "  BAIF Translation Engine - System & Model Validation" -ForegroundColor Cyan
Write-Host "=================================================================" -ForegroundColor Cyan
Write-Host ""

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $ScriptDir

# 1. Ensure FFmpeg is accessible
$CommonFFmpegPaths = @(
    "C:\ffmpeg\bin",
    "C:\Program Files\ffmpeg\bin",
    "C:\tools\ffmpeg\bin",
    "$env:LOCALAPPDATA\Programs\ffmpeg\bin"
)
foreach ($p in $CommonFFmpegPaths) {
    if ((Test-Path (Join-Path $p "ffmpeg.exe")) -and (Test-Path (Join-Path $p "ffprobe.exe"))) {
        $env:Path = "$env:Path;$p"
        break
    }
}

# 2. Check virtual environment
$VenvPython = Join-Path $ScriptDir ".venv\Scripts\python.exe"
if (-not (Test-Path $VenvPython)) {
    Write-Host "[ERROR] Virtual environment (.venv) not found!" -ForegroundColor Red
    Write-Host "[ACTION REQUIRED] Please run setup first by executing: .\setup_windows.bat" -ForegroundColor Yellow
    exit 1
}

Write-Host "[1/2] Running installation pre-flight check..." -ForegroundColor Yellow
& $VenvPython verify_installation.py
$preflightCode = $LASTEXITCODE

if ($preflightCode -ne 0) {
    Write-Host ""
    Write-Host "[ERROR] Installation pre-flight failed." -ForegroundColor Red
    Write-Host "Manual Steps to Resolve:" -ForegroundColor Yellow
    Write-Host "  1. If ffmpeg/ffprobe is missing: Download FFmpeg x64 from https://www.gyan.dev/ffmpeg/builds/, extract to C:\ffmpeg\bin, and add to PATH." -ForegroundColor Gray
    Write-Host "  2. If Python packages are missing: Run .\setup_windows.bat to re-install dependencies." -ForegroundColor Gray
    Write-Host "  3. If model files are missing: Run .\setup_windows.bat to download models." -ForegroundColor Gray
    exit 1
}

if ($FastPreflightOnly) {
    Write-Host ""
    Write-Host "[SUCCESS] Fast pre-flight passed." -ForegroundColor Green
    exit 0
}

Write-Host ""
Write-Host "[2/2] Running full model inference verification (ASR, NMT, TTS)..." -ForegroundColor Yellow
& $VenvPython verify_all_models.py
$modelCode = $LASTEXITCODE

if ($modelCode -ne 0) {
    Write-Host ""
    Write-Host "[ERROR] Model verification encountered errors." -ForegroundColor Red
    Write-Host "Manual Steps to Resolve:" -ForegroundColor Yellow
    Write-Host "  - Ensure all models are fully downloaded without interruption." -ForegroundColor Gray
    Write-Host "  - Re-run model provisioning via .\setup_windows.bat" -ForegroundColor Gray
    exit 1
}

Write-Host ""
Write-Host "=================================================================" -ForegroundColor Green
Write-Host "  ALL VALIDATION CHECKS PASSED SUCCESSFULLY!" -ForegroundColor Green
Write-Host "  The system is fully ready for offline translation and dubbing." -ForegroundColor Green
Write-Host "=================================================================" -ForegroundColor Green
