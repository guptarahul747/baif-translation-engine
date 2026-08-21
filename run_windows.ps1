# ==============================================================================
# BAIF Offline Translation Engine - Windows One-Click Run Script
# ==============================================================================
[CmdletBinding()]
param(
    [int]$Port = 8501
)

$ErrorActionPreference = "Stop"
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8
$env:PYTHONUTF8 = "1"
$env:PYTHONIOENCODING = "utf-8"

Write-Host "=================================================================" -ForegroundColor Cyan
Write-Host "🌾 Starting BAIF Offline Translation Engine" -ForegroundColor Cyan
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
    Write-Host "❌ Virtual environment (.venv) not found!" -ForegroundColor Red
    Write-Host "👉 Please run the setup script first: .\setup_windows.bat" -ForegroundColor Yellow
    Exit 1
}

Write-Host "🚀 Launching Streamlit Web UI on port $Port..." -ForegroundColor Green
Write-Host "👉 Opening browser at: http://localhost:$Port" -ForegroundColor Cyan
Write-Host "💡 Press Ctrl+C in this terminal window to stop the server." -ForegroundColor Gray
Write-Host ""

Start-Process "http://localhost:$Port"

& $VenvPython -m streamlit run app\web_ui.py --server.port $Port
