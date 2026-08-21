@echo off
setlocal
chcp 65001 >nul
set PYTHONUTF8=1
set PYTHONIOENCODING=utf-8

echo =================================================================
echo  BAIF Offline Translation Engine - Windows One-Click Setup
echo =================================================================
echo.

cd /d "%~dp0"

powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0setup_windows.ps1" %*

if %ERRORLEVEL% NEQ 0 (
    echo.
    echo [ERROR] Setup encountered an issue. Please review the output above.
    if not defined NO_PAUSE pause
    exit /b %ERRORLEVEL%
)

echo.
echo Setup finished successfully!
if not defined NO_PAUSE pause
