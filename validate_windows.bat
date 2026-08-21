@echo off
setlocal
chcp 65001 >nul
set PYTHONUTF8=1
set PYTHONIOENCODING=utf-8

echo =================================================================
echo  BAIF Offline Translation Engine - System Validation
echo =================================================================
echo.

cd /d "%~dp0"

powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0validate_windows.ps1" %*

if %ERRORLEVEL% NEQ 0 (
    echo.
    echo [ERROR] Validation failed. Please check the recommendations above.
    if not defined NO_PAUSE pause
    exit /b %ERRORLEVEL%
)

echo.
if not defined NO_PAUSE pause
