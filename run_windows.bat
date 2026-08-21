@echo off
setlocal
chcp 65001 >nul
set PYTHONUTF8=1
set PYTHONIOENCODING=utf-8

cd /d "%~dp0"

powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0run_windows.ps1" %*

if %ERRORLEVEL% NEQ 0 (
    echo.
    echo [ERROR] Application exited with an error.
    pause
)
