@echo off
setlocal
cd /d "%~dp0"

where py >nul 2>nul
if errorlevel 1 (
    echo Python was not found. Install Python 3.10 or newer and enable the Python launcher.
    pause
    exit /b 1
)

powershell -NoProfile -WindowStyle Hidden -ExecutionPolicy Bypass -Command "Start-Process -FilePath 'pyw' -ArgumentList 'SeedAndreas.pyw' -WorkingDirectory '%~dp0' -Verb RunAs"
