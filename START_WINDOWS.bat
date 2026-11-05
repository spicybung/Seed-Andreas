@echo off
setlocal
cd /d "%~dp0"

where py >nul 2>nul
if errorlevel 1 (
    echo Python was not found. Install Python 3.10 or newer and enable the Python launcher.
    pause
    exit /b 1
)

py SeedAndreas.pyw
if errorlevel 1 (
    echo.
    echo Seed Andreas failed to start. Check SeedAndreas_startup_error.txt in this folder.
    pause
)
