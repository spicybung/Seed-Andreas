@echo off
setlocal
cd /d "%~dp0"

where py >nul 2>nul
if errorlevel 1 (
    echo Python 3.10 or newer is required only on the developer build machine.
    pause
    exit /b 1
)

py -m pip install --disable-pip-version-check --upgrade pyinstaller
if errorlevel 1 (
    echo PyInstaller installation failed.
    pause
    exit /b 1
)

py -m PyInstaller --noconfirm --clean SeedAndreas.spec
if errorlevel 1 (
    echo Standalone Windows build failed.
    pause
    exit /b 1
)

echo.
echo Standalone application created:
echo %~dp0dist\SeedAndreas.exe
pause
