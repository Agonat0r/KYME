@echo off
title KYMA
cd /d "%~dp0"

echo.
echo   ========================================
echo    KYMA - Biosignal Control Platform
echo   ========================================
echo.

:: ── Check Python ──────────────────────────────────────────────
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo   [ERROR] Python not found.
    echo   Install Python 3.10+ from https://python.org
    echo   Make sure "Add to PATH" is checked during install.
    echo.
    pause
    exit /b 1
)

:: ── First-run setup ───────────────────────────────────────────
if not exist ".venv\Scripts\python.exe" (
    echo   First-time setup - this only happens once.
    echo   NOTE: PyTorch is ~2GB, this may take a few minutes.
    echo.
    echo   [1/3] Creating virtual environment...
    python -m venv .venv
    if %errorlevel% neq 0 (
        echo   [ERROR] Failed to create virtual environment.
        pause
        exit /b 1
    )
    echo   [2/3] Installing dependencies - this will take a while...
    .venv\Scripts\pip install -r requirements.txt --disable-pip-version-check
    if %errorlevel% neq 0 (
        echo   [ERROR] Failed to install dependencies.
        pause
        exit /b 1
    )
    echo   [3/3] Installing desktop window...
    .venv\Scripts\pip install pywebview --disable-pip-version-check
    echo.
    echo   Setup complete!
    echo.
)

:: ── Launch ────────────────────────────────────────────────────
echo   Starting KYMA...
echo   Close this window to stop.
echo.

if not exist "launch.py" (
    echo   [ERROR] launch.py not found. Make sure you're in the KYMA folder.
    pause
    exit /b 1
)

.venv\Scripts\python launch.py
echo.
echo   KYMA has stopped. Exit code: %errorlevel%
echo.
pause
