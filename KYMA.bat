@echo off
title KYMA
cd /d "%~dp0"
set "PYTHONNOUSERSITE=1"

echo.
echo   ========================================
echo    KYMA - Biosignal Control Platform
echo   ========================================
echo.

set "KYMA_PYTHON="
if exist ".venv\Scripts\python.exe" (
    .venv\Scripts\python.exe -s -c "import numpy, fastapi, uvicorn, pydantic" >nul 2>&1 && set "KYMA_PYTHON=.venv\Scripts\python.exe"
)

if not defined KYMA_PYTHON if exist "C:\packman-repo\python\3.10.18-nv1-windows-x86_64\python.exe" (
    echo   [WARN] Local .venv is unavailable or unhealthy; using bundled runtime.
    set "KYMA_PYTHON=C:\packman-repo\python\3.10.18-nv1-windows-x86_64\python.exe"
    set "PYTHONPATH=%cd%\server"
)

if not defined KYMA_PYTHON (
    set "KYMA_PYTHON=python"
)

"%KYMA_PYTHON%" --version >nul 2>&1
if %errorlevel% neq 0 (
    echo   [ERROR] Python not found.
    echo   Install Python 3.10+ from https://python.org
    echo   Make sure "Add to PATH" is checked during install.
    echo.
    pause
    exit /b 1
)

if not exist ".venv\Scripts\python.exe" (
    echo   First-time setup - this only happens once.
    echo   NOTE: PyTorch is ~2GB, this may take a few minutes.
    echo.
    echo   [1/3] Creating virtual environment...
    "%KYMA_PYTHON%" -s -m venv .venv
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

echo   Starting KYMA...
echo   Close this window to stop.
echo.

if not exist "launch.py" (
    echo   [ERROR] launch.py not found. Make sure you're in the KYMA folder.
    pause
    exit /b 1
)

"%KYMA_PYTHON%" -s launch.py %*
echo.
echo   KYMA has stopped. Exit code: %errorlevel%
echo.
pause
