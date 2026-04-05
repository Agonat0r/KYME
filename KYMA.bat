@echo off
title KYMA Launcher
echo.
echo  ============================================
echo   KYMA - Biosignal Control Platform
echo  ============================================
echo.

:: Check if Python is installed
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo  [ERROR] Python is not installed or not in PATH.
    echo  Please install Python 3.10+ from https://python.org
    echo.
    pause
    exit /b 1
)

:: Check if dependencies are installed, install if not
if not exist ".venv" (
    echo  [1/3] Creating virtual environment...
    python -m venv .venv
    echo  [2/3] Installing dependencies (first run only)...
    .venv\Scripts\pip install -r requirements.txt -q
    echo  [3/3] Installing desktop window support...
    .venv\Scripts\pip install pywebview -q
    echo.
    echo  Setup complete.
    echo.
) else (
    echo  Virtual environment found.
)

echo  Starting KYMA...
echo  Close this window to stop the server.
echo.

:: Launch with pywebview desktop window in mock mode
:: To use real hardware, remove --mock and set COM ports:
::   .venv\Scripts\python launch.py --cyton COM8 --arduino COM4
.venv\Scripts\python launch.py --mock
