@echo off
setlocal

REM ============================================================
REM Flow Risk Monitor - one-click local startup
REM Place this file in the project root, beside .env, backend/, frontend/
REM ============================================================

cd /d "%~dp0"

REM ---- Find the project's virtual environment -----------------
if exist ".venv\Scripts\python.exe" (
    set "PYTHON=%~dp0.venv\Scripts\python.exe"
) else if exist "venv\Scripts\python.exe" (
    set "PYTHON=%~dp0venv\Scripts\python.exe"
) else (
    echo.
    echo ERROR: Could not find a Python virtual environment.
    echo Expected .venv\Scripts\python.exe or venv\Scripts\python.exe
    echo.
    pause
    exit /b 1
)

REM ---- Read the public Google Client ID from .env --------------
if exist ".env" (
    for /f "tokens=1,* delims==" %%A in ('findstr /b "GOOGLE_CLIENT_ID=" ".env"') do (
        if "%%A"=="GOOGLE_CLIENT_ID" set "CLIENT_ID=%%B"
    )
)

echo.
echo Starting Flow Risk Monitor...
echo.

REM ---- Start FastAPI backend on port 8000 ----------------------
start "IDS Backend - FastAPI" "%PYTHON%" -m uvicorn backend.main:app --reload

REM ---- Start dashboard server on port 5500 ---------------------
start "IDS Frontend - Dashboard" "%PYTHON%" -m http.server 5500 --directory "%~dp0frontend"

REM ---- Give both servers a moment to start ---------------------
timeout /t 3 /nobreak >nul

REM ---- Open dashboard with Google Client ID --------------------
if defined CLIENT_ID (
    start "" "http://localhost:5500/dashboard.html?client_id=%CLIENT_ID%"
) else (
    echo WARNING: GOOGLE_CLIENT_ID was not found in .env
    start "" "http://localhost:5500/dashboard.html"
)

echo.
echo Backend : http://localhost:8000
echo Frontend: http://localhost:5500
echo.
echo Close the two server windows to stop the project.
echo.
exit /b 0
