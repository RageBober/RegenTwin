@echo off
REM RegenTwin — Quick Setup and Run (Backend + Frontend)
echo RegenTwin — Setup and Run
echo =======================================
echo.

REM Check Python
python --version >nul 2>&1
if errorlevel 1 (
    echo Python not found. Install Python 3.11+
    pause
    exit /b 1
)

REM Check Node.js
node --version >nul 2>&1
if errorlevel 1 (
    echo Node.js not found. Install Node.js 18+
    pause
    exit /b 1
)

REM Create venv if not exists
if not exist ".venv" (
    echo Creating virtual environment...
    python -m venv .venv
)

REM Activate venv
call .venv\Scripts\activate.bat

REM Install Python deps
echo Installing Python dependencies...
pip install -e ".[dev]" -q 2>nul || pip install numpy scipy pandas plotly fastapi uvicorn pydantic-settings loguru sqlalchemy alembic python-multipart -q

REM Install Node deps
if not exist "ui\node_modules" (
    echo Installing Node.js dependencies...
    cd ui
    npm install
    cd ..
)

echo.
echo Starting RegenTwin...
echo   Backend:  http://localhost:8000
echo   Frontend: http://localhost:5173
echo.
echo Press Ctrl+C to stop both services.
echo.

REM Start backend in background
start /b "RegenTwin Backend" python -m uvicorn src.api.main:app --host 127.0.0.1 --port 8000

REM Small delay to let backend start
timeout /t 2 /nobreak >nul

REM Start frontend (foreground — Ctrl+C stops it)
cd ui
npm run dev

REM When frontend stops, kill backend
taskkill /f /fi "WINDOWTITLE eq RegenTwin Backend" >nul 2>&1
