@echo off
setlocal

echo RegenTwin - Setup and Run
echo ================================
echo.

uv --version >nul 2>&1
if errorlevel 1 (
    echo UV not found. Install UV first: https://docs.astral.sh/uv/
    pause
    exit /b 1
)

node --version >nul 2>&1
if errorlevel 1 (
    echo Node.js not found. Install Node.js 18+
    pause
    exit /b 1
)

echo Syncing Python environment...
uv sync --extra dev

if not exist "ui\node_modules" (
    echo Installing frontend dependencies...
    cd ui
    npm install
    cd ..
)

echo.
echo Starting RegenTwin...
echo   Backend:  http://localhost:8000
echo   Frontend: http://localhost:5173
echo.

echo Releasing port 8000 if needed...
uv run python scripts/kill_port.py 8000

echo Launching backend...
start /b "RegenTwin Backend" uv run uvicorn src.api.main:app --host 127.0.0.1 --port 8000

timeout /t 2 /nobreak >nul

cd ui
npm run dev
