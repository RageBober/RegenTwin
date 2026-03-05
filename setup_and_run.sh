#!/bin/bash
# RegenTwin — Quick Setup and Run (Backend + Frontend)
set -e

echo "RegenTwin — Setup and Run"
echo "======================================="

# Check Python
if ! command -v python3 &> /dev/null && ! command -v python &> /dev/null; then
    echo "Python not found. Install Python 3.11+"
    exit 1
fi
PYTHON=$(command -v python3 || command -v python)

# Check Node.js
if ! command -v node &> /dev/null; then
    echo "Node.js not found. Install Node.js 18+"
    exit 1
fi

# Create venv if not exists
if [ ! -d ".venv" ]; then
    echo "Creating virtual environment..."
    $PYTHON -m venv .venv
fi

# Activate venv
source .venv/bin/activate 2>/dev/null || source .venv/Scripts/activate 2>/dev/null

# Install Python deps
echo "Installing Python dependencies..."
pip install -e ".[dev]" -q 2>/dev/null || pip install numpy scipy pandas plotly fastapi uvicorn pydantic-settings loguru sqlalchemy alembic python-multipart -q

# Install Node deps
if [ ! -d "ui/node_modules" ]; then
    echo "Installing Node.js dependencies..."
    (cd ui && npm install)
fi

echo ""
echo "Starting RegenTwin..."
echo "  Backend:  http://localhost:8000"
echo "  Frontend: http://localhost:5173"
echo ""
echo "Press Ctrl+C to stop both services."
echo ""

# Start backend in background
$PYTHON -m uvicorn src.api.main:app --host 127.0.0.1 --port 8000 &
BACKEND_PID=$!

# Cleanup on exit
cleanup() {
    echo "Stopping backend (PID $BACKEND_PID)..."
    kill $BACKEND_PID 2>/dev/null
    wait $BACKEND_PID 2>/dev/null
}
trap cleanup EXIT INT TERM

# Small delay to let backend start
sleep 2

# Start frontend (foreground)
cd ui
npm run dev
