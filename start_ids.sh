#!/usr/bin/env bash
# Flow Risk Monitor - one-click local startup for Linux
# Place this file in the project root, beside .env, backend/, and frontend/.

set -Eeuo pipefail

PROJECT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_DIR"

# Find the project's virtual environment.
if [[ -x ".venv/bin/python" ]]; then
    PYTHON="$PROJECT_DIR/.venv/bin/python"
elif [[ -x "venv/bin/python" ]]; then
    PYTHON="$PROJECT_DIR/venv/bin/python"
else
    echo
    echo "ERROR: Could not find a Python virtual environment."
    echo "Expected .venv/bin/python or venv/bin/python"
    echo "Create one with: python3 -m venv .venv"
    echo "Then install dependencies using: .venv/bin/pip install -r requirements.txt"
    echo
    exit 1
fi

# Read GOOGLE_CLIENT_ID from .env without sourcing/executing the file.
CLIENT_ID=""
if [[ -f ".env" ]]; then
    CLIENT_ID="$(sed -n 's/^[[:space:]]*GOOGLE_CLIENT_ID[[:space:]]*=[[:space:]]*//p' .env | head -n 1)"
    CLIENT_ID="${CLIENT_ID%$'\r'}"
    CLIENT_ID="${CLIENT_ID#\"}"
    CLIENT_ID="${CLIENT_ID%\"}"
    CLIENT_ID="${CLIENT_ID#\'}"
    CLIENT_ID="${CLIENT_ID%\'}"
fi

echo
echo "Starting Flow Risk Monitor..."
echo

# Stop both servers when this script exits or receives Ctrl+C.
BACKEND_PID=""
FRONTEND_PID=""
cleanup() {
    echo
    echo "Stopping Flow Risk Monitor..."
    [[ -z "$BACKEND_PID" ]] || kill "$BACKEND_PID" 2>/dev/null || true
    [[ -z "$FRONTEND_PID" ]] || kill "$FRONTEND_PID" 2>/dev/null || true
    wait 2>/dev/null || true
}
trap cleanup EXIT INT TERM

# Start FastAPI backend on port 8000.
"$PYTHON" -m uvicorn backend.main:app --reload &
BACKEND_PID=$!

# Start dashboard server on port 5500.
"$PYTHON" -m http.server 5500 --directory "$PROJECT_DIR/frontend" &
FRONTEND_PID=$!

# Give both servers a moment to start.
sleep 3

# Check whether either server exited during startup.
if ! kill -0 "$BACKEND_PID" 2>/dev/null; then
    echo "ERROR: Backend failed to start. Check the error output above."
    exit 1
fi
if ! kill -0 "$FRONTEND_PID" 2>/dev/null; then
    echo "ERROR: Frontend server failed to start. Check the error output above."
    exit 1
fi

# Open the dashboard.
if [[ -n "$CLIENT_ID" ]]; then
    DASHBOARD_URL="http://localhost:5500/dashboard.html?client_id=${CLIENT_ID}"
else
    echo "WARNING: GOOGLE_CLIENT_ID was not found in .env"
    DASHBOARD_URL="http://localhost:5500/dashboard.html"
fi

if command -v xdg-open >/dev/null 2>&1; then
    xdg-open "$DASHBOARD_URL" >/dev/null 2>&1 &
elif command -v gio >/dev/null 2>&1; then
    gio open "$DASHBOARD_URL" >/dev/null 2>&1 &
else
    echo "Could not detect a desktop browser opener. Open this URL manually:"
    echo "$DASHBOARD_URL"
fi

echo
echo "Backend : http://localhost:8000"
echo "Frontend: http://localhost:5500"
echo "Dashboard: $DASHBOARD_URL"
echo
echo "Press Ctrl+C to stop both servers."
echo

# Keep the script alive while either server is running.
wait -n "$BACKEND_PID" "$FRONTEND_PID"
