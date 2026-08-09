#!/usr/bin/env bash
# start.sh — Single-command launcher for the Driver Drowsiness Detection System.
# Starts the FastAPI backend and React frontend concurrently.
# Usage: bash start.sh [--no-browser]

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

NO_BROWSER=false
for arg in "$@"; do
    [[ "$arg" == "--no-browser" ]] && NO_BROWSER=true
done

# ── Colour helpers ────────────────────────────────────────────────────────────
RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'; NC='\033[0m'
info()    { echo -e "${GREEN}[start]${NC} $*"; }
warning() { echo -e "${YELLOW}[start]${NC} $*"; }
error()   { echo -e "${RED}[start]${NC} $*" >&2; }

# ── Cleanup on exit ───────────────────────────────────────────────────────────
PIDS=()
cleanup() {
    echo ""
    info "Shutting down…"
    for pid in "${PIDS[@]}"; do
        kill "$pid" 2>/dev/null || true
    done
    wait 2>/dev/null
    info "Done."
}
trap cleanup EXIT INT TERM

# ── Python environment ────────────────────────────────────────────────────────
if [[ -f "venv/bin/activate" ]]; then
    # shellcheck source=/dev/null
    source venv/bin/activate
    info "Activated virtual environment (venv/)."
elif [[ -f ".venv/bin/activate" ]]; then
    # shellcheck source=/dev/null
    source .venv/bin/activate
    info "Activated virtual environment (.venv/)."
else
    warning "No virtual environment found. Using system Python."
    warning "Consider running: python -m venv venv && source venv/bin/activate"
fi

# ── Python dependencies ───────────────────────────────────────────────────────
if ! python -c "import fastapi, cv2, mediapipe" 2>/dev/null; then
    info "Installing Python dependencies from requirements.txt…"
    pip install -r requirements.txt
fi

# ── Frontend dependencies ─────────────────────────────────────────────────────
if [[ ! -d "frontend/node_modules" ]]; then
    info "Installing frontend dependencies (npm install)…"
    npm --prefix frontend install
fi

# ── Start FastAPI backend ─────────────────────────────────────────────────────
info "Starting FastAPI backend on http://localhost:8000 …"
python server.py &
BACKEND_PID=$!
PIDS+=("$BACKEND_PID")

# Wait briefly for the backend to become ready
for i in $(seq 1 20); do
    if curl -sf http://localhost:8000/api/status >/dev/null 2>&1; then
        break
    fi
    sleep 0.5
done

# ── Start React frontend ──────────────────────────────────────────────────────
info "Starting React frontend on http://localhost:5173 …"
npm --prefix frontend run dev &
FRONTEND_PID=$!
PIDS+=("$FRONTEND_PID")

# ── Open browser ─────────────────────────────────────────────────────────────
if [[ "$NO_BROWSER" == false ]]; then
    sleep 2   # give Vite a moment to bind its port
    if command -v open >/dev/null 2>&1; then
        open http://localhost:5173
    elif command -v xdg-open >/dev/null 2>&1; then
        xdg-open http://localhost:5173
    fi
fi

info "System is running. Press Ctrl+C to stop."
info "  Backend  → http://localhost:8000"
info "  Frontend → http://localhost:5173"
echo ""

# Keep script alive until both child processes exit or user interrupts
wait "${PIDS[@]}"
