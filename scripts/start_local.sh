#!/usr/bin/env bash
# One-command local fallback that runs both API (FastAPI) and UI (Next.js)
# Usage: ./scripts/start_local.sh

set -e

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

echo "=== Starting NLP Support System Locally ==="

# Trap exit to cleanup background jobs
cleanup() {
  echo ""
  echo "Stopping services..."
  kill $(jobs -p) 2>/dev/null || true
  exit 0
}
trap cleanup SIGINT SIGTERM EXIT

# 1. Start FastAPI backend
echo "-> Starting FastAPI backend on http://localhost:8000..."
if [ -d ".venv" ]; then
  source .venv/bin/activate
fi
PYTHONPATH=. uvicorn api.main:app --host 0.0.0.0 --port 8000 &
API_PID=$!

# Wait for API to become ready
echo "Waiting for API to respond on /health..."
for i in {1..15}; do
  if curl -s http://localhost:8000/health > /dev/null 2>&1; then
    echo "✓ API is ready!"
    break
  fi
  sleep 1
done

# 2. Start Next.js frontend
echo "-> Starting Next.js frontend on http://localhost:3000..."
cd "$ROOT_DIR/web"
npm run dev &
WEB_PID=$!

echo ""
echo "=========================================================="
echo "  NLP Support System is live!"
echo "  Frontend : http://localhost:3000"
echo "  Backend  : http://localhost:8000"
echo "  API Docs : http://localhost:8000/docs"
echo "=========================================================="
echo ""
echo "To expose locally via Cloudflare Tunnel (no account required):"
echo "  npx cloudflared tunnel --url http://localhost:3000"
echo ""
echo "Press Ctrl+C to stop all services."

wait
