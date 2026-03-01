#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
BIN="$SCRIPT_DIR/zig-out/bin/nullclaw"
PORT="${BRAIN_ARCH_PORT:-21789}"

usage() {
  cat <<EOF
brain-arch v2 — agent management

Usage: ./agent.sh <command> [args]

Commands:
  gateway              Start the gateway (Telegram + HTTP)
  chat                 Interactive CLI chat
  chat "message"       One-shot message
  brain-dashboard      Open brain dashboard in browser
  brain-dashboard serve  Start gateway + open dashboard
  status               Show gateway status
  logs                 Tail gateway logs
  stop                 Stop the gateway
  build                Build the binary (with Telegram)
  help                 Show this help

Environment:
  BRAIN_ARCH_PORT      Gateway port (default: 21789)
EOF
}

require_bin() {
  if [ ! -f "$BIN" ]; then
    echo "Binary not found. Run: ./agent.sh build"
    exit 1
  fi
}

case "${1:-help}" in
  gateway)
    require_bin
    exec "$BIN" gateway
    ;;
  chat)
    require_bin
    shift
    if [ $# -gt 0 ]; then
      exec "$BIN" agent -m "$*"
    else
      exec "$BIN" agent
    fi
    ;;
  brain-dashboard)
    require_bin
    if [ "${2:-}" = "serve" ]; then
      echo "🧠 Starting gateway + brain dashboard..."
      echo "   Gateway:  http://localhost:$PORT"
      echo "   Dashboard: http://localhost:$PORT/brain/dashboard"
      echo ""
      # Start gateway in background, open browser, then attach
      "$BIN" gateway &
      GW_PID=$!
      # Wait for gateway to be ready
      for i in $(seq 1 30); do
        if curl -sf "http://localhost:$PORT/health" >/dev/null 2>&1; then
          break
        fi
        sleep 0.5
      done
      echo "🌐 Opening dashboard..."
      open "http://localhost:$PORT/brain/dashboard" 2>/dev/null || \
        xdg-open "http://localhost:$PORT/brain/dashboard" 2>/dev/null || \
        echo "Open http://localhost:$PORT/brain/dashboard in your browser"
      # Attach to gateway foreground
      wait $GW_PID
    else
      echo "🧠 Opening brain dashboard at http://localhost:$PORT/brain/dashboard"
      open "http://localhost:$PORT/brain/dashboard" 2>/dev/null || \
        xdg-open "http://localhost:$PORT/brain/dashboard" 2>/dev/null || \
        echo "Open http://localhost:$PORT/brain/dashboard in your browser"
    fi
    ;;
  status)
    curl -sf "http://localhost:$PORT/ready" 2>/dev/null | python3 -m json.tool 2>/dev/null || echo "Gateway not running"
    ;;
  stop)
    pkill -f "nullclaw gateway" 2>/dev/null && echo "Stopped" || echo "Not running"
    ;;
  build)
    echo "Building brain-arch v2..."
    ZIG="${ZIG:-$(which zig 2>/dev/null || echo /tmp/zig-aarch64-macos-0.15.2/zig)}"
    "$ZIG" build -Dchannels=telegram -Doptimize=ReleaseSafe
    echo "Done. Binary: $BIN"
    ;;
  logs)
    # Tail the process output
    echo "Attach to running gateway with: ./agent.sh gateway"
    ;;
  help|--help|-h|"")
    usage
    ;;
  *)
    echo "Unknown command: $1"
    usage
    exit 1
    ;;
esac
