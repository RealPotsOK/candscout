#!/usr/bin/env bash
set -euo pipefail
cmd="${1:?usage: dashboard.sh start|start-fg|stop|serve-lan|print-url <path>}"
PYTHON_BIN="${PYTHON:-./.venv/bin/python}"
MAKE_BIN="${MAKE:-make}"
REPORTS_HOST="${REPORTS_HOST:-192.168.2.197}"
REPORTS_PORT="${REPORTS_PORT:-8000}"
REPORTS_ROOT="${REPORTS_ROOT:-data/reports}"
LIVE_LOCAL_URL="${LIVE_LOCAL_URL:-http://127.0.0.1:${LIVE_HOST_PORT:-8080}}"
LIVE_PUBLIC_URL="${LIVE_PUBLIC_URL:-http://${REPORTS_HOST}:${LIVE_HOST_PORT:-8080}}"
CHECK_URL_PATH="${CHECK_URL_PATH:-${REPORTS_INDEX_URL_PATH:-index.html}}"

status_url="http://${REPORTS_HOST}:${REPORTS_PORT}/api/dashboard/status"

print_dashboard_urls() {
  echo "  main: http://${REPORTS_HOST}:${REPORTS_PORT}/"
  echo "  models: http://${REPORTS_HOST}:${REPORTS_PORT}/models"
  echo "  compare: http://${REPORTS_HOST}:${REPORTS_PORT}/compare"
  echo "  reports: http://${REPORTS_HOST}:${REPORTS_PORT}/reports"
  echo "  live: http://${REPORTS_HOST}:${REPORTS_PORT}/live"
}

port_pids() {
  if command -v lsof >/dev/null 2>&1; then
    lsof -tiTCP:"${REPORTS_PORT}" -sTCP:LISTEN 2>/dev/null || true
  elif command -v fuser >/dev/null 2>&1; then
    fuser -n tcp "${REPORTS_PORT}" 2>/dev/null || true
  elif command -v ss >/dev/null 2>&1; then
    ss -ltnp "sport = :${REPORTS_PORT}" 2>/dev/null | sed -n 's/.*pid=\([0-9][0-9]*\).*/\1/p' || true
  fi | sort -u
}

start_background() {
  if curl -fsS --max-time 2 "$status_url" >/dev/null 2>&1; then
    echo "CandScout dashboard already running:"
    print_dashboard_urls
    exit 0
  fi
  mkdir -p "$REPORTS_ROOT"
  echo "Starting CandScout dashboard in the background:"
  print_dashboard_urls
  echo "  live Docker target: ${LIVE_PUBLIC_URL}"
  setsid "$PYTHON_BIN" src/dashboard_server.py \
    --host "$REPORTS_HOST" \
    --port "$REPORTS_PORT" \
    --reports-root "$REPORTS_ROOT" \
    --root . \
    --live-url "$LIVE_LOCAL_URL" \
    --live-public-url "$LIVE_PUBLIC_URL" \
    > "${REPORTS_ROOT}/dashboard_server.log" 2>&1 < /dev/null &
  sleep 1
  if curl -fsS --max-time 3 "$status_url" >/dev/null; then
    echo "Dashboard started. Logs: ${REPORTS_ROOT}/dashboard_server.log"
  else
    echo "Dashboard did not respond. Check ${REPORTS_ROOT}/dashboard_server.log" >&2
    exit 1
  fi
}

case "$cmd" in
  start)
    start_background
    ;;
  start-fg)
    echo "Starting CandScout dashboard:"
    print_dashboard_urls
    echo "  live Docker target: ${LIVE_PUBLIC_URL}"
    exec "$PYTHON_BIN" src/dashboard_server.py \
      --host "$REPORTS_HOST" \
      --port "$REPORTS_PORT" \
      --reports-root "$REPORTS_ROOT" \
      --root . \
      --live-url "$LIVE_LOCAL_URL" \
      --live-public-url "$LIVE_PUBLIC_URL"
    ;;
  stop)
    echo "Stopping any server listening on port ${REPORTS_PORT}..."
    pids="$(port_pids)"
    if [[ -z "$pids" ]]; then
      echo "No process is listening on port ${REPORTS_PORT}."
      exit 0
    fi
    echo "Killing PID(s): ${pids}"
    kill $pids 2>/dev/null || true
    sleep 1
    still="$(port_pids)"
    if [[ -n "$still" ]]; then
      echo "Force killing PID(s): ${still}"
      kill -9 $still 2>/dev/null || true
    fi
    echo "Stopped server on port ${REPORTS_PORT}."
    ;;
  serve-lan)
    if curl -fsS --max-time 2 "$status_url" >/dev/null 2>&1; then
      echo "Dashboard server already appears to be running on ${REPORTS_HOST}:${REPORTS_PORT}."
    elif curl -fsS --max-time 2 "http://${REPORTS_HOST}:${REPORTS_PORT}/${CHECK_URL_PATH}" >/dev/null 2>&1; then
      echo "Port ${REPORTS_PORT} is already serving reports, but not the new dashboard."
      echo "Stop the old server, then run: make start"
      exit 1
    else
      "$MAKE_BIN" start
    fi
    ;;
  print-url)
    path="${2:?usage: dashboard.sh print-url <url-path>}"
    echo "Open this on your other laptop:"
    echo "http://${REPORTS_HOST}:${REPORTS_PORT}/${path}"
    echo "Report index:"
    echo "http://${REPORTS_HOST}:${REPORTS_PORT}/${REPORTS_INDEX_URL_PATH:-index.html}"
    ;;
  *)
    echo "Unknown dashboard command: $cmd" >&2
    exit 2
    ;;
esac
