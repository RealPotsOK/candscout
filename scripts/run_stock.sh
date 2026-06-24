#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON:-./.venv/bin/python}"
MAKE_BIN="${MAKE:-make}"

stock_end_resolved="${STOCK_END}"
stock_start_resolved="${STOCK_START:-}"
if [[ -z "$stock_start_resolved" ]]; then
  stock_start_resolved="$($PYTHON_BIN - <<'PY'
from datetime import datetime, timezone, timedelta
import os
raw = os.environ["STOCK_END"]
raw = raw[:-1] + "+00:00" if raw.endswith("Z") else raw
end = datetime.fromisoformat(raw)
end = end.replace(tzinfo=timezone.utc) if end.tzinfo is None else end.astimezone(timezone.utc)
start = end - timedelta(days=int(os.environ["STOCK_LOOKBACK_DAYS"]))
print(start.isoformat().replace("+00:00", "Z"))
PY
)"
fi

stock_symbol="$($PYTHON_BIN src/select_stock.py \
  --symbol "${STOCK_SYMBOL:-}" \
  --stock-list "${STOCK_LIST}" \
  --interval "${STOCK_INTERVAL}" \
  --start "$stock_start_resolved" \
  --end "$stock_end_resolved" \
  --min-rows "${STOCK_MIN_ROWS}" \
  --min-history-years "${STOCK_MIN_HISTORY_YEARS}")"

if [[ -z "$stock_symbol" ]]; then
  echo "No eligible stock selected"
  exit 1
fi

raw_data_path="data/downloads/yahoo/${stock_symbol}/${STOCK_INTERVAL}/candles.parquet"
archive_name="before-stock-${stock_symbol}-$(date -u +%Y%m%dT%H%M%SZ)"
stock_archive_name="stock-${stock_symbol}-$(date -u +%Y%m%dT%H%M%SZ)"

echo "Selected stock: ${stock_symbol}"
echo "Stock data source: yahoo"
echo "Stock interval: ${STOCK_INTERVAL}"
echo "Stock date range: ${stock_start_resolved} to ${stock_end_resolved}"
echo "Stock history filter: >= ${STOCK_MIN_HISTORY_YEARS} years and >= ${STOCK_MIN_ROWS} rows"
echo "Stock split: ${STOCK_SPLIT} train, final ${STOCK_SIM_TEST_FRACTION} simulation fraction"

if [[ "${STOCK_ARCHIVE_CURRENT}" != "0" ]]; then
  echo "Archiving current NN model/config before stock run: ${archive_name}"
  "$MAKE_BIN" nn-save NN_ARCHIVE_NAME="${archive_name}"
fi

"$MAKE_BIN" download \
  DATA_SOURCE=yahoo \
  SYMBOL="${stock_symbol}" \
  RANDOM_STOCK=0 \
  INTERVAL="${STOCK_INTERVAL}" \
  START="${stock_start_resolved}" \
  END="${stock_end_resolved}"

"$PYTHON_BIN" - "$raw_data_path" "${STOCK_MIN_ROWS}" <<'PY'
import sys
import pandas as pd
p = sys.argv[1]
minimum = int(sys.argv[2])
df = pd.read_parquet(p)
t = pd.to_datetime(df["open_time"], utc=True)
n = len(df)
print(f"Stock downloaded rows: {n}")
print(f"Stock downloaded range: {t.min()} to {t.max()}")
if n < minimum:
    raise SystemExit(f"Only {n} rows downloaded; expected at least {minimum}. Increase STOCK_LOOKBACK_DAYS or check the Yahoo range.")
PY

"$MAKE_BIN" experiment visualize nn-sim nn-sim-visualize reports-index \
  ASSET_ENV="${STOCK_ASSET_ENV}" \
  DATA_SOURCE=yahoo \
  SYMBOL="${stock_symbol}" \
  RANDOM_STOCK=0 \
  INTERVAL="${STOCK_INTERVAL}" \
  START="${stock_start_resolved}" \
  END="${stock_end_resolved}" \
  SPLIT="${STOCK_SPLIT}" \
  SIM_DEFAULT_TEST_FRACTION="${STOCK_SIM_TEST_FRACTION}"

echo "Archiving stock NN model/config: ${stock_archive_name}"
"$MAKE_BIN" nn-save \
  ASSET_ENV="${STOCK_ASSET_ENV}" \
  DATA_SOURCE=yahoo \
  SYMBOL="${stock_symbol}" \
  RANDOM_STOCK=0 \
  INTERVAL="${STOCK_INTERVAL}" \
  START="${stock_start_resolved}" \
  END="${stock_end_resolved}" \
  SPLIT="${STOCK_SPLIT}" \
  SIM_DEFAULT_TEST_FRACTION="${STOCK_SIM_TEST_FRACTION}" \
  NN_ARCHIVE_NAME="${stock_archive_name}"

echo "Open model visualization on your other laptop:"
echo "http://${REPORTS_HOST}:${REPORTS_PORT}/nn/${NN_MODEL_TYPE}/yahoo/${stock_symbol}/${STOCK_INTERVAL}/visualization.html"
echo "Open simulation visualization on your other laptop:"
echo "http://${REPORTS_HOST}:${REPORTS_PORT}/sim/nn/${NN_MODEL_TYPE}/yahoo/${stock_symbol}/${STOCK_INTERVAL}/visualization.html"
echo "Report index:"
echo "http://${REPORTS_HOST}:${REPORTS_PORT}/${REPORTS_INDEX_URL_PATH}"

if curl -fsS --max-time 2 "http://${REPORTS_HOST}:${REPORTS_PORT}/${REPORTS_INDEX_URL_PATH}" >/dev/null 2>&1; then
  echo "Reports server is running on ${REPORTS_HOST}:${REPORTS_PORT}."
else
  echo "Starting dashboard server in the background on ${REPORTS_HOST}:${REPORTS_PORT}."
  mkdir -p data/reports
  setsid "$PYTHON_BIN" src/dashboard_server.py \
    --host "${REPORTS_HOST}" \
    --port "${REPORTS_PORT}" \
    --reports-root data/reports \
    --root . \
    --live-url "${LIVE_LOCAL_URL}" \
    --live-public-url "${LIVE_PUBLIC_URL}" \
    > data/reports/dashboard_server.log 2>&1 < /dev/null &
  sleep 1
  curl -fsS --max-time 3 "http://${REPORTS_HOST}:${REPORTS_PORT}/api/dashboard/status" >/dev/null \
    && echo "Dashboard server started." \
    || { echo "Dashboard server did not respond. Check data/reports/dashboard_server.log"; exit 1; }
fi
