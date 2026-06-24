#!/usr/bin/env bash
set -euo pipefail
family="${1:?usage: visualize_sim.sh nn|lr|xgb|strategy}"
PYTHON_BIN="${PYTHON:-./.venv/bin/python}"
case "$family" in
  nn) trades="${NN_SIM_TRADES}"; report="${NN_SIM_REPORT}"; cmp_trades="${NN_SIM_LONG_SHORT_TRADES}"; cmp_report="${NN_SIM_LONG_SHORT_REPORT}"; output="${NN_SIM_VISUALIZATION_OUT}"; title="${SYMBOL} ${INTERVAL} Sequence NN Bank Simulation"; model_url="${NN_VISUALIZATION_URL_PATH}"; sim_url="${NN_SIM_VISUALIZATION_URL_PATH}"; label="Sequence NN simulation visualization" ;;
  lr) trades="${SIM_TRADES}"; report="${SIM_REPORT}"; cmp_trades="${SIM_LONG_SHORT_TRADES}"; cmp_report="${SIM_LONG_SHORT_REPORT}"; output="${SIM_VISUALIZATION_OUT}"; title="${SYMBOL} ${INTERVAL} Logistic Regression Bank Simulation"; model_url="${VISUALIZATION_URL_PATH}"; sim_url="${SIM_VISUALIZATION_URL_PATH}"; label="Daily bank visualization" ;;
  xgb) trades="${XGB_SIM_TRADES}"; report="${XGB_SIM_REPORT}"; cmp_trades="${XGB_SIM_LONG_SHORT_TRADES}"; cmp_report="${XGB_SIM_LONG_SHORT_REPORT}"; output="${XGB_SIM_VISUALIZATION_OUT}"; title="${SYMBOL} ${INTERVAL} XGBoost Bank Simulation"; model_url="${XGB_VISUALIZATION_URL_PATH}"; sim_url="${XGB_SIM_VISUALIZATION_URL_PATH}"; label="XGBoost simulation visualization" ;;
  strategy) trades="${STRATEGY_SIM_TRADES}"; report="${STRATEGY_SIM_REPORT}"; cmp_trades="${STRATEGY_SIM_LONG_SHORT_TRADES}"; cmp_report="${STRATEGY_SIM_LONG_SHORT_REPORT}"; output="${STRATEGY_SIM_VISUALIZATION_OUT}"; title="${SYMBOL} ${INTERVAL} Strategy Bank Simulation"; model_url="${STRATEGY_VISUALIZATION_URL_PATH}"; sim_url="${STRATEGY_SIM_VISUALIZATION_URL_PATH}"; label="Strategy simulation visualization" ;;
  *) echo "Unknown sim visualization family: $family" >&2; exit 2 ;;
esac
"$PYTHON_BIN" src/visualize_sim.py \
  --raw-data "${RAW_DATA}" \
  --trades "$trades" \
  --report "$report" \
  --comparison-trades "$cmp_trades" \
  --comparison-report "$cmp_report" \
  --output "$output" \
  --activity-bucket "${SIM_ACTIVITY_BUCKET}" \
  --marker-size-basis "${SIM_MARKER_SIZE_BASIS}" \
  --baseline-ma-windows "${SIM_BASELINE_MA_WINDOWS}" \
  --title "$title" \
  --nav-home-url "/${REPORTS_INDEX_URL_PATH}" \
  --nav-model-url "/${model_url}" \
  --nav-sim-url "/${sim_url}"
echo "$label: $output"
