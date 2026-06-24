#!/usr/bin/env bash
set -euo pipefail
family="${1:?usage: visualize_model.sh nn|lr|xgb|strategy}"
PYTHON_BIN="${PYTHON:-./.venv/bin/python}"
case "$family" in
  nn) predictions="${NN_PREDICTIONS_OUT}"; output="${NN_VISUALIZATION_OUT}"; title="${SYMBOL} ${INTERVAL} Sequence NN Inspection"; model_url="${NN_VISUALIZATION_URL_PATH}"; sim_url="${NN_SIM_VISUALIZATION_URL_PATH}"; label="Sequence NN visualization" ;;
  lr) predictions="${PREDICTIONS_OUT}"; output="${VISUALIZATION_OUT}"; title="${SYMBOL} ${INTERVAL} Model Inspection"; model_url="${VISUALIZATION_URL_PATH}"; sim_url="${SIM_VISUALIZATION_URL_PATH}"; label="Visualization" ;;
  xgb) predictions="${XGB_PREDICTIONS_OUT}"; output="${XGB_VISUALIZATION_OUT}"; title="${SYMBOL} ${INTERVAL} XGBoost Inspection"; model_url="${XGB_VISUALIZATION_URL_PATH}"; sim_url="${XGB_SIM_VISUALIZATION_URL_PATH}"; label="XGBoost visualization" ;;
  strategy) predictions="${STRATEGY_PREDICTIONS_OUT}"; output="${STRATEGY_VISUALIZATION_OUT}"; title="${SYMBOL} ${INTERVAL} Strategy ${STRATEGY_MODEL_TYPE} Inspection"; model_url="${STRATEGY_VISUALIZATION_URL_PATH}"; sim_url="${STRATEGY_SIM_VISUALIZATION_URL_PATH}"; label="Strategy visualization" ;;
  *) echo "Unknown visualization family: $family" >&2; exit 2 ;;
esac
"$PYTHON_BIN" src/visualize.py \
  --raw-data "${RAW_DATA}" \
  --predictions "$predictions" \
  --output "$output" \
  --threshold "${THRESHOLD}" \
  --exit-threshold "${EXIT_THRESHOLD}" \
  --trade-mode "${TRADE_MODE}" \
  --short-entry-threshold "${SHORT_ENTRY_THRESHOLD}" \
  --short-exit-threshold "${SHORT_EXIT_THRESHOLD}" \
  --fee "${FEE}" \
  --starting-cash "${VIS_STARTING_CASH}" \
  --baseline-ma-window "${VIS_BASELINE_MA_WINDOW}" \
  --max-browser-points "${VIS_MAX_BROWSER_POINTS}" \
  --title "$title" \
  --nav-home-url "/${REPORTS_INDEX_URL_PATH}" \
  --nav-model-url "/${model_url}" \
  --nav-sim-url "/${sim_url}"
echo "$label: $output"
