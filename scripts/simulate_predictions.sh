#!/usr/bin/env bash
set -euo pipefail
family="${1:?usage: simulate_predictions.sh nn|lr|xgb|strategy}"
PYTHON_BIN="${PYTHON:-./.venv/bin/python}"

case "$family" in
  nn)
    predictions="${NN_PREDICTIONS_OUT}"; report="${NN_SIM_REPORT}"; trades="${NN_SIM_TRADES}"; cmp_report="${NN_SIM_LONG_SHORT_REPORT}"; cmp_trades="${NN_SIM_LONG_SHORT_TRADES}"; label="Sequence NN daily bank"; default_fraction="${SIM_DEFAULT_TEST_FRACTION}" ;;
  lr)
    predictions="${PREDICTIONS_OUT}"; report="${SIM_REPORT}"; trades="${SIM_TRADES}"; cmp_report="${SIM_LONG_SHORT_REPORT}"; cmp_trades="${SIM_LONG_SHORT_TRADES}"; label="Daily bank"; default_fraction="${SIM_DEFAULT_TEST_FRACTION}" ;;
  xgb)
    predictions="${XGB_PREDICTIONS_OUT}"; report="${XGB_SIM_REPORT}"; trades="${XGB_SIM_TRADES}"; cmp_report="${XGB_SIM_LONG_SHORT_REPORT}"; cmp_trades="${XGB_SIM_LONG_SHORT_TRADES}"; label="XGBoost bank"; default_fraction="${XGB_SIM_DEFAULT_TEST_FRACTION}" ;;
  strategy)
    predictions="${STRATEGY_PREDICTIONS_OUT}"; report="${STRATEGY_SIM_REPORT}"; trades="${STRATEGY_SIM_TRADES}"; cmp_report="${STRATEGY_SIM_LONG_SHORT_REPORT}"; cmp_trades="${STRATEGY_SIM_LONG_SHORT_TRADES}"; label="Strategy bank"; default_fraction="${STRATEGY_SIM_DEFAULT_TEST_FRACTION:-${SIM_DEFAULT_TEST_FRACTION}}" ;;
  *) echo "Unknown simulation family: $family" >&2; exit 2 ;;
esac

window_args=()
[[ -n "${SIM_START:-}" ]] && window_args+=(--start "${SIM_START}")
[[ -n "${SIM_DURATION:-}" ]] && window_args+=(--duration "${SIM_DURATION}")
flip_arg="--no-allow-flip-position"
case "${ALLOW_FLIP_POSITION:-false}" in 1|true|yes) flip_arg="--allow-flip-position";; esac

"$PYTHON_BIN" src/daily_bank_sim.py \
  --predictions "$predictions" \
  "${window_args[@]}" \
  --default-test-fraction "$default_fraction" \
  --position-mode "${SIM_POSITION_MODE}" \
  --starting-cash "${SIM_STARTING_CASH}" \
  --min-invest "${SIM_MIN_INVEST}" \
  --max-invest "${SIM_MAX_INVEST}" \
  --max-short-invest "${SIM_MAX_SHORT_INVEST}" \
  --confidence-multiplier "${SIM_CONFIDENCE_MULTIPLIER}" \
  --short-confidence-multiplier "${SIM_SHORT_CONFIDENCE_MULTIPLIER}" \
  --trade-mode long_only \
  --comparison-trade-mode long_short \
  --threshold "${THRESHOLD}" \
  --exit-threshold "${EXIT_THRESHOLD}" \
  --short-entry-threshold "${SHORT_ENTRY_THRESHOLD}" \
  --short-exit-threshold "${SHORT_EXIT_THRESHOLD}" \
  "$flip_arg" \
  --borrow-fee "${BORROW_FEE}" \
  --leverage "${LEVERAGE}" \
  --liquidation-simulation "${LIQUIDATION_SIMULATION}" \
  --fee "${FEE}" \
  --max-hold-bars "${MAX_HOLD_BARS}" \
  --stop-loss "${STOP_LOSS}" \
  --take-profit "${TAKE_PROFIT}" \
  --slippage "${SIM_SLIPPAGE}" \
  --spread-pct "${SIM_SPREAD_PCT}" \
  --report-out "$report" \
  --trades-out "$trades" \
  --comparison-report-out "$cmp_report" \
  --comparison-trades-out "$cmp_trades"

echo "$label report: $report"
echo "$label trades: $trades"
[[ "$family" == "xgb" ]] && echo "XGBoost simulation default test fraction: $default_fraction"
