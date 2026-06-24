#!/usr/bin/env bash
set -euo pipefail
kind="${1:?usage: archive_run.sh nn|lr|xgb|strategy|nn-save}"
PYTHON_BIN="${PYTHON:-./.venv/bin/python}"

archive_args=()
for f in ${ENV_FILES:-}; do
  archive_args+=(--env-file "$f")
done

add_param() {
  archive_args+=(--param "$1=${2:-}")
}

add_common_params() {
  add_param ASSET_ENV "${ASSET_ENV:-}"
  add_param TRAINER_ENV "$1"
  add_param SYMBOL "${SYMBOL:-}"
  add_param DATA_SOURCE "${DATA_SOURCE:-}"
  add_param INTERVAL "${INTERVAL:-}"
  add_param START "${START:-}"
  add_param END "${END:-}"
  add_param SPLIT "${SPLIT:-}"
  add_param EDGE "${EDGE:-}"
  add_param SHORT_EDGE "${SHORT_EDGE:-}"
  add_param FEE "${FEE:-}"
  add_param THRESHOLD "${THRESHOLD:-}"
  add_param TRADE_MODE "${TRADE_MODE:-}"
  add_param SHORT_ENTRY_THRESHOLD "${SHORT_ENTRY_THRESHOLD:-}"
  add_param SHORT_EXIT_THRESHOLD "${SHORT_EXIT_THRESHOLD:-}"
  add_param SIM_STARTING_CASH "${SIM_STARTING_CASH:-}"
  add_param SIM_MIN_INVEST "${SIM_MIN_INVEST:-}"
  add_param SIM_MAX_INVEST "${SIM_MAX_INVEST:-}"
  add_param SIM_MAX_SHORT_INVEST "${SIM_MAX_SHORT_INVEST:-}"
  add_param SIM_CONFIDENCE_MULTIPLIER "${SIM_CONFIDENCE_MULTIPLIER:-}"
  add_param SIM_SHORT_CONFIDENCE_MULTIPLIER "${SIM_SHORT_CONFIDENCE_MULTIPLIER:-}"
  add_param SIM_POSITION_MODE "${SIM_POSITION_MODE:-}"
  add_param SIM_SLIPPAGE "${SIM_SLIPPAGE:-}"
  add_param SIM_SPREAD_PCT "${SIM_SPREAD_PCT:-}"
  add_param POSITION_MODE "${POSITION_MODE:-}"
  add_param EXIT_THRESHOLD "${EXIT_THRESHOLD:-}"
  add_param MAX_HOLD_BARS "${MAX_HOLD_BARS:-}"
  add_param STOP_LOSS "${STOP_LOSS:-}"
  add_param TAKE_PROFIT "${TAKE_PROFIT:-}"
  add_param DECISION_THRESHOLD "${DECISION_THRESHOLD:-}"
  add_param THRESHOLD_GRID "${THRESHOLD_GRID:-}"
  add_param OPTIMIZE_METRIC "${OPTIMIZE_METRIC:-}"
  add_param RAW_DATA "${RAW_DATA:-}"
}

add_nn_params() {
  add_param NN_BACKEND "${NN_BACKEND:-}"
  add_param NN_DEVICE "${NN_DEVICE:-}"
  add_param NN_MODEL_TYPE "${NN_MODEL_TYPE:-}"
  add_param NN_LOOKBACK "${NN_LOOKBACK:-}"
  add_param NN_SEQUENCE_FEATURE_SET "${NN_SEQUENCE_FEATURE_SET:-}"
  add_param NN_CNN_FILTERS "${NN_CNN_FILTERS:-}"
  add_param NN_CNN_KERNEL_SIZES "${NN_CNN_KERNEL_SIZES:-}"
  add_param NN_LSTM_HIDDEN_SIZE "${NN_LSTM_HIDDEN_SIZE:-}"
  add_param NN_LSTM_LAYERS "${NN_LSTM_LAYERS:-}"
  add_param NN_LSTM_DROPOUT "${NN_LSTM_DROPOUT:-}"
  add_param NN_GRU_HIDDEN_SIZE "${NN_GRU_HIDDEN_SIZE:-}"
  add_param NN_GRU_LAYERS "${NN_GRU_LAYERS:-}"
  add_param NN_GRU_DROPOUT "${NN_GRU_DROPOUT:-}"
  add_param NN_TRANSFORMER_D_MODEL "${NN_TRANSFORMER_D_MODEL:-}"
  add_param NN_TRANSFORMER_HEADS "${NN_TRANSFORMER_HEADS:-}"
  add_param NN_TRANSFORMER_LAYERS "${NN_TRANSFORMER_LAYERS:-}"
  add_param NN_TRANSFORMER_FF_DIM "${NN_TRANSFORMER_FF_DIM:-}"
  add_param NN_TRANSFORMER_DROPOUT "${NN_TRANSFORMER_DROPOUT:-}"
  add_param NN_HIDDEN_LAYERS "${NN_HIDDEN_LAYERS:-}"
  add_param NN_LR "${NN_LR:-}"
  add_param NN_EPOCHS "${NN_EPOCHS:-}"
  add_param NN_BATCH_SIZE "${NN_BATCH_SIZE:-}"
  add_param NN_L2 "${NN_L2:-}"
  add_param NN_CLASS_WEIGHT_MODE "${NN_CLASS_WEIGHT_MODE:-}"
  add_param NN_SEED "${NN_SEED:-}"
  add_param NN_MODEL_OUT "${NN_MODEL_OUT:-}"
  add_param NN_TRAIN_METRICS "${NN_TRAIN_METRICS:-}"
  add_param NN_BACKTEST_REPORT "${NN_BACKTEST_REPORT:-}"
  add_param NN_PREDICTIONS_OUT "${NN_PREDICTIONS_OUT:-}"
  add_param NN_SIM_REPORT "${NN_SIM_REPORT:-}"
  add_param NN_SIM_TRADES "${NN_SIM_TRADES:-}"
}

case "$kind" in
  nn)
    add_common_params "${TRAINER_ENV:-}"
    add_nn_params
    "$PYTHON_BIN" src/archive_model.py \
      --run-store "${RUNS_ROOT}" \
      --current-root "${CURRENT_ROOT}" \
      --write-current \
      --family nn \
      --model-type "${NN_MODEL_TYPE}" \
      --backend "${NN_BACKEND}" \
      --asset-env "${ASSET_ENV}" \
      --trainer-env "${TRAINER_ENV}" \
      --model "${NN_MODEL_OUT}" \
      --train-metrics "${NN_TRAIN_METRICS}" \
      --backtest-report "${NN_BACKTEST_REPORT}" \
      --predictions "${NN_PREDICTIONS_OUT}" \
      "${archive_args[@]}"
    ;;
  lr)
    [[ -n "${LR_TRAINER_ENV:-}" ]] && archive_args+=(--env-file "${LR_TRAINER_ENV}")
    add_common_params "${LR_TRAINER_ENV:-}"
    add_param LR "${LR:-}"
    add_param EPOCHS "${EPOCHS:-}"
    add_param L2 "${L2:-}"
    add_param CLASS_WEIGHT_MODE "${CLASS_WEIGHT_MODE:-}"
    add_param FEATURES_DATA "${FEATURES_DATA:-}"
    add_param FEATURES_META "${FEATURES_META:-}"
    add_param MODEL_OUT "${MODEL_OUT:-}"
    add_param TRAIN_METRICS "${TRAIN_METRICS:-}"
    add_param BACKTEST_REPORT "${BACKTEST_REPORT:-}"
    add_param PREDICTIONS_OUT "${PREDICTIONS_OUT:-}"
    "$PYTHON_BIN" src/archive_model.py \
      --run-store "${RUNS_ROOT}" \
      --current-root "${CURRENT_ROOT}" \
      --write-current \
      --family lr \
      --model-type logreg \
      --backend numpy \
      --asset-env "${ASSET_ENV}" \
      --trainer-env "${LR_TRAINER_ENV}" \
      --model "${MODEL_OUT}" \
      --train-metrics "${TRAIN_METRICS}" \
      --backtest-report "${BACKTEST_REPORT}" \
      --predictions "${PREDICTIONS_OUT}" \
      "${archive_args[@]}"
    ;;
  xgb)
    add_common_params "${XGB_TRAINER_ENV:-}"
    add_param XGB_N_ESTIMATORS "${XGB_N_ESTIMATORS:-}"
    add_param XGB_MAX_DEPTH "${XGB_MAX_DEPTH:-}"
    add_param XGB_LEARNING_RATE "${XGB_LEARNING_RATE:-}"
    add_param XGB_SUBSAMPLE "${XGB_SUBSAMPLE:-}"
    add_param XGB_COLSAMPLE_BYTREE "${XGB_COLSAMPLE_BYTREE:-}"
    add_param XGB_MIN_CHILD_WEIGHT "${XGB_MIN_CHILD_WEIGHT:-}"
    add_param XGB_REG_LAMBDA "${XGB_REG_LAMBDA:-}"
    add_param XGB_REG_ALPHA "${XGB_REG_ALPHA:-}"
    add_param XGB_GAMMA "${XGB_GAMMA:-}"
    add_param XGB_TREE_METHOD "${XGB_TREE_METHOD:-}"
    add_param XGB_DEVICE "${XGB_DEVICE:-}"
    add_param XGB_N_JOBS "${XGB_N_JOBS:-}"
    add_param XGB_CLASS_WEIGHT_MODE "${XGB_CLASS_WEIGHT_MODE:-}"
    add_param XGB_POS_WEIGHT "${XGB_POS_WEIGHT:-}"
    add_param XGB_SEED "${XGB_SEED:-}"
    add_param XGB_FEATURES_DATA "${XGB_FEATURES_DATA:-}"
    add_param XGB_FEATURES_META "${XGB_FEATURES_META:-}"
    add_param XGB_MODEL_OUT "${XGB_MODEL_OUT:-}"
    add_param XGB_TRAIN_METRICS "${XGB_TRAIN_METRICS:-}"
    add_param XGB_BACKTEST_REPORT "${XGB_BACKTEST_REPORT:-}"
    add_param XGB_PREDICTIONS_OUT "${XGB_PREDICTIONS_OUT:-}"
    add_param XGB_SIM_DEFAULT_TEST_FRACTION "${XGB_SIM_DEFAULT_TEST_FRACTION:-}"
    add_param XGB_SIM_REPORT "${XGB_SIM_REPORT:-}"
    add_param XGB_SIM_TRADES "${XGB_SIM_TRADES:-}"
    "$PYTHON_BIN" src/archive_model.py \
      --run-store "${RUNS_ROOT}" \
      --current-root "${CURRENT_ROOT}" \
      --write-current \
      --family xgb \
      --model-type xgboost \
      --backend xgboost \
      --asset-env "${ASSET_ENV}" \
      --trainer-env "${XGB_TRAINER_ENV}" \
      --model "${XGB_MODEL_OUT}" \
      --train-metrics "${XGB_TRAIN_METRICS}" \
      --backtest-report "${XGB_BACKTEST_REPORT}" \
      --predictions "${XGB_PREDICTIONS_OUT}" \
      --sim-report "${XGB_SIM_REPORT}" \
      --sim-trades "${XGB_SIM_TRADES}" \
      --visualization "${XGB_VISUALIZATION_OUT}" \
      --sim-visualization "${XGB_SIM_VISUALIZATION_OUT}" \
      "${archive_args[@]}"
    ;;
  strategy)
    add_common_params "${STRATEGY_TRAINER_ENV:-}"
    add_param STRATEGY_MODEL_TYPE "${STRATEGY_MODEL_TYPE:-}"
    add_param STRATEGY_MA_WINDOW "${STRATEGY_MA_WINDOW:-}"
    add_param STRATEGY_MODEL_OUT "${STRATEGY_MODEL_OUT:-}"
    add_param STRATEGY_TRAIN_METRICS "${STRATEGY_TRAIN_METRICS:-}"
    add_param STRATEGY_BACKTEST_REPORT "${STRATEGY_BACKTEST_REPORT:-}"
    add_param STRATEGY_PREDICTIONS_OUT "${STRATEGY_PREDICTIONS_OUT:-}"
    add_param STRATEGY_SIM_REPORT "${STRATEGY_SIM_REPORT:-}"
    add_param STRATEGY_SIM_TRADES "${STRATEGY_SIM_TRADES:-}"
    "$PYTHON_BIN" src/archive_model.py \
      --run-store "${RUNS_ROOT}" \
      --current-root "${CURRENT_ROOT}" \
      --write-current \
      --family strategy \
      --model-type "${STRATEGY_MODEL_TYPE}" \
      --backend rule_based \
      --asset-env "${ASSET_ENV}" \
      --trainer-env "${STRATEGY_TRAINER_ENV}" \
      --model "${STRATEGY_MODEL_OUT}" \
      --train-metrics "${STRATEGY_TRAIN_METRICS}" \
      --backtest-report "${STRATEGY_BACKTEST_REPORT}" \
      --predictions "${STRATEGY_PREDICTIONS_OUT}" \
      --sim-report "${STRATEGY_SIM_REPORT}" \
      --sim-trades "${STRATEGY_SIM_TRADES}" \
      --visualization "${STRATEGY_VISUALIZATION_OUT}" \
      --sim-visualization "${STRATEGY_SIM_VISUALIZATION_OUT}" \
      "${archive_args[@]}"
    ;;
  nn-save)
    add_common_params "${TRAINER_ENV:-}"
    add_nn_params
    add_param SIM_START "${SIM_START:-}"
    add_param SIM_DURATION "${SIM_DURATION:-}"
    add_param SIM_DEFAULT_TEST_FRACTION "${SIM_DEFAULT_TEST_FRACTION:-}"
    name_args=()
    [[ -n "${NN_ARCHIVE_NAME:-}" ]] && name_args+=(--name "${NN_ARCHIVE_NAME}")
    "$PYTHON_BIN" src/archive_model.py \
      --archive-root "${NN_ARCHIVE_ROOT}" \
      "${name_args[@]}" \
      --model "${NN_MODEL_OUT}" \
      --train-metrics "${NN_TRAIN_METRICS}" \
      --backtest-report "${NN_BACKTEST_REPORT}" \
      --predictions "${NN_PREDICTIONS_OUT}" \
      --sim-report "${NN_SIM_REPORT}" \
      --sim-trades "${NN_SIM_TRADES}" \
      --visualization "${NN_VISUALIZATION_OUT}" \
      --sim-visualization "${NN_SIM_VISUALIZATION_OUT}" \
      "${archive_args[@]}" \
      --include-diff
    ;;
  *)
    echo "Unknown archive kind: $kind" >&2
    exit 2
    ;;
esac
