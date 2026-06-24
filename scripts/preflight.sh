#!/usr/bin/env bash
set -euo pipefail
kind="${1:?usage: preflight.sh nn|lr|xgb}"
MAKE_BIN="${MAKE:-make}"
DATA_SOURCE="${DATA_SOURCE:-binance}"
SYMBOL="${SYMBOL:-SOLUSDT}"
RAW_PREFLIGHT="data/downloads/${DATA_SOURCE}/${SYMBOL}/5m/preflight_candles.parquet"

"$MAKE_BIN" download \
  INTERVAL=5m \
  START=2026-01-10T00:00:00Z \
  END=2026-01-13T00:00:00Z \
  RAW_DATA="$RAW_PREFLIGHT"

case "$kind" in
  nn)
    "$MAKE_BIN" experiment \
      INTERVAL=5m \
      START=2026-01-10T00:00:00Z \
      END=2026-01-13T00:00:00Z \
      RAW_DATA="$RAW_PREFLIGHT" \
      NN_MODEL_OUT="models/nn/cnn/${DATA_SOURCE}/${SYMBOL}/5m/preflight_model.npz" \
      NN_TRAIN_METRICS="models/nn/cnn/${DATA_SOURCE}/${SYMBOL}/5m/preflight_train_metrics.json" \
      NN_BACKTEST_REPORT="models/nn/cnn/${DATA_SOURCE}/${SYMBOL}/5m/preflight_backtest_report.json" \
      NN_PREDICTIONS_OUT="data/reports/nn/cnn/${DATA_SOURCE}/${SYMBOL}/5m/preflight_predictions.parquet" \
      NN_MODEL_TYPE=cnn \
      NN_LOOKBACK=20 \
      NN_CNN_FILTERS=4,8 \
      NN_CNN_KERNEL_SIZES=3,3 \
      NN_EPOCHS=2 \
      NN_HIDDEN_LAYERS=8 \
      NN_BATCH_SIZE=128
    ;;
  lr)
    "$MAKE_BIN" lr-experiment \
      INTERVAL=5m \
      START=2026-01-10T00:00:00Z \
      END=2026-01-13T00:00:00Z \
      RAW_DATA="$RAW_PREFLIGHT" \
      FEATURES_DATA="data/features/lr/${DATA_SOURCE}/${SYMBOL}/5m/preflight_features.parquet" \
      FEATURES_META="data/features/lr/${DATA_SOURCE}/${SYMBOL}/5m/preflight_features.meta.json" \
      MODEL_OUT="models/lr/${DATA_SOURCE}/${SYMBOL}/5m/preflight_logreg.npz" \
      TRAIN_METRICS="models/lr/${DATA_SOURCE}/${SYMBOL}/5m/preflight_train_metrics.json" \
      BACKTEST_REPORT="models/lr/${DATA_SOURCE}/${SYMBOL}/5m/preflight_backtest_report.json" \
      PREDICTIONS_OUT="data/reports/lr/${DATA_SOURCE}/${SYMBOL}/5m/preflight_predictions.parquet"
    ;;
  xgb)
    "$MAKE_BIN" xgb-experiment \
      INTERVAL=5m \
      START=2026-01-10T00:00:00Z \
      END=2026-01-13T00:00:00Z \
      RAW_DATA="$RAW_PREFLIGHT" \
      XGB_FEATURES_DATA="data/features/xgb/${DATA_SOURCE}/${SYMBOL}/5m/preflight_features.parquet" \
      XGB_FEATURES_META="data/features/xgb/${DATA_SOURCE}/${SYMBOL}/5m/preflight_features.meta.json" \
      XGB_MODEL_OUT="models/xgb/${DATA_SOURCE}/${SYMBOL}/5m/preflight_model.json" \
      XGB_TRAIN_METRICS="models/xgb/${DATA_SOURCE}/${SYMBOL}/5m/preflight_train_metrics.json" \
      XGB_BACKTEST_REPORT="models/xgb/${DATA_SOURCE}/${SYMBOL}/5m/preflight_backtest_report.json" \
      XGB_PREDICTIONS_OUT="data/reports/xgb/${DATA_SOURCE}/${SYMBOL}/5m/preflight_predictions.parquet" \
      XGB_N_ESTIMATORS=20 \
      XGB_MAX_DEPTH=2 \
      XGB_N_JOBS=2 \
      XGB_DEVICE=cuda
    ;;
  *)
    echo "Unknown preflight kind: $kind" >&2
    exit 2
    ;;
esac
