#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON:-./.venv/bin/python}"
MAKE_BIN="${MAKE:-make}"

mkdir -p live_sim/state
if [[ -f "${LIVE_MODEL_SOURCE}" ]]; then
  cp "${LIVE_MODEL_SOURCE}" live_sim/state/model.npz
elif [[ -f live_sim/state/model.npz ]]; then
  echo "Selected live model does not exist yet: ${LIVE_MODEL_SOURCE}"
  echo "Keeping existing live_sim/state/model.npz until retraining activates a fresh model."
else
  echo "Missing selected live model: ${LIVE_MODEL_SOURCE}"
  echo "Run make train or make experiment first, or keep an existing live_sim/state/model.npz."
  exit 1
fi

write_kv() { printf '%s=%s\n' "$1" "${2-}"; }
blank() { printf '\n'; }

{
  write_kv SYMBOL "${SYMBOL}"
  write_kv INTERVAL "${INTERVAL}"
  write_kv BINANCE_BASE_URL "https://api.binance.com"
  blank
  write_kv STARTING_CASH "${LIVE_STARTING_CASH}"
  write_kv FEE "${FEE}"
  write_kv SLIPPAGE "${LIVE_SLIPPAGE}"
  write_kv ENTRY_THRESHOLD "${THRESHOLD}"
  write_kv EXIT_THRESHOLD "${EXIT_THRESHOLD}"
  write_kv TRADE_MODE "${TRADE_MODE}"
  write_kv SHORT_ENTRY_THRESHOLD "${SHORT_ENTRY_THRESHOLD}"
  write_kv SHORT_EXIT_THRESHOLD "${SHORT_EXIT_THRESHOLD}"
  write_kv STOP_LOSS "${STOP_LOSS}"
  write_kv TAKE_PROFIT "${TAKE_PROFIT}"
  write_kv MAX_HOLD_BARS "${MAX_HOLD_BARS}"
  write_kv MAX_INVEST "${LIVE_MAX_INVEST}"
  write_kv MAX_SHORT_INVEST "${LIVE_MAX_SHORT_INVEST}"
  write_kv ALLOW_FLIP_POSITION "${ALLOW_FLIP_POSITION}"
  write_kv BORROW_FEE "${BORROW_FEE}"
  write_kv LEVERAGE "${LEVERAGE}"
  write_kv LIQUIDATION_SIMULATION "${LIQUIDATION_SIMULATION}"
  write_kv MIN_INVEST "${LIVE_MIN_INVEST}"
  write_kv CONFIDENCE_MULTIPLIER "${LIVE_CONFIDENCE_MULTIPLIER}"
  blank
  write_kv EXECUTION_MODE "${EXECUTION_MODE}"
  write_kv REAL_TRADING_ENABLED "${REAL_TRADING_ENABLED}"
  write_kv REAL_REQUIRE_MANUAL_ARM "${REAL_REQUIRE_MANUAL_ARM}"
  write_kv REAL_QUICK_ARM_ENABLED "${REAL_QUICK_ARM_ENABLED}"
  write_kv REAL_MAX_TOTAL_USD "${REAL_MAX_TOTAL_USD}"
  write_kv REAL_MAX_ORDER_USD "${REAL_MAX_ORDER_USD}"
  write_kv REAL_MIN_ORDER_USD "${REAL_MIN_ORDER_USD}"
  write_kv REAL_PORTFOLIO_MODE "${REAL_PORTFOLIO_MODE}"
  write_kv REAL_CASH_ASSET "${REAL_CASH_ASSET}"
  write_kv REAL_BASE_ASSET "${REAL_BASE_ASSET}"
  write_kv COINBASE_PRODUCT_ID "${COINBASE_PRODUCT_ID}"
  write_kv COINBASE_API_KEY "${COINBASE_API_KEY-}"
  write_kv COINBASE_API_SECRET "${COINBASE_API_SECRET-}"
  write_kv COINBASE_TIMEOUT "${COINBASE_TIMEOUT}"
  write_kv SOLANA_RPC_URL "${SOLANA_RPC_URL-}"
  write_kv SOLANA_KEYPAIR_PATH "${SOLANA_KEYPAIR_PATH}"
  write_kv SOL_RESERVED_FOR_GAS "${SOL_RESERVED_FOR_GAS}"
  write_kv SOLANA_RPC_TIMEOUT "${SOLANA_RPC_TIMEOUT}"
  write_kv SOLANA_CONFIRM_POLLS "${SOLANA_CONFIRM_POLLS}"
  write_kv SOLANA_CONFIRM_DELAY_SECONDS "${SOLANA_CONFIRM_DELAY_SECONDS}"
  write_kv JUPITER_BASE_URL "${JUPITER_BASE_URL}"
  write_kv JUPITER_PRODUCT_ID "${JUPITER_PRODUCT_ID}"
  write_kv JUPITER_SLIPPAGE_BPS "${JUPITER_SLIPPAGE_BPS}"
  write_kv JUPITER_PRIORITY_FEE_LAMPORTS "${JUPITER_PRIORITY_FEE_LAMPORTS}"
  write_kv JUPITER_TIMEOUT "${JUPITER_TIMEOUT}"
  write_kv REAL_ARM_TOKEN "${REAL_ARM_TOKEN-}"
  write_kv REAL_ORDER_STATUS_POLLS "${REAL_ORDER_STATUS_POLLS}"
  write_kv REAL_ORDER_STATUS_DELAY_SECONDS "${REAL_ORDER_STATUS_DELAY_SECONDS}"
  blank
  write_kv MODEL_PATH "/app/state/model.npz"
  write_kv DB_PATH "/app/state/live_sim.db"
  write_kv RESET_ON_START "false"
  write_kv ALLOW_RESET_API "false"
  write_kv POLL_ON_START "true"
  write_kv POLL_DELAY_SECONDS "8"
  write_kv KLINE_LIMIT_BUFFER "8"
  write_kv CATCHUP_ENABLED "${LIVE_CATCHUP_ENABLED}"
  write_kv CATCHUP_SPREAD_PCT "${LIVE_CATCHUP_SPREAD_PCT}"
  write_kv CATCHUP_MAX_BARS "${LIVE_CATCHUP_MAX_BARS}"
  write_kv CATCHUP_RETRY_SECONDS "${LIVE_CATCHUP_RETRY_SECONDS}"
  blank
  write_kv RETRAIN_ENABLED "true"
  write_kv RETRAIN_TIME_UTC "04:00"
  write_kv RETRAIN_FREQUENCY "${LIVE_RETRAIN_FREQUENCY}"
  write_kv RETRAIN_LOOKBACK_DAYS "${LIVE_RETRAIN_LOOKBACK_DAYS}"
  write_kv RETRAIN_TRAIN_START "${LIVE_RETRAIN_TRAIN_START}"
  write_kv RETRAIN_TRAIN_END "${LIVE_RETRAIN_TRAIN_END}"
  write_kv RETRAIN_ON_START "false"
  write_kv RETRAIN_KEEP_RUNS "10"
  write_kv RETRAIN_CACHE_DIR "${LIVE_RETRAIN_CACHE_DIR}"
  write_kv TRAINING_RUNS_DIR "/app/state/training_runs"
  blank
  write_kv TRAIN_MODEL_TYPE "${LIVE_TRAIN_MODEL_TYPE}"
  write_kv TRAIN_BACKEND "${LIVE_TRAIN_BACKEND}"
  write_kv TRAIN_DEVICE "${LIVE_TRAIN_DEVICE}"
  write_kv TRAIN_LOOKBACK "${LIVE_TRAIN_LOOKBACK}"
  write_kv TRAIN_SEQUENCE_FEATURE_SET "${LIVE_TRAIN_SEQUENCE_FEATURE_SET}"
  write_kv TRAIN_EDGE "${LIVE_TRAIN_EDGE}"
  write_kv TRAIN_SPLIT "${SPLIT}"
  write_kv TRAIN_USE_FULL_WINDOW "${LIVE_TRAIN_USE_FULL_WINDOW}"
  write_kv TRAIN_CNN_FILTERS "${NN_CNN_FILTERS}"
  write_kv TRAIN_CNN_KERNEL_SIZES "${NN_CNN_KERNEL_SIZES}"
  write_kv TRAIN_LSTM_HIDDEN_SIZE "${NN_LSTM_HIDDEN_SIZE}"
  write_kv TRAIN_LSTM_LAYERS "${NN_LSTM_LAYERS}"
  write_kv TRAIN_LSTM_DROPOUT "${NN_LSTM_DROPOUT}"
  write_kv TRAIN_GRU_HIDDEN_SIZE "${NN_GRU_HIDDEN_SIZE}"
  write_kv TRAIN_GRU_LAYERS "${NN_GRU_LAYERS}"
  write_kv TRAIN_GRU_DROPOUT "${NN_GRU_DROPOUT}"
  write_kv TRAIN_TRANSFORMER_D_MODEL "${NN_TRANSFORMER_D_MODEL}"
  write_kv TRAIN_TRANSFORMER_HEADS "${NN_TRANSFORMER_HEADS}"
  write_kv TRAIN_TRANSFORMER_LAYERS "${NN_TRANSFORMER_LAYERS}"
  write_kv TRAIN_TRANSFORMER_FF_DIM "${NN_TRANSFORMER_FF_DIM}"
  write_kv TRAIN_TRANSFORMER_DROPOUT "${NN_TRANSFORMER_DROPOUT}"
  write_kv TRAIN_HIDDEN_LAYERS "${NN_HIDDEN_LAYERS}"
  write_kv TRAIN_LR "${NN_LR}"
  write_kv TRAIN_EPOCHS "${NN_EPOCHS}"
  write_kv TRAIN_BATCH_SIZE "${NN_BATCH_SIZE}"
  write_kv TRAIN_L2 "${NN_L2}"
  write_kv TRAIN_DECISION_THRESHOLD "${THRESHOLD}"
  write_kv TRAIN_THRESHOLD_GRID "${THRESHOLD_GRID}"
  write_kv TRAIN_OPTIMIZE_METRIC "${OPTIMIZE_METRIC}"
  write_kv TRAIN_CLASS_WEIGHT_MODE "${NN_CLASS_WEIGHT_MODE}"
  write_kv TRAIN_SEED "${NN_SEED}"
  blank
  write_kv HOME "/app/state"
  write_kv USER "candscout"
  write_kv LOGNAME "candscout"
  write_kv XDG_CACHE_HOME "/app/state/.cache"
  write_kv TORCH_HOME "/app/state/.cache/torch"
  write_kv TORCHINDUCTOR_CACHE_DIR "/app/state/.cache/torchinductor"
  write_kv TRITON_CACHE_DIR "/app/state/.cache/triton"
  write_kv TORCHDYNAMO_DISABLE "1"
  blank
  write_kv HOST "0.0.0.0"
  write_kv PORT "8080"
  write_kv HOST_PORT "${LIVE_HOST_PORT}"
  write_kv HOST_UID "${HOST_UID}"
  write_kv HOST_GID "${HOST_GID}"
} > live_sim/.env

env_file_args=()
for env_file in ${ENV_FILES:-}; do
  env_file_args+=(--env-file "$env_file")
done

"$PYTHON_BIN" src/save_live_env_snapshot.py \
  --source-env live_sim/.env \
  --active-env "${LIVE_ENV_ACTIVE}" \
  --model-env "${LIVE_MODEL_ENV}" \
  --snapshot-root "${LIVE_ENV_SNAPSHOT_ROOT}" \
  --model-type "${LIVE_MODEL_TYPE}" \
  --data-source "${DATA_SOURCE}" \
  --symbol "${SYMBOL}" \
  --interval "${INTERVAL}" \
  --sim-report "${LIVE_SIM_REPORT}" \
  "${env_file_args[@]}" \
  --param ASSET_ENV="${ASSET_ENV}" \
  --param TRAINER_ENV="${TRAINER_ENV}" \
  --param SYMBOL="${SYMBOL}" \
  --param DATA_SOURCE="${DATA_SOURCE}" \
  --param INTERVAL="${INTERVAL}" \
  --param START="${START}" \
  --param END="${END}" \
  --param SPLIT="${SPLIT}" \
  --param EDGE="${EDGE}" \
  --param FEE="${FEE}" \
  --param THRESHOLD="${THRESHOLD}" \
  --param EXIT_THRESHOLD="${EXIT_THRESHOLD}" \
  --param MAX_HOLD_BARS="${MAX_HOLD_BARS}" \
  --param STOP_LOSS="${STOP_LOSS}" \
  --param TAKE_PROFIT="${TAKE_PROFIT}" \
  --param LIVE_MODEL_TYPE="${LIVE_MODEL_TYPE}" \
  --param LIVE_MODEL_SOURCE="${LIVE_MODEL_SOURCE}" \
  --param LIVE_RETRAIN_FREQUENCY="${LIVE_RETRAIN_FREQUENCY}" \
  --param LIVE_RETRAIN_TRAIN_START="${LIVE_RETRAIN_TRAIN_START}" \
  --param LIVE_RETRAIN_TRAIN_END="${LIVE_RETRAIN_TRAIN_END}" \
  --param LIVE_RETRAIN_LOOKBACK_DAYS="${LIVE_RETRAIN_LOOKBACK_DAYS}" \
  --param LIVE_TRAIN_MODEL_TYPE="${LIVE_TRAIN_MODEL_TYPE}" \
  --param LIVE_TRAIN_BACKEND="${LIVE_TRAIN_BACKEND}" \
  --param LIVE_TRAIN_DEVICE="${LIVE_TRAIN_DEVICE}" \
  --param LIVE_TRAIN_LOOKBACK="${LIVE_TRAIN_LOOKBACK}" \
  --param LIVE_TRAIN_SEQUENCE_FEATURE_SET="${LIVE_TRAIN_SEQUENCE_FEATURE_SET}" \
  --param LIVE_TRAIN_EDGE="${LIVE_TRAIN_EDGE}" \
  --param LIVE_TRAIN_USE_FULL_WINDOW="${LIVE_TRAIN_USE_FULL_WINDOW}" \
  --param LIVE_STARTING_CASH="${LIVE_STARTING_CASH}" \
  --param LIVE_MIN_INVEST="${LIVE_MIN_INVEST}" \
  --param LIVE_MAX_INVEST="${LIVE_MAX_INVEST}" \
  --param LIVE_CONFIDENCE_MULTIPLIER="${LIVE_CONFIDENCE_MULTIPLIER}" \
  --param LIVE_SLIPPAGE="${LIVE_SLIPPAGE}" \
  --param LIVE_CATCHUP_ENABLED="${LIVE_CATCHUP_ENABLED}" \
  --param LIVE_CATCHUP_SPREAD_PCT="${LIVE_CATCHUP_SPREAD_PCT}" \
  --param LIVE_CATCHUP_MAX_BARS="${LIVE_CATCHUP_MAX_BARS}" \
  --param LIVE_CATCHUP_RETRY_SECONDS="${LIVE_CATCHUP_RETRY_SECONDS}" \
  --param EXECUTION_MODE="${EXECUTION_MODE}" \
  --param REAL_TRADING_ENABLED="${REAL_TRADING_ENABLED}" \
  --param REAL_MAX_TOTAL_USD="${REAL_MAX_TOTAL_USD}" \
  --param REAL_MAX_ORDER_USD="${REAL_MAX_ORDER_USD}" \
  --param REAL_PORTFOLIO_MODE="${REAL_PORTFOLIO_MODE}" \
  --param REAL_CASH_ASSET="${REAL_CASH_ASSET}" \
  --param REAL_BASE_ASSET="${REAL_BASE_ASSET}" \
  --param COINBASE_PRODUCT_ID="${COINBASE_PRODUCT_ID}" \
  --param SOLANA_KEYPAIR_PATH="${SOLANA_KEYPAIR_PATH}" \
  --param SOL_RESERVED_FOR_GAS="${SOL_RESERVED_FOR_GAS}" \
  --param JUPITER_PRODUCT_ID="${JUPITER_PRODUCT_ID}" \
  --param JUPITER_SLIPPAGE_BPS="${JUPITER_SLIPPAGE_BPS}" \
  --param NN_MODEL_TYPE="${NN_MODEL_TYPE}" \
  --param NN_BACKEND="${NN_BACKEND}" \
  --param NN_DEVICE="${NN_DEVICE}" \
  --param NN_LOOKBACK="${NN_LOOKBACK}" \
  --param NN_SEQUENCE_FEATURE_SET="${NN_SEQUENCE_FEATURE_SET}" \
  --param NN_CNN_FILTERS="${NN_CNN_FILTERS}" \
  --param NN_CNN_KERNEL_SIZES="${NN_CNN_KERNEL_SIZES}" \
  --param NN_LSTM_HIDDEN_SIZE="${NN_LSTM_HIDDEN_SIZE}" \
  --param NN_LSTM_LAYERS="${NN_LSTM_LAYERS}" \
  --param NN_LSTM_DROPOUT="${NN_LSTM_DROPOUT}" \
  --param NN_GRU_HIDDEN_SIZE="${NN_GRU_HIDDEN_SIZE}" \
  --param NN_GRU_LAYERS="${NN_GRU_LAYERS}" \
  --param NN_GRU_DROPOUT="${NN_GRU_DROPOUT}" \
  --param NN_TRANSFORMER_D_MODEL="${NN_TRANSFORMER_D_MODEL}" \
  --param NN_TRANSFORMER_HEADS="${NN_TRANSFORMER_HEADS}" \
  --param NN_TRANSFORMER_LAYERS="${NN_TRANSFORMER_LAYERS}" \
  --param NN_TRANSFORMER_FF_DIM="${NN_TRANSFORMER_FF_DIM}" \
  --param NN_TRANSFORMER_DROPOUT="${NN_TRANSFORMER_DROPOUT}" \
  --param NN_HIDDEN_LAYERS="${NN_HIDDEN_LAYERS}" \
  --param NN_LR="${NN_LR}" \
  --param NN_EPOCHS="${NN_EPOCHS}" \
  --param NN_BATCH_SIZE="${NN_BATCH_SIZE}" \
  --param NN_L2="${NN_L2}" \
  --param NN_CLASS_WEIGHT_MODE="${NN_CLASS_WEIGHT_MODE}" \
  --param NN_SEED="${NN_SEED}" \
  --param NN_SIM_REPORT="${NN_SIM_REPORT}"

echo "Synced live_sim from main config:"
echo "  model:    requested ${LIVE_MODEL_SOURCE}; active copy live_sim/state/model.npz"
echo "  env:      ${LIVE_MODEL_ENV} -> ${LIVE_ENV_ACTIVE} -> live_sim/.env"
echo "  snapshot: ${LIVE_ENV_SNAPSHOT_ROOT}"
echo "  symbol:   ${SYMBOL}"
echo "  interval: ${INTERVAL}"
echo "  catch-up: enabled=${LIVE_CATCHUP_ENABLED}, historical spread=${LIVE_CATCHUP_SPREAD_PCT}, max bars=${LIVE_CATCHUP_MAX_BARS} (0=unlimited)"
echo "  real:     execution=${EXECUTION_MODE}, enabled=${REAL_TRADING_ENABLED}, mode=${REAL_PORTFOLIO_MODE}, coinbase=${COINBASE_PRODUCT_ID}, jupiter=${JUPITER_PRODUCT_ID}, cash=${REAL_CASH_ASSET}, base=${REAL_BASE_ASSET}"
echo "  retrain:  every ${LIVE_RETRAIN_FREQUENCY}, window ${LIVE_RETRAIN_TRAIN_START} to ${LIVE_RETRAIN_TRAIN_END} as rolling duration"
echo "  train:    ${LIVE_TRAIN_MODEL_TYPE}, backend=${LIVE_TRAIN_BACKEND}, device=${LIVE_TRAIN_DEVICE}, lookback=${LIVE_TRAIN_LOOKBACK}, feature_set=${LIVE_TRAIN_SEQUENCE_FEATURE_SET}, filters=${NN_CNN_FILTERS}, gru_hidden=${NN_GRU_HIDDEN_SIZE}, lstm_hidden=${NN_LSTM_HIDDEN_SIZE}, transformer=${NN_TRANSFORMER_D_MODEL}x${NN_TRANSFORMER_LAYERS}/h${NN_TRANSFORMER_HEADS}, hidden=${NN_HIDDEN_LAYERS}"
