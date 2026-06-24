SHELL := /bin/bash
.EXPORT_ALL_VARIABLES:

# ---------- Environment defaults ----------
# Bootstrap runtime/local first so they can select ASSET_ENV and TRAINER_ENV.
-include env/runtime.env env/local.env
ASSET_ENV ?= env/assets/solusdt_3m.env
TRAINER_ENV ?= env/trainers/cnn_torch.env
XGB_TRAINER_ENV ?= env/trainers/xgboost.env
STRATEGY_TRAINER_ENV ?= env/trainers/strategy.env
ENV_FILES := \
	env/runtime.env \
	$(ASSET_ENV) \
	$(TRAINER_ENV) \
	$(XGB_TRAINER_ENV) \
	$(STRATEGY_TRAINER_ENV) \
	env/core.env \
	env/sequence_nn.env \
	env/logistic_regression.env \
	env/simulation.env \
	env/features.env \
	env/paths.env \
	env/local.env

-include $(ENV_FILES)

# ---------- Derived helpers ----------
RANDOM_STOCK_FLAG = $(if $(filter 1 true yes,$(RANDOM_STOCK)),--random-stock,)
SIM_WINDOW_ARGS = $(if $(SIM_START),--start $(SIM_START),) $(if $(SIM_DURATION),--duration $(SIM_DURATION),)
NN_VISUALIZATION_FILE ?= $(notdir $(NN_VISUALIZATION_OUT))
NN_VISUALIZATION_URL_PATH ?= $(patsubst data/reports/%,%,$(NN_VISUALIZATION_OUT))
VISUALIZATION_URL_PATH ?= $(patsubst data/reports/%,%,$(VISUALIZATION_OUT))
VISUALIZATION_FILE ?= $(notdir $(VISUALIZATION_OUT))
NN_SIM_VISUALIZATION_URL_PATH ?= $(patsubst data/reports/%,%,$(NN_SIM_VISUALIZATION_OUT))
SIM_VISUALIZATION_URL_PATH ?= $(patsubst data/reports/%,%,$(SIM_VISUALIZATION_OUT))
REPORTS_INDEX_URL_PATH ?= $(patsubst data/reports/%,%,$(REPORTS_INDEX_OUT))
XGB_VISUALIZATION_URL_PATH ?= $(patsubst data/reports/%,%,$(XGB_VISUALIZATION_OUT))
XGB_SIM_VISUALIZATION_URL_PATH ?= $(patsubst data/reports/%,%,$(XGB_SIM_VISUALIZATION_OUT))
XGB_SIM_DEFAULT_TEST_FRACTION ?= $(shell $(PYTHON) -c 's=float("$(SPLIT)"); print(f"{max(0.000001, min(0.999999, 1.0 - s)):.12g}")' 2>/dev/null || echo 0.05)
STRATEGY_VISUALIZATION_URL_PATH ?= $(patsubst data/reports/%,%,$(STRATEGY_VISUALIZATION_OUT))
STRATEGY_SIM_VISUALIZATION_URL_PATH ?= $(patsubst data/reports/%,%,$(STRATEGY_SIM_VISUALIZATION_OUT))
STRATEGY_SIM_DEFAULT_TEST_FRACTION ?= $(shell $(PYTHON) -c 's=float("$(SPLIT)"); print(f"{max(0.000001, min(0.999999, 1.0 - s)):.12g}")' 2>/dev/null || echo 0.05)
NN_ARCHIVE_ROOT ?= models/archive/nn/$(NN_MODEL_TYPE)/$(DATA_SOURCE)/$(SYMBOL)/$(INTERVAL)
NN_ARCHIVE_NAME ?=
NN_ARCHIVE_NAME_ARG = $(if $(NN_ARCHIVE_NAME),--name $(NN_ARCHIVE_NAME),)
XGB_POS_WEIGHT_ARG = $(if $(strip $(XGB_POS_WEIGHT)),--pos-weight $(XGB_POS_WEIGHT),)
LIVE_LOCAL_URL ?= http://127.0.0.1:$(LIVE_HOST_PORT)
LIVE_PUBLIC_URL ?= http://$(REPORTS_HOST):$(LIVE_HOST_PORT)

GENERATED_DIRS := $(sort $(dir \
	$(RAW_DATA) \
	$(DOWNLOAD_CACHE) \
	$(FEATURES_DATA) $(FEATURES_META) \
	$(MODEL_OUT) $(TRAIN_METRICS) $(BACKTEST_REPORT) $(PREDICTIONS_OUT) \
	$(NN_MODEL_OUT) $(NN_TRAIN_METRICS) $(NN_BACKTEST_REPORT) $(NN_PREDICTIONS_OUT) $(NN_VISUALIZATION_OUT) \
	$(XGB_FEATURES_DATA) $(XGB_FEATURES_META) $(XGB_MODEL_OUT) $(XGB_TRAIN_METRICS) $(XGB_BACKTEST_REPORT) $(XGB_PREDICTIONS_OUT) $(XGB_VISUALIZATION_OUT) $(XGB_SIM_REPORT) $(XGB_SIM_TRADES) $(XGB_SIM_LONG_SHORT_REPORT) $(XGB_SIM_LONG_SHORT_TRADES) $(XGB_SIM_VISUALIZATION_OUT) \
	$(STRATEGY_MODEL_OUT) $(STRATEGY_TRAIN_METRICS) $(STRATEGY_BACKTEST_REPORT) $(STRATEGY_PREDICTIONS_OUT) $(STRATEGY_VISUALIZATION_OUT) $(STRATEGY_SIM_REPORT) $(STRATEGY_SIM_TRADES) $(STRATEGY_SIM_LONG_SHORT_REPORT) $(STRATEGY_SIM_LONG_SHORT_TRADES) $(STRATEGY_SIM_VISUALIZATION_OUT) \
	$(DIAG_REPORT) $(DIAG_TABLE) $(DIAG_TEST_PREDICTIONS) $(SWEEP_OUTPUT_DIR)/ $(VISUALIZATION_OUT) \
	$(SIM_REPORT) $(SIM_TRADES) $(SIM_LONG_SHORT_REPORT) $(SIM_LONG_SHORT_TRADES) $(SIM_VISUALIZATION_OUT) $(NN_SIM_REPORT) $(NN_SIM_TRADES) $(NN_SIM_LONG_SHORT_REPORT) $(NN_SIM_LONG_SHORT_TRADES) $(NN_SIM_VISUALIZATION_OUT) $(REPORTS_INDEX_OUT) \
))


.PHONY: help install dirs download nn train backtest experiment diagnostic sweep visualize sim sim-visualize sim-graph reports-index start start-fg stop day-sim serve-reports serve-reports-lan graph serve-lan preflight run run-stock stock-run clean smoke repo-status
.PHONY: live-sync live-setup live-build live-build-sync live-up live-up-sync live-down live-logs live-reset live-test live-real-check live-retrain-status live-retrain-now live-clear-retrain-lock live-cache-status live-update-model live-update-model-sync update-model
.PHONY: github-check github-init github-commit github-create-private github-push github-publish
.PHONY: save-run lr-save-run list-runs show-current
.PHONY: nn-train nn-backtest nn-experiment nn-diagnostic nn-visualize nn-sim nn-sim-visualize nn-sim-graph nn-graph nn-serve-lan nn-servre-lan nn-save save-nn-model
.PHONY: lr-features lr-train lr-backtest lr-experiment lr-diagnostic lr-sweep lr-visualize lr-sim lr-sim-visualize lr-sim-graph lr-graph lr-serve-lan lr-preflight
.PHONY: xgb-features xgb-train xgb-backtest xgb-experiment xgb-visualize xgb-sim xgb-sim-visualize xgb-sim-graph xgb-graph xgb-save-run xgb-preflight gxboost-experiment gxboost-train gxboost-backtest gxboost-sim
.PHONY: strategy-train strategy-backtest strategy-experiment strategy-visualize strategy-sim strategy-sim-visualize strategy-sim-graph strategy-graph strategy-save-run

run:
	$(MAKE) download
	$(MAKE) experiment
	$(MAKE) graph

STOCK_INTERVAL ?= 1d
STOCK_SYMBOL ?=
STOCK_START ?=
STOCK_END ?= $(END)
STOCK_LOOKBACK_DAYS ?= 1095
STOCK_SPLIT ?= 0.95
STOCK_MIN_ROWS ?= 700
STOCK_MIN_HISTORY_YEARS ?= 3
STOCK_SIM_TEST_FRACTION ?= 0.05
STOCK_ARCHIVE_CURRENT ?= 1
STOCK_ASSET_ENV ?= $(if $(filter 1d,$(STOCK_INTERVAL)),env/assets/stock_yahoo_1d.env,$(if $(filter 1h,$(STOCK_INTERVAL)),env/assets/stock_yahoo_1h.env,$(ASSET_ENV)))

run-stock stock-run:
	@scripts/run_stock.sh

help:
	@echo "Targets:"
	@echo "  make install      - install Python dependencies into .venv"
	@echo "  make download     - download raw candles"
	@echo "  make train        - train sequence neural net"
	@echo "  make backtest     - run sequence neural net backtest + predictions parquet"
	@echo "  make experiment   - run sequence neural net train -> sequence backtest"
	@echo "  make visualize    - create sequence neural net HTML visualization"
	@echo "  make sim          - simulate bank-account trades from sequence neural net predictions"
	@echo "  make sim-visualize - create bank simulation HTML visualization"
	@echo "  make sim-graph    - serve bank simulation HTML visualization on LAN"
	@echo "  make reports-index - create data/reports/index.html navigation page"
	@echo "  make start        - start the main dashboard in the background at http://$(REPORTS_HOST):$(REPORTS_PORT)/"
	@echo "  make start-fg     - start the main dashboard in the foreground"
	@echo "  make stop         - stop whatever is listening on REPORTS_PORT"
	@echo "  make save-run      - save latest sequence NN artifacts into models/runs and update models/current"
	@echo "  make lr-save-run   - save latest LR artifacts into models/runs and update models/current"
	@echo "  make list-runs     - list recent canonical model runs"
	@echo "  make show-current  - show current model pointers"
	@echo "  make graph    - serve sequence neural net report on LAN"
	@echo "  make serve-lan    - alias for make graph"
	@echo "  make run          - run default sequence neural net pipeline -> serve LAN report"
	@echo "  make run-stock    - archive current NN, choose a >=3-year Yahoo stock, train/test/visualize/sim"
	@echo "  make nn-graph - serve sequence neural net report on LAN"
	@echo "  make nn-save      - archive current NN model, env files, and resolved params"
	@echo "  make lr-experiment - run old logistic-regression feature -> train -> backtest"
	@echo "  make xgb-experiment - run XGBoost feature -> train -> backtest without touching NN/LR outputs"
	@echo "  make xgb-visualize - create XGBoost HTML visualization"
	@echo "  make xgb-sim       - simulate XGBoost predictions on the final test split"
	@echo "  make xgb-sim-visualize - create XGBoost bank simulation visualization"
	@echo "  make xgb-save-run  - save latest XGBoost artifacts into models/runs and update models/current"
	@echo "  make strategy-experiment - run rule strategy baseline/backtest (buy-hold, previous movement, MA)"
	@echo "  make strategy-visualize  - create rule strategy HTML visualization"
	@echo "  make strategy-sim        - simulate rule strategy with one-position account logic"
	@echo "  make lr-visualize - create old logistic-regression HTML visualization"
	@echo "  make lr-sim       - simulate old logistic-regression predictions"
	@echo "  make lr-diagnostic - build old logistic-regression diagnostic report"
	@echo "  make lr-sweep     - run old logistic-regression preset/edge sweep"
	@echo "  make serve-reports - serve dashboard at http://127.0.0.1:$(REPORTS_PORT)/"
	@echo "  make smoke        - compile source files without downloading/training"
	@echo "  make repo-status  - show git state and generated artifact sizes"
	@echo "  make github-publish - create private GitHub repo and push source"
	@echo "  make preflight    - 3-day quick pipeline sanity run"
	@echo "  make live-up      - start Docker live paper-trading server"
	@echo "  make live-logs    - follow Docker live paper-trading logs"
	@echo "  make live-sync    - sync live_sim .env/model from current main NN config"
	@echo "  make live-real-check - validate Coinbase config/balances without placing orders"
	@echo "  make live-retrain-status - show live Docker retraining status"
	@echo "  make live-cache-status - show persisted live retrain candle caches"
	@echo "  make live-clear-retrain-lock - remove stale retrain lock when no retrain is running"
	@echo "  make update-model - manually retrain and activate the live Docker model"
	@echo "  make live-sync LIVE_MODEL_TYPE=lstm TRAINER_ENV=env/trainers/lstm_torch.env"
	@echo "  make live-sync LIVE_MODEL_TYPE=gru TRAINER_ENV=env/trainers/gru_torch.env"
	@echo "  make live-sync LIVE_RETRAIN_FREQUENCY=10h"
	@echo "  make live-sync LIVE_RETRAIN_TRAIN_START=2025-01-01T00:00:00Z LIVE_RETRAIN_TRAIN_END=2026-01-01T00:00:00Z"
	@echo ""
	@echo "Common overrides example:"
	@echo "  make experiment NN_BACKEND=torch NN_DEVICE=cuda NN_MODEL_TYPE=cnn NN_LOOKBACK=50 NN_CNN_FILTERS=16,32 NN_CNN_KERNEL_SIZES=5,3 NN_EPOCHS=25"
	@echo "  make experiment TRAINER_ENV=env/trainers/transformer_torch.env"
	@echo "  make experiment TRAINER_ENV=env/trainers/lstm_torch.env NN_LOOKBACK=70 NN_SEQUENCE_FEATURE_SET=technical EDGE=0"
	@echo "  make experiment TRAINER_ENV=env/trainers/gru_torch.env NN_LOOKBACK=70 NN_SEQUENCE_FEATURE_SET=technical EDGE=0"
	@echo "  make experiment ASSET_ENV=env/assets/btcusdt_5m.env TRAINER_ENV=env/trainers/cnn_torch.env"
	@echo "  make experiment TRAINER_ENV=env/trainers/mlp_torch.env"
	@echo "  make xgb-experiment XGB_DEVICE=cuda XGB_N_ESTIMATORS=300 XGB_MAX_DEPTH=4"
	@echo "  make xgb-experiment XGB_DEVICE=cuda XGB_TREE_METHOD=hist"
	@echo "  make xgb-sim XGB_SIM_DEFAULT_TEST_FRACTION=0.05"
	@echo "  make strategy-experiment STRATEGY_MODEL_TYPE=ma STRATEGY_MA_WINDOW=50"
	@echo "  make strategy-experiment STRATEGY_MODEL_TYPE=buy_hold"
	@echo "  make strategy-experiment STRATEGY_MODEL_TYPE=prev_movement"
	@echo "  make experiment AUTO_SAVE_RUN=0"
	@echo "  make experiment NN_MODEL_TYPE=cnn NN_LOOKBACK=20 NN_CNN_FILTERS=8,16 NN_CNN_KERNEL_SIZES=3,3"
	@echo "  make experiment NN_MODEL_TYPE=mlp"
	@echo "  make experiment TRAINER_ENV=env/trainers/lstm_torch.env"
	@echo "  make experiment SYMBOL=SOLUSDT INTERVAL=5m START=2026-04-18T00:00:00Z END=2026-05-18T00:00:00Z EDGE=0.0005 SPLIT=0.8 FEE=0.0001 THRESHOLD=0.55"
	@echo "  make download DATA_SOURCE=yahoo SYMBOL=AAPL INTERVAL=5m"
	@echo "  make download DATA_SOURCE=yahoo RANDOM_STOCK=1 INTERVAL=1d"
	@echo "  make sim"
	@echo "  make sim-visualize"
	@echo "  make sim SIM_START=2026-01-12 SIM_DURATION=1D"
	@echo "  make xgb-sim SIM_MIN_INVEST=100 SIM_MAX_INVEST=5000 SIM_CONFIDENCE_MULTIPLIER=3"
	@echo "  make run-stock STOCK_SYMBOL=V"
	@echo "  make run-stock STOCK_LOOKBACK_DAYS=1095 STOCK_MIN_HISTORY_YEARS=3 STOCK_SIM_TEST_FRACTION=0.05"
	@echo "  make run-stock STOCK_INTERVAL=1h STOCK_LOOKBACK_DAYS=720 STOCK_MIN_HISTORY_YEARS=2"
	@echo "  make run-stock STOCK_INTERVAL=90m STOCK_LOOKBACK_DAYS=60 STOCK_MIN_HISTORY_YEARS=0.15 STOCK_MIN_ROWS=100"
	@echo "  make sim-visualize SIM_ACTIVITY_BUCKET=hour SIM_MARKER_SIZE_BASIS=usd"
	@echo "  make nn-save NN_ARCHIVE_NAME=my-good-sol-model"
	@echo ""
	@echo "Fee/position testing examples:"
	@echo "  make backtest FEE=0.0005 POSITION_MODE=hold EXIT_THRESHOLD=0.45 MAX_HOLD_BARS=120"
	@echo "  make lr-backtest POSITION_MODE=one_bar FEE=0.001 THRESHOLD=0.55"

dirs:
	@mkdir -p data data/reports models $(GENERATED_DIRS)


nn: nn-experiment

features: lr-features

train: nn-train

backtest: nn-backtest

experiment: nn-experiment

diagnostic: nn-diagnostic

sweep: lr-sweep

visualize: nn-visualize

sim: nn-sim sim-visualize sim-graph

sim-visualize: nn-sim-visualize

sim-graph: nn-sim-graph

install:
	$(PIP) install -r requirements.txt

smoke:
	$(PYTHON) -m py_compile $(shell find src live_sim/app -name '*.py' | sort)

repo-status:
	@echo "Project: $(PROJECT_NAME)"
	@echo "Git status:"
	@if git rev-parse --is-inside-work-tree >/dev/null 2>&1; then \
		git status --short --branch; \
	else \
		echo "  not a git repository yet"; \
	fi
	@echo ""
	@echo "Generated artifact sizes:"
	@du -sh data models .venv 2>/dev/null || true
	@echo ""
	@echo "Source/config files intended for git:"
	@find src docs env .github -type f ! -path '*/__pycache__/*' ! -name '*.pyc' ! -name 'local.env' 2>/dev/null | sort
	@printf '%s\n' Makefile requirements.txt README.md .gitignore

github-check:
	@scripts/github.sh check

github-init:
	@scripts/github.sh init

github-commit: github-init
	@scripts/github.sh commit

github-create-private: github-check github-init
	@scripts/github.sh create-private

github-push: github-check github-init
	@scripts/github.sh push

github-publish:
	@scripts/github.sh publish

download: dirs
	$(PYTHON) src/download.py \
		--source $(DATA_SOURCE) \
		--symbol $(SYMBOL) \
		$(RANDOM_STOCK_FLAG) \
		--stock-list $(STOCK_LIST) \
		--interval $(INTERVAL) \
		--start $(START) \
		--end $(END) \
		--out $(RAW_DATA) \
		--cache-file $(DOWNLOAD_CACHE)

lr-features: dirs
	$(PYTHON) src/features.py \
		--input $(RAW_DATA) \
		--output $(FEATURES_DATA) \
		--meta-out $(FEATURES_META) \
		--edge $(EDGE) \
		--interval $(INTERVAL) \
		--short-edge $(SHORT_EDGE) \
		--return-windows $(RETURN_WINDOWS) \
		--vol-windows $(VOL_WINDOWS) \
		--sma-short-window $(SMA_SHORT_WINDOW) \
		--sma-long-window $(SMA_LONG_WINDOW) \
		--extra-sma-windows $(EXTRA_SMA_WINDOWS) \
		--volume-z-window $(VOLUME_Z_WINDOW) \
		--volume-ratio-windows $(VOLUME_RATIO_WINDOWS) \
		$(TIME_FEATURE_FLAG)

lr-train: dirs
	$(PYTHON) src/train.py \
		--features $(FEATURES_DATA) \
		--feature-meta $(FEATURES_META) \
		--model-out $(MODEL_OUT) \
		--metrics-out $(TRAIN_METRICS) \
		--split $(SPLIT) \
		--lr $(LR) \
		--epochs $(EPOCHS) \
		--l2 $(L2) \
		--decision-threshold $(DECISION_THRESHOLD) \
		--threshold-grid $(THRESHOLD_GRID) \
		--short-edge $(SHORT_EDGE) \
		--optimize-metric $(OPTIMIZE_METRIC) \
		--class-weight-mode $(CLASS_WEIGHT_MODE)

lr-backtest: dirs
	$(PYTHON) src/backtest.py \
		--features $(FEATURES_DATA) \
		--model $(MODEL_OUT) \
		--fee $(FEE) \
		--threshold $(THRESHOLD) \
		--split $(SPLIT) \
		--position-mode $(POSITION_MODE) \
		--exit-threshold $(EXIT_THRESHOLD) \
		--max-hold-bars $(MAX_HOLD_BARS) \
		--stop-loss $(STOP_LOSS) \
		--take-profit $(TAKE_PROFIT) \
		--report-out $(BACKTEST_REPORT) \
		--predictions-out $(PREDICTIONS_OUT)

lr-experiment: lr-features lr-train lr-backtest
	@echo "Logistic-regression experiment complete."
	@echo "Raw data:      $(RAW_DATA)"
	@echo "Features:      $(FEATURES_DATA)"
	@echo "Model:         $(MODEL_OUT)"
	@echo "Train metrics: $(TRAIN_METRICS)"
	@echo "Backtest:      $(BACKTEST_REPORT)"
	@echo "Predictions:   $(PREDICTIONS_OUT)"
	@if [ "$(AUTO_SAVE_RUN)" != "0" ]; then \
		$(MAKE) lr-save-run; \
	fi

nn-train: dirs
	$(PYTHON) src/train_sequence_nn.py \
		--raw-data $(RAW_DATA) \
		--model-out $(NN_MODEL_OUT) \
		--metrics-out $(NN_TRAIN_METRICS) \
		--model-type $(NN_MODEL_TYPE) \
		--backend $(NN_BACKEND) \
		--device $(NN_DEVICE) \
		--lookback $(NN_LOOKBACK) \
		--sequence-feature-set $(NN_SEQUENCE_FEATURE_SET) \
		--edge $(EDGE) \
		--split $(SPLIT) \
		--short-edge $(SHORT_EDGE) \
		--cnn-filters $(NN_CNN_FILTERS) \
		--cnn-kernel-sizes $(NN_CNN_KERNEL_SIZES) \
		--lstm-hidden-size $(NN_LSTM_HIDDEN_SIZE) \
		--lstm-layers $(NN_LSTM_LAYERS) \
		--lstm-dropout $(NN_LSTM_DROPOUT) \
		--gru-hidden-size $(NN_GRU_HIDDEN_SIZE) \
		--gru-layers $(NN_GRU_LAYERS) \
		--gru-dropout $(NN_GRU_DROPOUT) \
		--transformer-d-model $(NN_TRANSFORMER_D_MODEL) \
		--transformer-heads $(NN_TRANSFORMER_HEADS) \
		--transformer-layers $(NN_TRANSFORMER_LAYERS) \
		--transformer-ff-dim $(NN_TRANSFORMER_FF_DIM) \
		--transformer-dropout $(NN_TRANSFORMER_DROPOUT) \
		--hidden-layers $(NN_HIDDEN_LAYERS) \
		--lr $(NN_LR) \
		--epochs $(NN_EPOCHS) \
		--batch-size $(NN_BATCH_SIZE) \
		--l2 $(NN_L2) \
		--decision-threshold $(DECISION_THRESHOLD) \
		--threshold-grid $(THRESHOLD_GRID) \
		--short-edge $(SHORT_EDGE) \
		--optimize-metric $(OPTIMIZE_METRIC) \
		--class-weight-mode $(NN_CLASS_WEIGHT_MODE) \
		--seed $(NN_SEED)

nn-backtest: dirs
	$(PYTHON) src/backtest_sequence_nn.py \
		--raw-data $(RAW_DATA) \
		--model $(NN_MODEL_OUT) \
		--fee $(FEE) \
		--threshold $(THRESHOLD) \
		--split $(SPLIT) \
		--position-mode $(POSITION_MODE) \
		--exit-threshold $(EXIT_THRESHOLD) \
		--max-hold-bars $(MAX_HOLD_BARS) \
		--stop-loss $(STOP_LOSS) \
		--take-profit $(TAKE_PROFIT) \
		--report-out $(NN_BACKTEST_REPORT) \
		--predictions-out $(NN_PREDICTIONS_OUT)

nn-experiment: nn-train nn-backtest
	@echo "Sequence neural net experiment complete."
	@echo "Raw data:      $(RAW_DATA)"
	@echo "Model:         $(NN_MODEL_OUT)"
	@echo "Train metrics: $(NN_TRAIN_METRICS)"
	@echo "Backtest:      $(NN_BACKTEST_REPORT)"
	@echo "Predictions:   $(NN_PREDICTIONS_OUT)"
	@if [ "$(AUTO_SAVE_RUN)" != "0" ]; then \
		$(MAKE) save-run; \
	fi

nn-diagnostic:
	@echo "Sequence NN diagnostics are written during train/backtest:"
	@echo "Train metrics: $(NN_TRAIN_METRICS)"
	@echo "Backtest:      $(NN_BACKTEST_REPORT)"
	@echo "Predictions:   $(NN_PREDICTIONS_OUT)"

save-run: dirs
	@scripts/archive_run.sh nn

lr-save-run: dirs
	@scripts/archive_run.sh lr

xgb-features: dirs
	$(PYTHON) src/features.py \
		--input $(RAW_DATA) \
		--output $(XGB_FEATURES_DATA) \
		--meta-out $(XGB_FEATURES_META) \
		--edge $(EDGE) \
		--interval $(INTERVAL) \
		--short-edge $(SHORT_EDGE) \
		--return-windows $(RETURN_WINDOWS) \
		--vol-windows $(VOL_WINDOWS) \
		--sma-short-window $(SMA_SHORT_WINDOW) \
		--sma-long-window $(SMA_LONG_WINDOW) \
		--extra-sma-windows $(EXTRA_SMA_WINDOWS) \
		--volume-z-window $(VOLUME_Z_WINDOW) \
		--volume-ratio-windows $(VOLUME_RATIO_WINDOWS) \
		$(TIME_FEATURE_FLAG)

xgb-train: dirs
	$(PYTHON) src/xgboost_model/train_xgboost.py \
		--features $(XGB_FEATURES_DATA) \
		--feature-meta $(XGB_FEATURES_META) \
		--model-out $(XGB_MODEL_OUT) \
		--metrics-out $(XGB_TRAIN_METRICS) \
		--split $(SPLIT) \
		--decision-threshold $(DECISION_THRESHOLD) \
		--threshold-grid $(THRESHOLD_GRID) \
		--short-edge $(SHORT_EDGE) \
		--optimize-metric $(OPTIMIZE_METRIC) \
		--class-weight-mode $(XGB_CLASS_WEIGHT_MODE) \
		$(XGB_POS_WEIGHT_ARG) \
		--n-estimators $(XGB_N_ESTIMATORS) \
		--max-depth $(XGB_MAX_DEPTH) \
		--learning-rate $(XGB_LEARNING_RATE) \
		--subsample $(XGB_SUBSAMPLE) \
		--colsample-bytree $(XGB_COLSAMPLE_BYTREE) \
		--min-child-weight $(XGB_MIN_CHILD_WEIGHT) \
		--reg-lambda $(XGB_REG_LAMBDA) \
		--reg-alpha $(XGB_REG_ALPHA) \
		--gamma $(XGB_GAMMA) \
		--tree-method $(XGB_TREE_METHOD) \
		--device $(XGB_DEVICE) \
		--n-jobs $(XGB_N_JOBS) \
		--seed $(XGB_SEED)

xgb-backtest: dirs
	$(PYTHON) src/xgboost_model/backtest_xgboost.py \
		--features $(XGB_FEATURES_DATA) \
		--model $(XGB_MODEL_OUT) \
		--fee $(FEE) \
		--threshold $(THRESHOLD) \
		--split $(SPLIT) \
		--position-mode $(POSITION_MODE) \
		--exit-threshold $(EXIT_THRESHOLD) \
		--max-hold-bars $(MAX_HOLD_BARS) \
		--stop-loss $(STOP_LOSS) \
		--take-profit $(TAKE_PROFIT) \
		--report-out $(XGB_BACKTEST_REPORT) \
		--predictions-out $(XGB_PREDICTIONS_OUT)

xgb-experiment gxboost-experiment: xgb-features xgb-train xgb-backtest
	@echo "XGBoost experiment complete."
	@echo "Raw data:      $(RAW_DATA)"
	@echo "Features:      $(XGB_FEATURES_DATA)"
	@echo "Model:         $(XGB_MODEL_OUT)"
	@echo "Train metrics: $(XGB_TRAIN_METRICS)"
	@echo "Backtest:      $(XGB_BACKTEST_REPORT)"
	@echo "Predictions:   $(XGB_PREDICTIONS_OUT)"
	@if [ "$(AUTO_SAVE_RUN)" != "0" ]; then \
		$(MAKE) xgb-save-run; \
	fi

gxboost-train: xgb-train

gxboost-backtest: xgb-backtest

xgb-save-run: dirs
	@scripts/archive_run.sh xgb

xgb-visualize: dirs
	@scripts/visualize_model.sh xgb

xgb-sim gxboost-sim: dirs
	@scripts/simulate_predictions.sh xgb

xgb-sim-visualize: dirs
	@scripts/visualize_sim.sh xgb

xgb-sim-graph: xgb-sim-visualize reports-index
	@echo "Open this on your other laptop:"
	@echo "http://$(REPORTS_HOST):$(REPORTS_PORT)/$(XGB_SIM_VISUALIZATION_URL_PATH)"
	@echo "Report index:"
	@echo "http://$(REPORTS_HOST):$(REPORTS_PORT)/$(REPORTS_INDEX_URL_PATH)"
	$(MAKE) serve-reports-lan CHECK_URL_PATH=$(XGB_SIM_VISUALIZATION_URL_PATH)

xgb-graph: xgb-visualize reports-index
	@echo "Open this on your other laptop:"
	@echo "http://$(REPORTS_HOST):$(REPORTS_PORT)/$(XGB_VISUALIZATION_URL_PATH)"
	@echo "Report index:"
	@echo "http://$(REPORTS_HOST):$(REPORTS_PORT)/$(REPORTS_INDEX_URL_PATH)"
	$(MAKE) serve-reports-lan CHECK_URL_PATH=$(XGB_VISUALIZATION_URL_PATH)

xgb-preflight:
	@scripts/preflight.sh xgb

strategy-train: dirs
	$(PYTHON) src/strategy_model/train_strategy.py \
		--raw-data $(RAW_DATA) \
		--model-type $(STRATEGY_MODEL_TYPE) \
		--model-out $(STRATEGY_MODEL_OUT) \
		--metrics-out $(STRATEGY_TRAIN_METRICS) \
		--split $(SPLIT) \
		--edge $(EDGE) \
		--ma-window $(STRATEGY_MA_WINDOW) \
		--threshold $(THRESHOLD)

strategy-backtest: dirs
	$(PYTHON) src/strategy_model/backtest_strategy.py \
		--raw-data $(RAW_DATA) \
		--model $(STRATEGY_MODEL_OUT) \
		--fee $(FEE) \
		--threshold $(THRESHOLD) \
		--split $(SPLIT) \
		--position-mode $(POSITION_MODE) \
		--exit-threshold $(EXIT_THRESHOLD) \
		--max-hold-bars $(MAX_HOLD_BARS) \
		--stop-loss $(STOP_LOSS) \
		--take-profit $(TAKE_PROFIT) \
		--report-out $(STRATEGY_BACKTEST_REPORT) \
		--predictions-out $(STRATEGY_PREDICTIONS_OUT)

strategy-experiment: strategy-train strategy-backtest
	@echo "Strategy experiment complete."
	@echo "Raw data:      $(RAW_DATA)"
	@echo "Strategy:      $(STRATEGY_MODEL_TYPE)"
	@echo "Model:         $(STRATEGY_MODEL_OUT)"
	@echo "Train metrics: $(STRATEGY_TRAIN_METRICS)"
	@echo "Backtest:      $(STRATEGY_BACKTEST_REPORT)"
	@echo "Predictions:   $(STRATEGY_PREDICTIONS_OUT)"
	@if [ "$(AUTO_SAVE_RUN)" != "0" ]; then \
		$(MAKE) strategy-save-run; \
	fi

strategy-visualize: dirs
	@scripts/visualize_model.sh strategy

strategy-sim: dirs
	@scripts/simulate_predictions.sh strategy

strategy-sim-visualize: dirs
	@scripts/visualize_sim.sh strategy

strategy-graph: strategy-visualize reports-index
	@scripts/dashboard.sh print-url $(STRATEGY_VISUALIZATION_URL_PATH)
	$(MAKE) serve-reports-lan CHECK_URL_PATH=$(STRATEGY_VISUALIZATION_URL_PATH)

strategy-sim-graph: strategy-sim-visualize reports-index
	@scripts/dashboard.sh print-url $(STRATEGY_SIM_VISUALIZATION_URL_PATH)
	$(MAKE) serve-reports-lan CHECK_URL_PATH=$(STRATEGY_SIM_VISUALIZATION_URL_PATH)

strategy-save-run: dirs
	@scripts/archive_run.sh strategy

list-runs:
	$(PYTHON) src/archive_model.py --list-runs --run-store $(RUNS_ROOT)

show-current:
	$(PYTHON) src/archive_model.py --show-current --current-root $(CURRENT_ROOT)

nn-save save-nn-model: dirs
	@scripts/archive_run.sh nn-save

nn-visualize: dirs
	@scripts/visualize_model.sh nn

nn-sim: dirs
	@scripts/simulate_predictions.sh nn

nn-sim-visualize: dirs
	@scripts/visualize_sim.sh nn

lr-diagnostic: dirs
	$(PYTHON) src/diagnostic_report.py \
		--features $(FEATURES_DATA) \
		--model $(MODEL_OUT) \
		--split $(SPLIT) \
		--fee $(FEE) \
		--thresholds $(THRESHOLD_GRID) \
		--report-out $(DIAG_REPORT) \
		--threshold-table-out $(DIAG_TABLE) \
		--test-predictions-out $(DIAG_TEST_PREDICTIONS)
	@echo "Diagnostic report: $(DIAG_REPORT)"
	@echo "Threshold table:   $(DIAG_TABLE)"
	@echo "Test predictions:  $(DIAG_TEST_PREDICTIONS)"

lr-visualize: dirs
	@scripts/visualize_model.sh lr

lr-sim: dirs
	@scripts/simulate_predictions.sh lr

lr-sim-visualize: dirs
	@scripts/visualize_sim.sh lr

day-sim: sim

serve-reports:
	$(MAKE) start REPORTS_HOST=127.0.0.1

CHECK_URL_PATH ?= $(REPORTS_INDEX_URL_PATH)

start: dirs reports-index
	@scripts/dashboard.sh start

start-fg: dirs reports-index
	@scripts/dashboard.sh start-fg

stop:
	@scripts/dashboard.sh stop

serve-reports-lan:
	@scripts/dashboard.sh serve-lan

reports-index: dirs
	$(PYTHON) src/report_index.py \
		--output $(REPORTS_INDEX_OUT) \
		--title "$(PROJECT_NAME) reports" \
		--model-url /$(NN_VISUALIZATION_URL_PATH) \
		--sim-url /$(NN_SIM_VISUALIZATION_URL_PATH) \
		--lr-model-url /$(VISUALIZATION_URL_PATH) \
		--lr-sim-url /$(SIM_VISUALIZATION_URL_PATH) \
		--xgb-model-url /$(XGB_VISUALIZATION_URL_PATH) \
		--xgb-sim-url /$(XGB_SIM_VISUALIZATION_URL_PATH) \
		--strategy-model-url /$(STRATEGY_VISUALIZATION_URL_PATH) \
		--strategy-sim-url /$(STRATEGY_SIM_VISUALIZATION_URL_PATH)
	@echo "Reports index: $(REPORTS_INDEX_OUT)"

nn-graph: nn-visualize reports-index
	@scripts/dashboard.sh print-url $(NN_VISUALIZATION_URL_PATH)
	$(MAKE) serve-reports-lan CHECK_URL_PATH=$(NN_VISUALIZATION_URL_PATH)

nn-serve-lan: nn-graph

lan: nn-graph

graph: nn-graph

serve-lan: graph

nn-sim-graph: nn-sim-visualize reports-index
	@scripts/dashboard.sh print-url $(NN_SIM_VISUALIZATION_URL_PATH)
	$(MAKE) serve-reports-lan CHECK_URL_PATH=$(NN_SIM_VISUALIZATION_URL_PATH)

lr-graph: lr-visualize reports-index
	@scripts/dashboard.sh print-url $(VISUALIZATION_URL_PATH)
	$(MAKE) serve-reports-lan CHECK_URL_PATH=$(VISUALIZATION_URL_PATH)

lr-serve-lan: lr-graph

lr-sim-graph: lr-sim-visualize reports-index
	@scripts/dashboard.sh print-url $(SIM_VISUALIZATION_URL_PATH)
	$(MAKE) serve-reports-lan CHECK_URL_PATH=$(SIM_VISUALIZATION_URL_PATH)

lr-sweep: dirs
	$(PYTHON) src/sweep.py \
		--raw-input $(RAW_DATA) \
		--interval $(INTERVAL) \
		--output-dir $(SWEEP_OUTPUT_DIR) \
		--threshold-grid $(THRESHOLD_GRID) \
		--optimize-metric $(OPTIMIZE_METRIC) \
		--split $(SPLIT) \
		--lr $(LR) \
		--epochs $(EPOCHS) \
		--l2 $(L2) \
		--class-weight-mode $(CLASS_WEIGHT_MODE) \
		--fee $(FEE)

preflight:
	@scripts/preflight.sh nn

lr-preflight:
	@scripts/preflight.sh lr

clean:
	rm -rf src/__pycache__

LIVE_MODEL_TYPE ?= $(NN_MODEL_TYPE)
LIVE_MODEL_SOURCE ?= models/nn/$(LIVE_MODEL_TYPE)/$(DATA_SOURCE)/$(SYMBOL)/$(INTERVAL)/model.npz
LIVE_ENV_ACTIVE ?= live_sim/env/active.env
LIVE_MODEL_ENV ?= live_sim/env/models/$(DATA_SOURCE)/$(SYMBOL)/$(INTERVAL)/$(LIVE_MODEL_TYPE).env
LIVE_ENV_SNAPSHOT_ROOT ?= models/live_env_snapshots
LIVE_SIM_REPORT ?= models/sim/nn/$(LIVE_MODEL_TYPE)/$(DATA_SOURCE)/$(SYMBOL)/$(INTERVAL)/bank_report.json
LIVE_RETRAIN_FREQUENCY ?= 1d
LIVE_RETRAIN_TRAIN_START ?= $(START)
LIVE_RETRAIN_TRAIN_END ?= $(END)
LIVE_RETRAIN_LOOKBACK_DAYS ?= 913
LIVE_RETRAIN_CACHE_DIR ?= /app/state/downloads
LIVE_TRAIN_MODEL_TYPE ?= $(LIVE_MODEL_TYPE)
LIVE_TRAIN_BACKEND ?= $(NN_BACKEND)
LIVE_TRAIN_DEVICE ?= $(NN_DEVICE)
LIVE_TRAIN_LOOKBACK ?= $(NN_LOOKBACK)
LIVE_TRAIN_SEQUENCE_FEATURE_SET ?= $(NN_SEQUENCE_FEATURE_SET)
LIVE_TRAIN_EDGE ?= $(EDGE)
LIVE_TRAIN_USE_FULL_WINDOW ?= true
LIVE_STARTING_CASH ?= $(SIM_STARTING_CASH)
LIVE_MAX_INVEST ?= $(SIM_MAX_INVEST)
LIVE_MAX_SHORT_INVEST ?= $(SIM_MAX_SHORT_INVEST)
LIVE_MIN_INVEST ?= $(SIM_MIN_INVEST)
LIVE_CONFIDENCE_MULTIPLIER ?= $(SIM_CONFIDENCE_MULTIPLIER)
LIVE_SLIPPAGE ?= $(SIM_SLIPPAGE)
LIVE_CATCHUP_ENABLED ?= true
LIVE_CATCHUP_SPREAD_PCT ?= $(SIM_SPREAD_PCT)
LIVE_CATCHUP_MAX_BARS ?= 0
LIVE_CATCHUP_RETRY_SECONDS ?= 60
LIVE_HOST_PORT ?= 8080
EXECUTION_MODE ?= paper
REAL_TRADING_ENABLED ?= false
REAL_REQUIRE_MANUAL_ARM ?= true
REAL_QUICK_ARM_ENABLED ?= false
REAL_MAX_TOTAL_USD ?= 20
REAL_MAX_ORDER_USD ?= 5
REAL_MIN_ORDER_USD ?= 1
REAL_PORTFOLIO_MODE ?= account_balances
REAL_CASH_ASSET ?= USDC
REAL_BASE_ASSET ?= SOL
COINBASE_PRODUCT_ID ?= SOL-USD
COINBASE_API_KEY ?=
COINBASE_API_SECRET ?=
COINBASE_TIMEOUT ?= 10
SOLANA_RPC_URL ?= <your Helius or QuickNode RPC URL>
SOLANA_KEYPAIR_PATH ?= /app/state/solana-keypair.json
SOL_RESERVED_FOR_GAS ?= 0.02
SOLANA_RPC_TIMEOUT ?= 10
SOLANA_CONFIRM_POLLS ?= 20
SOLANA_CONFIRM_DELAY_SECONDS ?= 1
JUPITER_BASE_URL ?= https://lite-api.jup.ag/swap/v1
JUPITER_PRODUCT_ID ?= SOL-USDC
JUPITER_SLIPPAGE_BPS ?= 50
JUPITER_PRIORITY_FEE_LAMPORTS ?= auto
JUPITER_TIMEOUT ?= 10
REAL_ARM_TOKEN ?= <local arm token, any text>
REAL_ORDER_STATUS_POLLS ?= 5
REAL_ORDER_STATUS_DELAY_SECONDS ?= 0.75
HOST_UID := $(shell id -u)
HOST_GID := $(shell id -g)

live-sync:
	@scripts/live_sync.sh

live-setup:
	$(MAKE) -C live_sim setup

live-build:
	$(MAKE) -C live_sim build

live-build-sync: live-sync
	$(MAKE) -C live_sim build

live-up:
	$(MAKE) -C live_sim up

live-up-sync: live-sync
	$(MAKE) -C live_sim up

live-down:
	$(MAKE) -C live_sim down

live-logs:
	$(MAKE) -C live_sim logs

live-reset:
	$(MAKE) -C live_sim reset

live-test:
	$(MAKE) -C live_sim test

live-real-check:
	$(MAKE) -C live_sim real-check

live-retrain-status:
	$(MAKE) -C live_sim retrain-status

live-retrain-now:
	$(MAKE) -C live_sim retrain-now

live-clear-retrain-lock:
	$(MAKE) -C live_sim clear-stale-retrain-lock

live-cache-status:
	$(MAKE) -C live_sim cache-status

live-update-model:
	$(MAKE) -C live_sim update-model

live-update-model-sync: live-sync
	$(MAKE) -C live_sim update-model

update-model: live-update-model
