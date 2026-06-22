SHELL := /bin/bash

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
	@STOCK_END_RESOLVED="$(STOCK_END)"; \
	STOCK_START_RESOLVED="$(STOCK_START)"; \
	if [ -z "$${STOCK_START_RESOLVED}" ]; then \
		STOCK_START_RESOLVED="$$(STOCK_END="$${STOCK_END_RESOLVED}" STOCK_LOOKBACK_DAYS="$(STOCK_LOOKBACK_DAYS)" $(PYTHON) -c 'from datetime import datetime, timezone, timedelta; import os; raw=os.environ["STOCK_END"]; raw=raw[:-1]+"+00:00" if raw.endswith("Z") else raw; end=datetime.fromisoformat(raw); end=end.replace(tzinfo=timezone.utc) if end.tzinfo is None else end.astimezone(timezone.utc); start=end-timedelta(days=int(os.environ["STOCK_LOOKBACK_DAYS"])); print(start.isoformat().replace("+00:00","Z"))')"; \
	fi; \
	STOCK_SYMBOL="$$( \
		$(PYTHON) src/select_stock.py \
			--symbol "$(STOCK_SYMBOL)" \
			--stock-list "$(STOCK_LIST)" \
			--interval "$(STOCK_INTERVAL)" \
			--start "$${STOCK_START_RESOLVED}" \
			--end "$${STOCK_END_RESOLVED}" \
			--min-rows "$(STOCK_MIN_ROWS)" \
			--min-history-years "$(STOCK_MIN_HISTORY_YEARS)" \
	)"; \
	if [ -z "$${STOCK_SYMBOL}" ]; then echo "No eligible stock selected"; exit 1; fi; \
	RAW_DATA_PATH="data/downloads/yahoo/$${STOCK_SYMBOL}/$(STOCK_INTERVAL)/candles.parquet"; \
	ARCHIVE_NAME="before-stock-$${STOCK_SYMBOL}-$$(date -u +%Y%m%dT%H%M%SZ)"; \
	STOCK_ARCHIVE_NAME="stock-$${STOCK_SYMBOL}-$$(date -u +%Y%m%dT%H%M%SZ)"; \
	echo "Selected stock: $${STOCK_SYMBOL}"; \
	echo "Stock data source: yahoo"; \
	echo "Stock interval: $(STOCK_INTERVAL)"; \
	echo "Stock date range: $${STOCK_START_RESOLVED} to $${STOCK_END_RESOLVED}"; \
	echo "Stock history filter: >= $(STOCK_MIN_HISTORY_YEARS) years and >= $(STOCK_MIN_ROWS) rows"; \
	echo "Stock split: $(STOCK_SPLIT) train, final $(STOCK_SIM_TEST_FRACTION) simulation fraction"; \
	if [ "$(STOCK_ARCHIVE_CURRENT)" != "0" ]; then \
		echo "Archiving current NN model/config before stock run: $${ARCHIVE_NAME}"; \
		$(MAKE) nn-save NN_ARCHIVE_NAME="$${ARCHIVE_NAME}"; \
	fi; \
	$(MAKE) download \
		DATA_SOURCE=yahoo \
		SYMBOL="$${STOCK_SYMBOL}" \
		RANDOM_STOCK=0 \
		INTERVAL="$(STOCK_INTERVAL)" \
		START="$${STOCK_START_RESOLVED}" \
		END="$${STOCK_END_RESOLVED}"; \
	$(PYTHON) -c 'import sys, pandas as pd; p=sys.argv[1]; m=int(sys.argv[2]); df=pd.read_parquet(p); t=pd.to_datetime(df["open_time"], utc=True); n=len(df); print(f"Stock downloaded rows: {n}"); print(f"Stock downloaded range: {t.min()} to {t.max()}"); sys.exit(f"Only {n} rows downloaded; expected at least {m}. Increase STOCK_LOOKBACK_DAYS or check the Yahoo range.") if n < m else None' "$${RAW_DATA_PATH}" "$(STOCK_MIN_ROWS)"; \
	$(MAKE) experiment visualize nn-sim nn-sim-visualize reports-index \
		ASSET_ENV="$(STOCK_ASSET_ENV)" \
		DATA_SOURCE=yahoo \
		SYMBOL="$${STOCK_SYMBOL}" \
		RANDOM_STOCK=0 \
		INTERVAL="$(STOCK_INTERVAL)" \
		START="$${STOCK_START_RESOLVED}" \
		END="$${STOCK_END_RESOLVED}" \
		SPLIT="$(STOCK_SPLIT)" \
		SIM_DEFAULT_TEST_FRACTION="$(STOCK_SIM_TEST_FRACTION)"; \
	echo "Archiving stock NN model/config: $${STOCK_ARCHIVE_NAME}"; \
	$(MAKE) nn-save \
		ASSET_ENV="$(STOCK_ASSET_ENV)" \
		DATA_SOURCE=yahoo \
		SYMBOL="$${STOCK_SYMBOL}" \
		RANDOM_STOCK=0 \
		INTERVAL="$(STOCK_INTERVAL)" \
		START="$${STOCK_START_RESOLVED}" \
		END="$${STOCK_END_RESOLVED}" \
		SPLIT="$(STOCK_SPLIT)" \
		SIM_DEFAULT_TEST_FRACTION="$(STOCK_SIM_TEST_FRACTION)" \
		NN_ARCHIVE_NAME="$${STOCK_ARCHIVE_NAME}"; \
	echo "Open model visualization on your other laptop:"; \
	echo "http://$(REPORTS_HOST):$(REPORTS_PORT)/nn/$(NN_MODEL_TYPE)/yahoo/$${STOCK_SYMBOL}/$(STOCK_INTERVAL)/visualization.html"; \
	echo "Open simulation visualization on your other laptop:"; \
	echo "http://$(REPORTS_HOST):$(REPORTS_PORT)/sim/nn/$(NN_MODEL_TYPE)/yahoo/$${STOCK_SYMBOL}/$(STOCK_INTERVAL)/visualization.html"; \
	echo "Report index:"; \
	echo "http://$(REPORTS_HOST):$(REPORTS_PORT)/$(REPORTS_INDEX_URL_PATH)"; \
	if curl -fsS --max-time 2 "http://$(REPORTS_HOST):$(REPORTS_PORT)/$(REPORTS_INDEX_URL_PATH)" >/dev/null 2>&1; then \
		echo "Reports server is running on $(REPORTS_HOST):$(REPORTS_PORT)."; \
	else \
		echo "Starting dashboard server in the background on $(REPORTS_HOST):$(REPORTS_PORT)."; \
		mkdir -p data/reports; \
		setsid $(PYTHON) src/dashboard_server.py \
			--host $(REPORTS_HOST) \
			--port $(REPORTS_PORT) \
			--reports-root data/reports \
			--root . \
			--live-url $(LIVE_LOCAL_URL) \
			--live-public-url $(LIVE_PUBLIC_URL) \
			> data/reports/dashboard_server.log 2>&1 < /dev/null & \
		sleep 1; \
		curl -fsS --max-time 3 "http://$(REPORTS_HOST):$(REPORTS_PORT)/api/dashboard/status" >/dev/null \
			&& echo "Dashboard server started." \
			|| { echo "Dashboard server did not respond. Check data/reports/dashboard_server.log"; exit 1; }; \
	fi

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
	@command -v git >/dev/null 2>&1 || { echo "Missing git. Install git, then rerun make github-publish."; exit 1; }
	@command -v gh >/dev/null 2>&1 || { echo "Missing GitHub CLI 'gh'. Install gh, run 'gh auth login', then rerun make github-publish."; exit 1; }
	@gh auth status >/dev/null 2>&1 || { echo "GitHub CLI is not authenticated. Run 'gh auth login', then rerun make github-publish."; exit 1; }
	@echo "GitHub CLI is installed and authenticated."

github-init:
	@if git rev-parse --is-inside-work-tree >/dev/null 2>&1; then \
		echo "Already inside a git repository."; \
	else \
		git init -b main; \
	fi
	@git branch -M main

github-commit: github-init
	@git var GIT_AUTHOR_IDENT >/dev/null 2>&1 || { echo "Missing git author identity. Run: git config --global user.name 'Your Name' && git config --global user.email 'you@example.com'"; exit 1; }
	@git add .gitignore .github/workflows/smoke.yml README.md docs/project_structure.md Makefile requirements.txt env src
	@if git diff --cached --quiet; then \
		echo "No source/config/docs changes staged for commit."; \
	else \
		git commit -m "$(COMMIT_MSG)"; \
	fi

github-create-private: github-check github-init
	@if git remote get-url origin >/dev/null 2>&1; then \
		echo "origin already exists: $$(git remote get-url origin)"; \
	else \
		gh repo create $(GITHUB_REPO) --private --source=. --remote=origin; \
	fi

github-push: github-check github-init
	@git remote get-url origin >/dev/null 2>&1 || { echo "Missing origin remote. Run make github-create-private first."; exit 1; }
	git push -u origin main

github-publish:
	$(MAKE) github-check
	$(MAKE) github-init
	$(MAKE) github-commit
	$(MAKE) github-create-private
	$(MAKE) github-push

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
	$(PYTHON) src/archive_model.py \
		--run-store $(RUNS_ROOT) \
		--current-root $(CURRENT_ROOT) \
		--write-current \
		--family nn \
		--model-type $(NN_MODEL_TYPE) \
		--backend $(NN_BACKEND) \
		--asset-env $(ASSET_ENV) \
		--trainer-env $(TRAINER_ENV) \
		--model $(NN_MODEL_OUT) \
		--train-metrics $(NN_TRAIN_METRICS) \
		--backtest-report $(NN_BACKTEST_REPORT) \
		--predictions $(NN_PREDICTIONS_OUT) \
		$(addprefix --env-file ,$(ENV_FILES)) \
		--param ASSET_ENV="$(ASSET_ENV)" \
		--param TRAINER_ENV="$(TRAINER_ENV)" \
		--param SYMBOL="$(SYMBOL)" \
		--param DATA_SOURCE="$(DATA_SOURCE)" \
		--param INTERVAL="$(INTERVAL)" \
		--param START="$(START)" \
		--param END="$(END)" \
		--param SPLIT="$(SPLIT)" \
		--param EDGE="$(EDGE)" \
		--param FEE="$(FEE)" \
		--param THRESHOLD="$(THRESHOLD)" \
		--param SIM_STARTING_CASH="$(SIM_STARTING_CASH)" \
		--param SIM_MIN_INVEST="$(SIM_MIN_INVEST)" \
		--param SIM_MAX_INVEST="$(SIM_MAX_INVEST)" \
		--param SIM_CONFIDENCE_MULTIPLIER="$(SIM_CONFIDENCE_MULTIPLIER)" \
		--param SIM_POSITION_MODE="$(SIM_POSITION_MODE)" \
		--param SIM_SLIPPAGE="$(SIM_SLIPPAGE)" \
		--param SIM_SPREAD_PCT="$(SIM_SPREAD_PCT)" \
		--param POSITION_MODE="$(POSITION_MODE)" \
		--param EXIT_THRESHOLD="$(EXIT_THRESHOLD)" \
		--param MAX_HOLD_BARS="$(MAX_HOLD_BARS)" \
		--param STOP_LOSS="$(STOP_LOSS)" \
		--param TAKE_PROFIT="$(TAKE_PROFIT)" \
		--param NN_BACKEND="$(NN_BACKEND)" \
		--param NN_DEVICE="$(NN_DEVICE)" \
		--param NN_MODEL_TYPE="$(NN_MODEL_TYPE)" \
		--param NN_LOOKBACK="$(NN_LOOKBACK)" \
		--param NN_SEQUENCE_FEATURE_SET="$(NN_SEQUENCE_FEATURE_SET)" \
		--param NN_CNN_FILTERS="$(NN_CNN_FILTERS)" \
		--param NN_CNN_KERNEL_SIZES="$(NN_CNN_KERNEL_SIZES)" \
		--param NN_LSTM_HIDDEN_SIZE="$(NN_LSTM_HIDDEN_SIZE)" \
		--param NN_LSTM_LAYERS="$(NN_LSTM_LAYERS)" \
		--param NN_LSTM_DROPOUT="$(NN_LSTM_DROPOUT)" \
		--param NN_GRU_HIDDEN_SIZE="$(NN_GRU_HIDDEN_SIZE)" \
		--param NN_GRU_LAYERS="$(NN_GRU_LAYERS)" \
		--param NN_GRU_DROPOUT="$(NN_GRU_DROPOUT)" \
		--param NN_TRANSFORMER_D_MODEL="$(NN_TRANSFORMER_D_MODEL)" \
		--param NN_TRANSFORMER_HEADS="$(NN_TRANSFORMER_HEADS)" \
		--param NN_TRANSFORMER_LAYERS="$(NN_TRANSFORMER_LAYERS)" \
		--param NN_TRANSFORMER_FF_DIM="$(NN_TRANSFORMER_FF_DIM)" \
		--param NN_TRANSFORMER_DROPOUT="$(NN_TRANSFORMER_DROPOUT)" \
		--param NN_HIDDEN_LAYERS="$(NN_HIDDEN_LAYERS)" \
		--param NN_LR="$(NN_LR)" \
		--param NN_EPOCHS="$(NN_EPOCHS)" \
		--param NN_BATCH_SIZE="$(NN_BATCH_SIZE)" \
		--param NN_L2="$(NN_L2)" \
		--param NN_CLASS_WEIGHT_MODE="$(NN_CLASS_WEIGHT_MODE)" \
		--param NN_SEED="$(NN_SEED)" \
		--param DECISION_THRESHOLD="$(DECISION_THRESHOLD)" \
		--param THRESHOLD_GRID="$(THRESHOLD_GRID)" \
		--param OPTIMIZE_METRIC="$(OPTIMIZE_METRIC)" \
		--param RAW_DATA="$(RAW_DATA)" \
		--param NN_MODEL_OUT="$(NN_MODEL_OUT)" \
		--param NN_TRAIN_METRICS="$(NN_TRAIN_METRICS)" \
		--param NN_BACKTEST_REPORT="$(NN_BACKTEST_REPORT)" \
		--param NN_PREDICTIONS_OUT="$(NN_PREDICTIONS_OUT)"

lr-save-run: dirs
	$(PYTHON) src/archive_model.py \
		--run-store $(RUNS_ROOT) \
		--current-root $(CURRENT_ROOT) \
		--write-current \
		--family lr \
		--model-type logreg \
		--backend numpy \
		--asset-env $(ASSET_ENV) \
		--trainer-env $(LR_TRAINER_ENV) \
		--model $(MODEL_OUT) \
		--train-metrics $(TRAIN_METRICS) \
		--backtest-report $(BACKTEST_REPORT) \
		--predictions $(PREDICTIONS_OUT) \
		$(addprefix --env-file ,$(ENV_FILES)) \
		--env-file $(LR_TRAINER_ENV) \
		--param ASSET_ENV="$(ASSET_ENV)" \
		--param TRAINER_ENV="$(LR_TRAINER_ENV)" \
		--param SYMBOL="$(SYMBOL)" \
		--param DATA_SOURCE="$(DATA_SOURCE)" \
		--param INTERVAL="$(INTERVAL)" \
		--param START="$(START)" \
		--param END="$(END)" \
		--param SPLIT="$(SPLIT)" \
		--param EDGE="$(EDGE)" \
		--param FEE="$(FEE)" \
		--param THRESHOLD="$(THRESHOLD)" \
		--param SIM_STARTING_CASH="$(SIM_STARTING_CASH)" \
		--param SIM_MIN_INVEST="$(SIM_MIN_INVEST)" \
		--param SIM_MAX_INVEST="$(SIM_MAX_INVEST)" \
		--param SIM_CONFIDENCE_MULTIPLIER="$(SIM_CONFIDENCE_MULTIPLIER)" \
		--param SIM_POSITION_MODE="$(SIM_POSITION_MODE)" \
		--param SIM_SLIPPAGE="$(SIM_SLIPPAGE)" \
		--param SIM_SPREAD_PCT="$(SIM_SPREAD_PCT)" \
		--param POSITION_MODE="$(POSITION_MODE)" \
		--param EXIT_THRESHOLD="$(EXIT_THRESHOLD)" \
		--param MAX_HOLD_BARS="$(MAX_HOLD_BARS)" \
		--param STOP_LOSS="$(STOP_LOSS)" \
		--param TAKE_PROFIT="$(TAKE_PROFIT)" \
		--param LR="$(LR)" \
		--param EPOCHS="$(EPOCHS)" \
		--param L2="$(L2)" \
		--param CLASS_WEIGHT_MODE="$(CLASS_WEIGHT_MODE)" \
		--param DECISION_THRESHOLD="$(DECISION_THRESHOLD)" \
		--param THRESHOLD_GRID="$(THRESHOLD_GRID)" \
		--param OPTIMIZE_METRIC="$(OPTIMIZE_METRIC)" \
		--param RAW_DATA="$(RAW_DATA)" \
		--param FEATURES_DATA="$(FEATURES_DATA)" \
		--param FEATURES_META="$(FEATURES_META)" \
		--param MODEL_OUT="$(MODEL_OUT)" \
		--param TRAIN_METRICS="$(TRAIN_METRICS)" \
		--param BACKTEST_REPORT="$(BACKTEST_REPORT)" \
		--param PREDICTIONS_OUT="$(PREDICTIONS_OUT)"

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
	$(PYTHON) src/archive_model.py \
		--run-store $(RUNS_ROOT) \
		--current-root $(CURRENT_ROOT) \
		--write-current \
		--family xgb \
		--model-type xgboost \
		--backend xgboost \
		--asset-env $(ASSET_ENV) \
		--trainer-env $(XGB_TRAINER_ENV) \
		--model $(XGB_MODEL_OUT) \
		--train-metrics $(XGB_TRAIN_METRICS) \
		--backtest-report $(XGB_BACKTEST_REPORT) \
		--predictions $(XGB_PREDICTIONS_OUT) \
		--sim-report $(XGB_SIM_REPORT) \
		--sim-trades $(XGB_SIM_TRADES) \
		--visualization $(XGB_VISUALIZATION_OUT) \
		--sim-visualization $(XGB_SIM_VISUALIZATION_OUT) \
		$(addprefix --env-file ,$(ENV_FILES)) \
		--param ASSET_ENV="$(ASSET_ENV)" \
		--param TRAINER_ENV="$(XGB_TRAINER_ENV)" \
		--param SYMBOL="$(SYMBOL)" \
		--param DATA_SOURCE="$(DATA_SOURCE)" \
		--param INTERVAL="$(INTERVAL)" \
		--param START="$(START)" \
		--param END="$(END)" \
		--param SPLIT="$(SPLIT)" \
		--param EDGE="$(EDGE)" \
		--param FEE="$(FEE)" \
		--param THRESHOLD="$(THRESHOLD)" \
		--param SIM_STARTING_CASH="$(SIM_STARTING_CASH)" \
		--param SIM_MIN_INVEST="$(SIM_MIN_INVEST)" \
		--param SIM_MAX_INVEST="$(SIM_MAX_INVEST)" \
		--param SIM_CONFIDENCE_MULTIPLIER="$(SIM_CONFIDENCE_MULTIPLIER)" \
		--param SIM_POSITION_MODE="$(SIM_POSITION_MODE)" \
		--param SIM_SLIPPAGE="$(SIM_SLIPPAGE)" \
		--param SIM_SPREAD_PCT="$(SIM_SPREAD_PCT)" \
		--param POSITION_MODE="$(POSITION_MODE)" \
		--param EXIT_THRESHOLD="$(EXIT_THRESHOLD)" \
		--param MAX_HOLD_BARS="$(MAX_HOLD_BARS)" \
		--param STOP_LOSS="$(STOP_LOSS)" \
		--param TAKE_PROFIT="$(TAKE_PROFIT)" \
		--param XGB_N_ESTIMATORS="$(XGB_N_ESTIMATORS)" \
		--param XGB_MAX_DEPTH="$(XGB_MAX_DEPTH)" \
		--param XGB_LEARNING_RATE="$(XGB_LEARNING_RATE)" \
		--param XGB_SUBSAMPLE="$(XGB_SUBSAMPLE)" \
		--param XGB_COLSAMPLE_BYTREE="$(XGB_COLSAMPLE_BYTREE)" \
		--param XGB_MIN_CHILD_WEIGHT="$(XGB_MIN_CHILD_WEIGHT)" \
		--param XGB_REG_LAMBDA="$(XGB_REG_LAMBDA)" \
		--param XGB_REG_ALPHA="$(XGB_REG_ALPHA)" \
		--param XGB_GAMMA="$(XGB_GAMMA)" \
		--param XGB_TREE_METHOD="$(XGB_TREE_METHOD)" \
		--param XGB_DEVICE="$(XGB_DEVICE)" \
		--param XGB_N_JOBS="$(XGB_N_JOBS)" \
		--param XGB_CLASS_WEIGHT_MODE="$(XGB_CLASS_WEIGHT_MODE)" \
		--param XGB_POS_WEIGHT="$(XGB_POS_WEIGHT)" \
		--param XGB_SEED="$(XGB_SEED)" \
		--param DECISION_THRESHOLD="$(DECISION_THRESHOLD)" \
		--param THRESHOLD_GRID="$(THRESHOLD_GRID)" \
		--param OPTIMIZE_METRIC="$(OPTIMIZE_METRIC)" \
		--param RAW_DATA="$(RAW_DATA)" \
		--param XGB_FEATURES_DATA="$(XGB_FEATURES_DATA)" \
		--param XGB_FEATURES_META="$(XGB_FEATURES_META)" \
		--param XGB_MODEL_OUT="$(XGB_MODEL_OUT)" \
		--param XGB_TRAIN_METRICS="$(XGB_TRAIN_METRICS)" \
		--param XGB_BACKTEST_REPORT="$(XGB_BACKTEST_REPORT)" \
		--param XGB_PREDICTIONS_OUT="$(XGB_PREDICTIONS_OUT)" \
		--param XGB_SIM_DEFAULT_TEST_FRACTION="$(XGB_SIM_DEFAULT_TEST_FRACTION)" \
		--param XGB_SIM_REPORT="$(XGB_SIM_REPORT)" \
		--param XGB_SIM_TRADES="$(XGB_SIM_TRADES)"

xgb-visualize: dirs
	$(PYTHON) src/visualize.py \
		--raw-data $(RAW_DATA) \
		--predictions $(XGB_PREDICTIONS_OUT) \
		--output $(XGB_VISUALIZATION_OUT) \
		--threshold $(THRESHOLD) \
		--exit-threshold $(EXIT_THRESHOLD) \
		--trade-mode $(TRADE_MODE) \
		--short-entry-threshold $(SHORT_ENTRY_THRESHOLD) \
		--short-exit-threshold $(SHORT_EXIT_THRESHOLD) \
		--fee $(FEE) \
		--starting-cash $(VIS_STARTING_CASH) \
		--baseline-ma-window $(VIS_BASELINE_MA_WINDOW) \
		--max-browser-points $(VIS_MAX_BROWSER_POINTS) \
		--title "$(SYMBOL) $(INTERVAL) XGBoost Inspection" \
		--nav-home-url /$(REPORTS_INDEX_URL_PATH) \
		--nav-model-url /$(XGB_VISUALIZATION_URL_PATH) \
		--nav-sim-url /$(XGB_SIM_VISUALIZATION_URL_PATH)
	@echo "XGBoost visualization: $(XGB_VISUALIZATION_OUT)"

xgb-sim gxboost-sim: dirs
	$(PYTHON) src/daily_bank_sim.py \
		--predictions $(XGB_PREDICTIONS_OUT) \
		$(SIM_WINDOW_ARGS) \
		--default-test-fraction $(XGB_SIM_DEFAULT_TEST_FRACTION) \
		--position-mode $(SIM_POSITION_MODE) \
		--starting-cash $(SIM_STARTING_CASH) \
		--min-invest $(SIM_MIN_INVEST) \
		--max-invest '$(SIM_MAX_INVEST)' \
		--max-short-invest '$(SIM_MAX_SHORT_INVEST)' \
		--confidence-multiplier $(SIM_CONFIDENCE_MULTIPLIER) \
		--short-confidence-multiplier $(SIM_SHORT_CONFIDENCE_MULTIPLIER) \
		--trade-mode long_only \
		--comparison-trade-mode long_short \
		--threshold $(THRESHOLD) \
		--exit-threshold $(EXIT_THRESHOLD) \
		--short-entry-threshold $(SHORT_ENTRY_THRESHOLD) \
		--short-exit-threshold $(SHORT_EXIT_THRESHOLD) \
		$(if $(filter 1 true yes,$(ALLOW_FLIP_POSITION)),--allow-flip-position,--no-allow-flip-position) \
		--borrow-fee $(BORROW_FEE) \
		--leverage $(LEVERAGE) \
		--liquidation-simulation $(LIQUIDATION_SIMULATION) \
		--fee $(FEE) \
		--max-hold-bars $(MAX_HOLD_BARS) \
		--stop-loss $(STOP_LOSS) \
		--take-profit $(TAKE_PROFIT) \
		--slippage $(SIM_SLIPPAGE) \
		--spread-pct $(SIM_SPREAD_PCT) \
		--report-out $(XGB_SIM_REPORT) \
		--trades-out $(XGB_SIM_TRADES) \
		--comparison-report-out $(XGB_SIM_LONG_SHORT_REPORT) \
		--comparison-trades-out $(XGB_SIM_LONG_SHORT_TRADES)
	@echo "XGBoost bank report: $(XGB_SIM_REPORT)"
	@echo "XGBoost bank trades: $(XGB_SIM_TRADES)"
	@echo "XGBoost simulation default test fraction: $(XGB_SIM_DEFAULT_TEST_FRACTION)"

xgb-sim-visualize: dirs
	$(PYTHON) src/visualize_sim.py \
		--raw-data $(RAW_DATA) \
		--trades $(XGB_SIM_TRADES) \
		--report $(XGB_SIM_REPORT) \
		--comparison-trades $(XGB_SIM_LONG_SHORT_TRADES) \
		--comparison-report $(XGB_SIM_LONG_SHORT_REPORT) \
		--output $(XGB_SIM_VISUALIZATION_OUT) \
		--activity-bucket $(SIM_ACTIVITY_BUCKET) \
		--marker-size-basis $(SIM_MARKER_SIZE_BASIS) \
		--baseline-ma-windows $(SIM_BASELINE_MA_WINDOWS) \
		--title "$(SYMBOL) $(INTERVAL) XGBoost Bank Simulation" \
		--nav-home-url /$(REPORTS_INDEX_URL_PATH) \
		--nav-model-url /$(XGB_VISUALIZATION_URL_PATH) \
		--nav-sim-url /$(XGB_SIM_VISUALIZATION_URL_PATH)
	@echo "XGBoost simulation visualization: $(XGB_SIM_VISUALIZATION_OUT)"

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
	$(MAKE) download \
		INTERVAL=5m \
		START=2026-01-10T00:00:00Z \
		END=2026-01-13T00:00:00Z \
		RAW_DATA=data/downloads/$(DATA_SOURCE)/$(SYMBOL)/5m/preflight_candles.parquet
	$(MAKE) xgb-experiment \
		INTERVAL=5m \
		START=2026-01-10T00:00:00Z \
		END=2026-01-13T00:00:00Z \
		RAW_DATA=data/downloads/$(DATA_SOURCE)/$(SYMBOL)/5m/preflight_candles.parquet \
		XGB_FEATURES_DATA=data/features/xgb/$(DATA_SOURCE)/$(SYMBOL)/5m/preflight_features.parquet \
		XGB_FEATURES_META=data/features/xgb/$(DATA_SOURCE)/$(SYMBOL)/5m/preflight_features.meta.json \
		XGB_MODEL_OUT=models/xgb/$(DATA_SOURCE)/$(SYMBOL)/5m/preflight_model.json \
		XGB_TRAIN_METRICS=models/xgb/$(DATA_SOURCE)/$(SYMBOL)/5m/preflight_train_metrics.json \
		XGB_BACKTEST_REPORT=models/xgb/$(DATA_SOURCE)/$(SYMBOL)/5m/preflight_backtest_report.json \
		XGB_PREDICTIONS_OUT=data/reports/xgb/$(DATA_SOURCE)/$(SYMBOL)/5m/preflight_predictions.parquet \
		XGB_N_ESTIMATORS=20 \
		XGB_MAX_DEPTH=2 \
		XGB_N_JOBS=2 \
		XGB_DEVICE=cuda

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
	$(PYTHON) src/visualize.py \
		--raw-data $(RAW_DATA) \
		--predictions $(STRATEGY_PREDICTIONS_OUT) \
		--output $(STRATEGY_VISUALIZATION_OUT) \
		--threshold $(THRESHOLD) \
		--exit-threshold $(EXIT_THRESHOLD) \
		--trade-mode $(TRADE_MODE) \
		--short-entry-threshold $(SHORT_ENTRY_THRESHOLD) \
		--short-exit-threshold $(SHORT_EXIT_THRESHOLD) \
		--fee $(FEE) \
		--starting-cash $(VIS_STARTING_CASH) \
		--baseline-ma-window $(VIS_BASELINE_MA_WINDOW) \
		--max-browser-points $(VIS_MAX_BROWSER_POINTS) \
		--title "$(SYMBOL) $(INTERVAL) Strategy $(STRATEGY_MODEL_TYPE) Inspection" \
		--nav-home-url /$(REPORTS_INDEX_URL_PATH) \
		--nav-model-url /$(STRATEGY_VISUALIZATION_URL_PATH) \
		--nav-sim-url /$(STRATEGY_SIM_VISUALIZATION_URL_PATH)
	@echo "Strategy visualization: $(STRATEGY_VISUALIZATION_OUT)"

strategy-sim: dirs
	$(PYTHON) src/daily_bank_sim.py \
		--predictions $(STRATEGY_PREDICTIONS_OUT) \
		$(SIM_WINDOW_ARGS) \
		--default-test-fraction $(STRATEGY_SIM_DEFAULT_TEST_FRACTION) \
		--position-mode $(SIM_POSITION_MODE) \
		--starting-cash $(SIM_STARTING_CASH) \
		--min-invest $(SIM_MIN_INVEST) \
		--max-invest '$(SIM_MAX_INVEST)' \
		--max-short-invest '$(SIM_MAX_SHORT_INVEST)' \
		--confidence-multiplier $(SIM_CONFIDENCE_MULTIPLIER) \
		--short-confidence-multiplier $(SIM_SHORT_CONFIDENCE_MULTIPLIER) \
		--trade-mode long_only \
		--comparison-trade-mode long_short \
		--threshold $(THRESHOLD) \
		--exit-threshold $(EXIT_THRESHOLD) \
		--short-entry-threshold $(SHORT_ENTRY_THRESHOLD) \
		--short-exit-threshold $(SHORT_EXIT_THRESHOLD) \
		$(if $(filter 1 true yes,$(ALLOW_FLIP_POSITION)),--allow-flip-position,--no-allow-flip-position) \
		--borrow-fee $(BORROW_FEE) \
		--leverage $(LEVERAGE) \
		--liquidation-simulation $(LIQUIDATION_SIMULATION) \
		--fee $(FEE) \
		--max-hold-bars $(MAX_HOLD_BARS) \
		--stop-loss $(STOP_LOSS) \
		--take-profit $(TAKE_PROFIT) \
		--slippage $(SIM_SLIPPAGE) \
		--spread-pct $(SIM_SPREAD_PCT) \
		--report-out $(STRATEGY_SIM_REPORT) \
		--trades-out $(STRATEGY_SIM_TRADES) \
		--comparison-report-out $(STRATEGY_SIM_LONG_SHORT_REPORT) \
		--comparison-trades-out $(STRATEGY_SIM_LONG_SHORT_TRADES)
	@echo "Strategy bank report: $(STRATEGY_SIM_REPORT)"
	@echo "Strategy bank trades: $(STRATEGY_SIM_TRADES)"

strategy-sim-visualize: dirs
	$(PYTHON) src/visualize_sim.py \
		--raw-data $(RAW_DATA) \
		--trades $(STRATEGY_SIM_TRADES) \
		--report $(STRATEGY_SIM_REPORT) \
		--comparison-trades $(STRATEGY_SIM_LONG_SHORT_TRADES) \
		--comparison-report $(STRATEGY_SIM_LONG_SHORT_REPORT) \
		--output $(STRATEGY_SIM_VISUALIZATION_OUT) \
		--activity-bucket $(SIM_ACTIVITY_BUCKET) \
		--marker-size-basis $(SIM_MARKER_SIZE_BASIS) \
		--baseline-ma-windows $(SIM_BASELINE_MA_WINDOWS) \
		--title "$(SYMBOL) $(INTERVAL) Strategy $(STRATEGY_MODEL_TYPE) Bank Simulation" \
		--nav-home-url /$(REPORTS_INDEX_URL_PATH) \
		--nav-model-url /$(STRATEGY_VISUALIZATION_URL_PATH) \
		--nav-sim-url /$(STRATEGY_SIM_VISUALIZATION_URL_PATH)
	@echo "Strategy simulation visualization: $(STRATEGY_SIM_VISUALIZATION_OUT)"

strategy-graph: strategy-visualize reports-index
	@echo "Open this on your other laptop:"
	@echo "http://$(REPORTS_HOST):$(REPORTS_PORT)/$(STRATEGY_VISUALIZATION_URL_PATH)"
	@echo "Report index:"
	@echo "http://$(REPORTS_HOST):$(REPORTS_PORT)/$(REPORTS_INDEX_URL_PATH)"
	$(MAKE) serve-reports-lan CHECK_URL_PATH=$(STRATEGY_VISUALIZATION_URL_PATH)

strategy-sim-graph: strategy-sim-visualize reports-index
	@echo "Open this on your other laptop:"
	@echo "http://$(REPORTS_HOST):$(REPORTS_PORT)/$(STRATEGY_SIM_VISUALIZATION_URL_PATH)"
	@echo "Report index:"
	@echo "http://$(REPORTS_HOST):$(REPORTS_PORT)/$(REPORTS_INDEX_URL_PATH)"
	$(MAKE) serve-reports-lan CHECK_URL_PATH=$(STRATEGY_SIM_VISUALIZATION_URL_PATH)

strategy-save-run: dirs
	$(PYTHON) src/archive_model.py \
		--run-store $(RUNS_ROOT) \
		--current-root $(CURRENT_ROOT) \
		--write-current \
		--family strategy \
		--model-type $(STRATEGY_MODEL_TYPE) \
		--backend rule_based \
		--asset-env $(ASSET_ENV) \
		--trainer-env $(STRATEGY_TRAINER_ENV) \
		--model $(STRATEGY_MODEL_OUT) \
		--train-metrics $(STRATEGY_TRAIN_METRICS) \
		--backtest-report $(STRATEGY_BACKTEST_REPORT) \
		--predictions $(STRATEGY_PREDICTIONS_OUT) \
		--sim-report $(STRATEGY_SIM_REPORT) \
		--sim-trades $(STRATEGY_SIM_TRADES) \
		--visualization $(STRATEGY_VISUALIZATION_OUT) \
		--sim-visualization $(STRATEGY_SIM_VISUALIZATION_OUT) \
		$(addprefix --env-file ,$(ENV_FILES)) \
		--param ASSET_ENV="$(ASSET_ENV)" \
		--param TRAINER_ENV="$(STRATEGY_TRAINER_ENV)" \
		--param SYMBOL="$(SYMBOL)" \
		--param DATA_SOURCE="$(DATA_SOURCE)" \
		--param INTERVAL="$(INTERVAL)" \
		--param START="$(START)" \
		--param END="$(END)" \
		--param SPLIT="$(SPLIT)" \
		--param EDGE="$(EDGE)" \
		--param FEE="$(FEE)" \
		--param THRESHOLD="$(THRESHOLD)" \
		--param SIM_STARTING_CASH="$(SIM_STARTING_CASH)" \
		--param SIM_MIN_INVEST="$(SIM_MIN_INVEST)" \
		--param SIM_MAX_INVEST="$(SIM_MAX_INVEST)" \
		--param SIM_CONFIDENCE_MULTIPLIER="$(SIM_CONFIDENCE_MULTIPLIER)" \
		--param SIM_POSITION_MODE="$(SIM_POSITION_MODE)" \
		--param SIM_SLIPPAGE="$(SIM_SLIPPAGE)" \
		--param SIM_SPREAD_PCT="$(SIM_SPREAD_PCT)" \
		--param POSITION_MODE="$(POSITION_MODE)" \
		--param EXIT_THRESHOLD="$(EXIT_THRESHOLD)" \
		--param MAX_HOLD_BARS="$(MAX_HOLD_BARS)" \
		--param STOP_LOSS="$(STOP_LOSS)" \
		--param TAKE_PROFIT="$(TAKE_PROFIT)" \
		--param STRATEGY_MODEL_TYPE="$(STRATEGY_MODEL_TYPE)" \
		--param STRATEGY_MA_WINDOW="$(STRATEGY_MA_WINDOW)" \
		--param RAW_DATA="$(RAW_DATA)" \
		--param STRATEGY_MODEL_OUT="$(STRATEGY_MODEL_OUT)" \
		--param STRATEGY_TRAIN_METRICS="$(STRATEGY_TRAIN_METRICS)" \
		--param STRATEGY_BACKTEST_REPORT="$(STRATEGY_BACKTEST_REPORT)" \
		--param STRATEGY_PREDICTIONS_OUT="$(STRATEGY_PREDICTIONS_OUT)" \
		--param STRATEGY_SIM_REPORT="$(STRATEGY_SIM_REPORT)" \
		--param STRATEGY_SIM_TRADES="$(STRATEGY_SIM_TRADES)"

list-runs:
	$(PYTHON) src/archive_model.py --list-runs --run-store $(RUNS_ROOT)

show-current:
	$(PYTHON) src/archive_model.py --show-current --current-root $(CURRENT_ROOT)

nn-save save-nn-model: dirs
	$(PYTHON) src/archive_model.py \
		--archive-root $(NN_ARCHIVE_ROOT) \
		$(NN_ARCHIVE_NAME_ARG) \
		--model $(NN_MODEL_OUT) \
		--train-metrics $(NN_TRAIN_METRICS) \
		--backtest-report $(NN_BACKTEST_REPORT) \
		--predictions $(NN_PREDICTIONS_OUT) \
		--sim-report $(NN_SIM_REPORT) \
		--sim-trades $(NN_SIM_TRADES) \
		--visualization $(NN_VISUALIZATION_OUT) \
		--sim-visualization $(NN_SIM_VISUALIZATION_OUT) \
		$(addprefix --env-file ,$(ENV_FILES)) \
		--include-diff \
		--param SYMBOL="$(SYMBOL)" \
		--param DATA_SOURCE="$(DATA_SOURCE)" \
		--param INTERVAL="$(INTERVAL)" \
		--param START="$(START)" \
		--param END="$(END)" \
		--param SPLIT="$(SPLIT)" \
		--param EDGE="$(EDGE)" \
		--param FEE="$(FEE)" \
		--param THRESHOLD="$(THRESHOLD)" \
		--param POSITION_MODE="$(POSITION_MODE)" \
		--param EXIT_THRESHOLD="$(EXIT_THRESHOLD)" \
		--param MAX_HOLD_BARS="$(MAX_HOLD_BARS)" \
		--param STOP_LOSS="$(STOP_LOSS)" \
		--param TAKE_PROFIT="$(TAKE_PROFIT)" \
		--param NN_BACKEND="$(NN_BACKEND)" \
		--param NN_DEVICE="$(NN_DEVICE)" \
		--param NN_MODEL_TYPE="$(NN_MODEL_TYPE)" \
		--param NN_LOOKBACK="$(NN_LOOKBACK)" \
		--param NN_SEQUENCE_FEATURE_SET="$(NN_SEQUENCE_FEATURE_SET)" \
		--param NN_CNN_FILTERS="$(NN_CNN_FILTERS)" \
		--param NN_CNN_KERNEL_SIZES="$(NN_CNN_KERNEL_SIZES)" \
		--param NN_LSTM_HIDDEN_SIZE="$(NN_LSTM_HIDDEN_SIZE)" \
		--param NN_LSTM_LAYERS="$(NN_LSTM_LAYERS)" \
		--param NN_LSTM_DROPOUT="$(NN_LSTM_DROPOUT)" \
		--param NN_GRU_HIDDEN_SIZE="$(NN_GRU_HIDDEN_SIZE)" \
		--param NN_GRU_LAYERS="$(NN_GRU_LAYERS)" \
		--param NN_GRU_DROPOUT="$(NN_GRU_DROPOUT)" \
		--param NN_TRANSFORMER_D_MODEL="$(NN_TRANSFORMER_D_MODEL)" \
		--param NN_TRANSFORMER_HEADS="$(NN_TRANSFORMER_HEADS)" \
		--param NN_TRANSFORMER_LAYERS="$(NN_TRANSFORMER_LAYERS)" \
		--param NN_TRANSFORMER_FF_DIM="$(NN_TRANSFORMER_FF_DIM)" \
		--param NN_TRANSFORMER_DROPOUT="$(NN_TRANSFORMER_DROPOUT)" \
		--param NN_HIDDEN_LAYERS="$(NN_HIDDEN_LAYERS)" \
		--param NN_LR="$(NN_LR)" \
		--param NN_EPOCHS="$(NN_EPOCHS)" \
		--param NN_BATCH_SIZE="$(NN_BATCH_SIZE)" \
		--param NN_L2="$(NN_L2)" \
		--param NN_CLASS_WEIGHT_MODE="$(NN_CLASS_WEIGHT_MODE)" \
		--param NN_SEED="$(NN_SEED)" \
		--param DECISION_THRESHOLD="$(DECISION_THRESHOLD)" \
		--param THRESHOLD_GRID="$(THRESHOLD_GRID)" \
		--param OPTIMIZE_METRIC="$(OPTIMIZE_METRIC)" \
		--param SIM_START="$(SIM_START)" \
		--param SIM_DURATION="$(SIM_DURATION)" \
		--param SIM_DEFAULT_TEST_FRACTION="$(SIM_DEFAULT_TEST_FRACTION)" \
		--param SIM_STARTING_CASH="$(SIM_STARTING_CASH)" \
		--param SIM_MIN_INVEST="$(SIM_MIN_INVEST)" \
		--param SIM_MAX_INVEST="$(SIM_MAX_INVEST)" \
		--param SIM_CONFIDENCE_MULTIPLIER="$(SIM_CONFIDENCE_MULTIPLIER)" \
		--param SIM_POSITION_MODE="$(SIM_POSITION_MODE)" \
		--param SIM_SLIPPAGE="$(SIM_SLIPPAGE)" \
		--param SIM_SPREAD_PCT="$(SIM_SPREAD_PCT)" \
		--param RAW_DATA="$(RAW_DATA)" \
		--param NN_MODEL_OUT="$(NN_MODEL_OUT)" \
		--param NN_TRAIN_METRICS="$(NN_TRAIN_METRICS)" \
		--param NN_BACKTEST_REPORT="$(NN_BACKTEST_REPORT)" \
		--param NN_PREDICTIONS_OUT="$(NN_PREDICTIONS_OUT)" \
		--param NN_SIM_REPORT="$(NN_SIM_REPORT)" \
		--param NN_SIM_TRADES="$(NN_SIM_TRADES)"

nn-visualize: dirs
	$(PYTHON) src/visualize.py \
		--raw-data $(RAW_DATA) \
		--predictions $(NN_PREDICTIONS_OUT) \
		--output $(NN_VISUALIZATION_OUT) \
		--threshold $(THRESHOLD) \
		--exit-threshold $(EXIT_THRESHOLD) \
		--trade-mode $(TRADE_MODE) \
		--short-entry-threshold $(SHORT_ENTRY_THRESHOLD) \
		--short-exit-threshold $(SHORT_EXIT_THRESHOLD) \
		--fee $(FEE) \
		--starting-cash $(VIS_STARTING_CASH) \
		--baseline-ma-window $(VIS_BASELINE_MA_WINDOW) \
		--max-browser-points $(VIS_MAX_BROWSER_POINTS) \
		--title "$(SYMBOL) $(INTERVAL) Sequence NN Inspection" \
		--nav-home-url /$(REPORTS_INDEX_URL_PATH) \
		--nav-model-url /$(NN_VISUALIZATION_URL_PATH) \
		--nav-sim-url /$(NN_SIM_VISUALIZATION_URL_PATH)
	@echo "Sequence NN visualization: $(NN_VISUALIZATION_OUT)"

nn-sim: dirs
	$(PYTHON) src/daily_bank_sim.py \
		--predictions $(NN_PREDICTIONS_OUT) \
		$(SIM_WINDOW_ARGS) \
		--default-test-fraction $(SIM_DEFAULT_TEST_FRACTION) \
		--position-mode $(SIM_POSITION_MODE) \
		--starting-cash $(SIM_STARTING_CASH) \
		--min-invest $(SIM_MIN_INVEST) \
		--max-invest '$(SIM_MAX_INVEST)' \
		--max-short-invest '$(SIM_MAX_SHORT_INVEST)' \
		--confidence-multiplier $(SIM_CONFIDENCE_MULTIPLIER) \
		--short-confidence-multiplier $(SIM_SHORT_CONFIDENCE_MULTIPLIER) \
		--trade-mode long_only \
		--comparison-trade-mode long_short \
		--threshold $(THRESHOLD) \
		--exit-threshold $(EXIT_THRESHOLD) \
		--short-entry-threshold $(SHORT_ENTRY_THRESHOLD) \
		--short-exit-threshold $(SHORT_EXIT_THRESHOLD) \
		$(if $(filter 1 true yes,$(ALLOW_FLIP_POSITION)),--allow-flip-position,--no-allow-flip-position) \
		--borrow-fee $(BORROW_FEE) \
		--leverage $(LEVERAGE) \
		--liquidation-simulation $(LIQUIDATION_SIMULATION) \
		--fee $(FEE) \
		--max-hold-bars $(MAX_HOLD_BARS) \
		--stop-loss $(STOP_LOSS) \
		--take-profit $(TAKE_PROFIT) \
		--slippage $(SIM_SLIPPAGE) \
		--spread-pct $(SIM_SPREAD_PCT) \
		--report-out $(NN_SIM_REPORT) \
		--trades-out $(NN_SIM_TRADES) \
		--comparison-report-out $(NN_SIM_LONG_SHORT_REPORT) \
		--comparison-trades-out $(NN_SIM_LONG_SHORT_TRADES)
	@echo "Sequence NN daily bank report: $(NN_SIM_REPORT)"
	@echo "Sequence NN daily bank trades: $(NN_SIM_TRADES)"

nn-sim-visualize: dirs
	$(PYTHON) src/visualize_sim.py \
		--raw-data $(RAW_DATA) \
		--trades $(NN_SIM_TRADES) \
		--report $(NN_SIM_REPORT) \
		--comparison-trades $(NN_SIM_LONG_SHORT_TRADES) \
		--comparison-report $(NN_SIM_LONG_SHORT_REPORT) \
		--output $(NN_SIM_VISUALIZATION_OUT) \
		--activity-bucket $(SIM_ACTIVITY_BUCKET) \
		--marker-size-basis $(SIM_MARKER_SIZE_BASIS) \
		--baseline-ma-windows $(SIM_BASELINE_MA_WINDOWS) \
		--title "$(SYMBOL) $(INTERVAL) Sequence NN Bank Simulation" \
		--nav-home-url /$(REPORTS_INDEX_URL_PATH) \
		--nav-model-url /$(NN_VISUALIZATION_URL_PATH) \
		--nav-sim-url /$(NN_SIM_VISUALIZATION_URL_PATH)
	@echo "Sequence NN simulation visualization: $(NN_SIM_VISUALIZATION_OUT)"

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
	$(PYTHON) src/visualize.py \
		--raw-data $(RAW_DATA) \
		--predictions $(PREDICTIONS_OUT) \
		--output $(VISUALIZATION_OUT) \
		--threshold $(THRESHOLD) \
		--exit-threshold $(EXIT_THRESHOLD) \
		--trade-mode $(TRADE_MODE) \
		--short-entry-threshold $(SHORT_ENTRY_THRESHOLD) \
		--short-exit-threshold $(SHORT_EXIT_THRESHOLD) \
		--fee $(FEE) \
		--starting-cash $(VIS_STARTING_CASH) \
		--baseline-ma-window $(VIS_BASELINE_MA_WINDOW) \
		--max-browser-points $(VIS_MAX_BROWSER_POINTS) \
		--title "$(SYMBOL) $(INTERVAL) Model Inspection" \
		--nav-home-url /$(REPORTS_INDEX_URL_PATH) \
		--nav-model-url /$(VISUALIZATION_URL_PATH) \
		--nav-sim-url /$(SIM_VISUALIZATION_URL_PATH)
	@echo "Visualization: $(VISUALIZATION_OUT)"

lr-sim: dirs
	$(PYTHON) src/daily_bank_sim.py \
		--predictions $(PREDICTIONS_OUT) \
		$(SIM_WINDOW_ARGS) \
		--default-test-fraction $(SIM_DEFAULT_TEST_FRACTION) \
		--position-mode $(SIM_POSITION_MODE) \
		--starting-cash $(SIM_STARTING_CASH) \
		--min-invest $(SIM_MIN_INVEST) \
		--max-invest '$(SIM_MAX_INVEST)' \
		--max-short-invest '$(SIM_MAX_SHORT_INVEST)' \
		--confidence-multiplier $(SIM_CONFIDENCE_MULTIPLIER) \
		--short-confidence-multiplier $(SIM_SHORT_CONFIDENCE_MULTIPLIER) \
		--trade-mode long_only \
		--comparison-trade-mode long_short \
		--threshold $(THRESHOLD) \
		--exit-threshold $(EXIT_THRESHOLD) \
		--short-entry-threshold $(SHORT_ENTRY_THRESHOLD) \
		--short-exit-threshold $(SHORT_EXIT_THRESHOLD) \
		$(if $(filter 1 true yes,$(ALLOW_FLIP_POSITION)),--allow-flip-position,--no-allow-flip-position) \
		--borrow-fee $(BORROW_FEE) \
		--leverage $(LEVERAGE) \
		--liquidation-simulation $(LIQUIDATION_SIMULATION) \
		--fee $(FEE) \
		--max-hold-bars $(MAX_HOLD_BARS) \
		--stop-loss $(STOP_LOSS) \
		--take-profit $(TAKE_PROFIT) \
		--slippage $(SIM_SLIPPAGE) \
		--spread-pct $(SIM_SPREAD_PCT) \
		--report-out $(SIM_REPORT) \
		--trades-out $(SIM_TRADES) \
		--comparison-report-out $(SIM_LONG_SHORT_REPORT) \
		--comparison-trades-out $(SIM_LONG_SHORT_TRADES)
	@echo "Daily bank report: $(SIM_REPORT)"
	@echo "Daily bank trades: $(SIM_TRADES)"

lr-sim-visualize: dirs
	$(PYTHON) src/visualize_sim.py \
		--raw-data $(RAW_DATA) \
		--trades $(SIM_TRADES) \
		--report $(SIM_REPORT) \
		--comparison-trades $(SIM_LONG_SHORT_TRADES) \
		--comparison-report $(SIM_LONG_SHORT_REPORT) \
		--output $(SIM_VISUALIZATION_OUT) \
		--activity-bucket $(SIM_ACTIVITY_BUCKET) \
		--marker-size-basis $(SIM_MARKER_SIZE_BASIS) \
		--baseline-ma-windows $(SIM_BASELINE_MA_WINDOWS) \
		--title "$(SYMBOL) $(INTERVAL) Logistic Regression Bank Simulation" \
		--nav-home-url /$(REPORTS_INDEX_URL_PATH) \
		--nav-model-url /$(VISUALIZATION_URL_PATH) \
		--nav-sim-url /$(SIM_VISUALIZATION_URL_PATH)
	@echo "Daily bank visualization: $(SIM_VISUALIZATION_OUT)"

day-sim: sim

serve-reports:
	$(MAKE) start REPORTS_HOST=127.0.0.1

CHECK_URL_PATH ?= $(REPORTS_INDEX_URL_PATH)

start: dirs reports-index
	@if curl -fsS --max-time 2 "http://$(REPORTS_HOST):$(REPORTS_PORT)/api/dashboard/status" >/dev/null 2>&1; then \
		echo "CryptoPred dashboard already running:"; \
		echo "  main: http://$(REPORTS_HOST):$(REPORTS_PORT)/"; \
		echo "  models: http://$(REPORTS_HOST):$(REPORTS_PORT)/models"; \
		echo "  compare: http://$(REPORTS_HOST):$(REPORTS_PORT)/compare"; \
		echo "  reports: http://$(REPORTS_HOST):$(REPORTS_PORT)/reports"; \
		echo "  live: http://$(REPORTS_HOST):$(REPORTS_PORT)/live"; \
		exit 0; \
	fi
	@mkdir -p data/reports
	@echo "Starting CryptoPred dashboard in the background:"
	@echo "  main: http://$(REPORTS_HOST):$(REPORTS_PORT)/"
	@echo "  models: http://$(REPORTS_HOST):$(REPORTS_PORT)/models"
	@echo "  compare: http://$(REPORTS_HOST):$(REPORTS_PORT)/compare"
	@echo "  reports: http://$(REPORTS_HOST):$(REPORTS_PORT)/reports"
	@echo "  live: http://$(REPORTS_HOST):$(REPORTS_PORT)/live"
	@echo "  live Docker target: $(LIVE_PUBLIC_URL)"
	@setsid $(PYTHON) src/dashboard_server.py \
		--host $(REPORTS_HOST) \
		--port $(REPORTS_PORT) \
		--reports-root data/reports \
		--root . \
		--live-url $(LIVE_LOCAL_URL) \
		--live-public-url $(LIVE_PUBLIC_URL) \
		> data/reports/dashboard_server.log 2>&1 < /dev/null &
	@sleep 1
	@curl -fsS --max-time 3 "http://$(REPORTS_HOST):$(REPORTS_PORT)/api/dashboard/status" >/dev/null \
		&& echo "Dashboard started. Logs: data/reports/dashboard_server.log" \
		|| { echo "Dashboard did not respond. Check data/reports/dashboard_server.log"; exit 1; }

start-fg: dirs reports-index
	@echo "Starting CryptoPred dashboard:"
	@echo "  main: http://$(REPORTS_HOST):$(REPORTS_PORT)/"
	@echo "  models: http://$(REPORTS_HOST):$(REPORTS_PORT)/models"
	@echo "  compare: http://$(REPORTS_HOST):$(REPORTS_PORT)/compare"
	@echo "  reports: http://$(REPORTS_HOST):$(REPORTS_PORT)/reports"
	@echo "  live: http://$(REPORTS_HOST):$(REPORTS_PORT)/live"
	@echo "  live Docker target: $(LIVE_PUBLIC_URL)"
	$(PYTHON) src/dashboard_server.py \
		--host $(REPORTS_HOST) \
		--port $(REPORTS_PORT) \
		--reports-root data/reports \
		--root . \
		--live-url $(LIVE_LOCAL_URL) \
		--live-public-url $(LIVE_PUBLIC_URL)

stop:
	@echo "Stopping any server listening on port $(REPORTS_PORT)..."
	@PIDS="$$( \
		if command -v lsof >/dev/null 2>&1; then \
			lsof -tiTCP:$(REPORTS_PORT) -sTCP:LISTEN 2>/dev/null; \
		elif command -v fuser >/dev/null 2>&1; then \
			fuser -n tcp $(REPORTS_PORT) 2>/dev/null; \
		elif command -v ss >/dev/null 2>&1; then \
			ss -ltnp "sport = :$(REPORTS_PORT)" 2>/dev/null | sed -n 's/.*pid=\([0-9][0-9]*\).*/\1/p'; \
		else \
			echo ""; \
		fi \
		| sort -u)"; \
	if [ -z "$$PIDS" ]; then \
		echo "No process is listening on port $(REPORTS_PORT)."; \
		exit 0; \
	fi; \
	echo "Killing PID(s): $$PIDS"; \
	kill $$PIDS 2>/dev/null || true; \
	sleep 1; \
	STILL="$$( \
		if command -v lsof >/dev/null 2>&1; then \
			lsof -tiTCP:$(REPORTS_PORT) -sTCP:LISTEN 2>/dev/null; \
		elif command -v fuser >/dev/null 2>&1; then \
			fuser -n tcp $(REPORTS_PORT) 2>/dev/null; \
		elif command -v ss >/dev/null 2>&1; then \
			ss -ltnp "sport = :$(REPORTS_PORT)" 2>/dev/null | sed -n 's/.*pid=\([0-9][0-9]*\).*/\1/p'; \
		else \
			echo ""; \
		fi \
		| sort -u)"; \
	if [ -n "$$STILL" ]; then \
		echo "Force killing PID(s): $$STILL"; \
		kill -9 $$STILL 2>/dev/null || true; \
	fi; \
	echo "Stopped server on port $(REPORTS_PORT)."

serve-reports-lan:
	@if curl -fsS --max-time 2 "http://$(REPORTS_HOST):$(REPORTS_PORT)/api/dashboard/status" >/dev/null 2>&1; then \
		echo "Dashboard server already appears to be running on $(REPORTS_HOST):$(REPORTS_PORT)."; \
	elif curl -fsS --max-time 2 "http://$(REPORTS_HOST):$(REPORTS_PORT)/$(CHECK_URL_PATH)" >/dev/null 2>&1; then \
		echo "Port $(REPORTS_PORT) is already serving reports, but not the new dashboard."; \
		echo "Stop the old server, then run: make start"; \
		exit 1; \
	else \
		$(MAKE) start; \
	fi

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
	@echo "Open this on your other laptop:"
	@echo "http://$(REPORTS_HOST):$(REPORTS_PORT)/$(NN_VISUALIZATION_URL_PATH)"
	@echo "Report index:"
	@echo "http://$(REPORTS_HOST):$(REPORTS_PORT)/$(REPORTS_INDEX_URL_PATH)"
	$(MAKE) serve-reports-lan CHECK_URL_PATH=$(NN_VISUALIZATION_URL_PATH)

nn-serve-lan: nn-graph

lan: nn-graph

graph: nn-graph

serve-lan: graph

nn-sim-graph: nn-sim-visualize reports-index
	@echo "Open this on your other laptop:"
	@echo "http://$(REPORTS_HOST):$(REPORTS_PORT)/$(NN_SIM_VISUALIZATION_URL_PATH)"
	@echo "Report index:"
	@echo "http://$(REPORTS_HOST):$(REPORTS_PORT)/$(REPORTS_INDEX_URL_PATH)"
	$(MAKE) serve-reports-lan CHECK_URL_PATH=$(NN_SIM_VISUALIZATION_URL_PATH)

lr-graph: lr-visualize reports-index
	@echo "Open this on your other laptop:"
	@echo "http://$(REPORTS_HOST):$(REPORTS_PORT)/$(VISUALIZATION_URL_PATH)"
	@echo "Report index:"
	@echo "http://$(REPORTS_HOST):$(REPORTS_PORT)/$(REPORTS_INDEX_URL_PATH)"
	$(MAKE) serve-reports-lan CHECK_URL_PATH=$(VISUALIZATION_URL_PATH)

lr-serve-lan: lr-graph

lr-sim-graph: lr-sim-visualize reports-index
	@echo "Open this on your other laptop:"
	@echo "http://$(REPORTS_HOST):$(REPORTS_PORT)/$(SIM_VISUALIZATION_URL_PATH)"
	@echo "Report index:"
	@echo "http://$(REPORTS_HOST):$(REPORTS_PORT)/$(REPORTS_INDEX_URL_PATH)"
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
	$(MAKE) download \
		INTERVAL=5m \
		START=2026-01-10T00:00:00Z \
		END=2026-01-13T00:00:00Z \
		RAW_DATA=data/downloads/$(DATA_SOURCE)/$(SYMBOL)/5m/preflight_candles.parquet
	$(MAKE) experiment \
		INTERVAL=5m \
		START=2026-01-10T00:00:00Z \
		END=2026-01-13T00:00:00Z \
		RAW_DATA=data/downloads/$(DATA_SOURCE)/$(SYMBOL)/5m/preflight_candles.parquet \
		NN_MODEL_OUT=models/nn/cnn/$(DATA_SOURCE)/$(SYMBOL)/5m/preflight_model.npz \
		NN_TRAIN_METRICS=models/nn/cnn/$(DATA_SOURCE)/$(SYMBOL)/5m/preflight_train_metrics.json \
		NN_BACKTEST_REPORT=models/nn/cnn/$(DATA_SOURCE)/$(SYMBOL)/5m/preflight_backtest_report.json \
		NN_PREDICTIONS_OUT=data/reports/nn/cnn/$(DATA_SOURCE)/$(SYMBOL)/5m/preflight_predictions.parquet \
		NN_MODEL_TYPE=cnn \
		NN_LOOKBACK=20 \
		NN_CNN_FILTERS=4,8 \
		NN_CNN_KERNEL_SIZES=3,3 \
		NN_EPOCHS=2 \
		NN_HIDDEN_LAYERS=8 \
		NN_BATCH_SIZE=128

lr-preflight:
	$(MAKE) download \
		INTERVAL=5m \
		START=2026-01-10T00:00:00Z \
		END=2026-01-13T00:00:00Z \
		RAW_DATA=data/downloads/$(DATA_SOURCE)/$(SYMBOL)/5m/preflight_candles.parquet
	$(MAKE) lr-experiment \
		INTERVAL=5m \
		START=2026-01-10T00:00:00Z \
		END=2026-01-13T00:00:00Z \
		RAW_DATA=data/downloads/$(DATA_SOURCE)/$(SYMBOL)/5m/preflight_candles.parquet \
		FEATURES_DATA=data/features/lr/$(DATA_SOURCE)/$(SYMBOL)/5m/preflight_features.parquet \
		FEATURES_META=data/features/lr/$(DATA_SOURCE)/$(SYMBOL)/5m/preflight_features.meta.json \
		MODEL_OUT=models/lr/$(DATA_SOURCE)/$(SYMBOL)/5m/preflight_logreg.npz \
		TRAIN_METRICS=models/lr/$(DATA_SOURCE)/$(SYMBOL)/5m/preflight_train_metrics.json \
		BACKTEST_REPORT=models/lr/$(DATA_SOURCE)/$(SYMBOL)/5m/preflight_backtest_report.json \
		PREDICTIONS_OUT=data/reports/lr/$(DATA_SOURCE)/$(SYMBOL)/5m/preflight_predictions.parquet

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
REAL_CASH_ASSET ?= USD
REAL_BASE_ASSET ?= SOL
COINBASE_PRODUCT_ID ?= SOL-USD
COINBASE_API_KEY ?=
COINBASE_API_SECRET ?=
COINBASE_TIMEOUT ?= 10
REAL_ARM_TOKEN ?=
REAL_ORDER_STATUS_POLLS ?= 5
REAL_ORDER_STATUS_DELAY_SECONDS ?= 0.75
HOST_UID := $(shell id -u)
HOST_GID := $(shell id -g)

live-sync:
	@mkdir -p live_sim/state
	@if [ -f "$(LIVE_MODEL_SOURCE)" ]; then \
		cp "$(LIVE_MODEL_SOURCE)" live_sim/state/model.npz; \
	elif [ -f live_sim/state/model.npz ]; then \
		echo "Selected live model does not exist yet: $(LIVE_MODEL_SOURCE)"; \
		echo "Keeping existing live_sim/state/model.npz until retraining activates a fresh model."; \
	else \
		echo "Missing selected live model: $(LIVE_MODEL_SOURCE)"; \
		echo "Run make train or make experiment first, or keep an existing live_sim/state/model.npz."; \
		exit 1; \
	fi
	@printf '%s\n' \
		'SYMBOL=$(SYMBOL)' \
		'INTERVAL=$(INTERVAL)' \
		'BINANCE_BASE_URL=https://api.binance.com' \
		'' \
		'STARTING_CASH=$(LIVE_STARTING_CASH)' \
		'FEE=$(FEE)' \
		'SLIPPAGE=$(LIVE_SLIPPAGE)' \
		'ENTRY_THRESHOLD=$(THRESHOLD)' \
		'EXIT_THRESHOLD=$(EXIT_THRESHOLD)' \
		'TRADE_MODE=$(TRADE_MODE)' \
		'SHORT_ENTRY_THRESHOLD=$(SHORT_ENTRY_THRESHOLD)' \
		'SHORT_EXIT_THRESHOLD=$(SHORT_EXIT_THRESHOLD)' \
		'STOP_LOSS=$(STOP_LOSS)' \
		'TAKE_PROFIT=$(TAKE_PROFIT)' \
		'MAX_HOLD_BARS=$(MAX_HOLD_BARS)' \
		'MAX_INVEST=$(LIVE_MAX_INVEST)' \
		'MAX_SHORT_INVEST=$(LIVE_MAX_SHORT_INVEST)' \
		'ALLOW_FLIP_POSITION=$(ALLOW_FLIP_POSITION)' \
		'BORROW_FEE=$(BORROW_FEE)' \
		'LEVERAGE=$(LEVERAGE)' \
		'LIQUIDATION_SIMULATION=$(LIQUIDATION_SIMULATION)' \
		'MIN_INVEST=$(LIVE_MIN_INVEST)' \
		'CONFIDENCE_MULTIPLIER=$(LIVE_CONFIDENCE_MULTIPLIER)' \
		'' \
		'EXECUTION_MODE=$(EXECUTION_MODE)' \
		'REAL_TRADING_ENABLED=$(REAL_TRADING_ENABLED)' \
		'REAL_REQUIRE_MANUAL_ARM=$(REAL_REQUIRE_MANUAL_ARM)' \
		'REAL_QUICK_ARM_ENABLED=$(REAL_QUICK_ARM_ENABLED)' \
		'REAL_MAX_TOTAL_USD=$(REAL_MAX_TOTAL_USD)' \
		'REAL_MAX_ORDER_USD=$(REAL_MAX_ORDER_USD)' \
		'REAL_MIN_ORDER_USD=$(REAL_MIN_ORDER_USD)' \
		'REAL_PORTFOLIO_MODE=$(REAL_PORTFOLIO_MODE)' \
		'REAL_CASH_ASSET=$(REAL_CASH_ASSET)' \
		'REAL_BASE_ASSET=$(REAL_BASE_ASSET)' \
		'COINBASE_PRODUCT_ID=$(COINBASE_PRODUCT_ID)' \
		'COINBASE_API_KEY=$(COINBASE_API_KEY)' \
		'COINBASE_API_SECRET=$(COINBASE_API_SECRET)' \
		'COINBASE_TIMEOUT=$(COINBASE_TIMEOUT)' \
		'REAL_ARM_TOKEN=$(REAL_ARM_TOKEN)' \
		'REAL_ORDER_STATUS_POLLS=$(REAL_ORDER_STATUS_POLLS)' \
		'REAL_ORDER_STATUS_DELAY_SECONDS=$(REAL_ORDER_STATUS_DELAY_SECONDS)' \
		'' \
		'MODEL_PATH=/app/state/model.npz' \
		'DB_PATH=/app/state/live_sim.db' \
		'RESET_ON_START=false' \
		'ALLOW_RESET_API=false' \
		'POLL_ON_START=true' \
		'POLL_DELAY_SECONDS=8' \
		'KLINE_LIMIT_BUFFER=8' \
		'CATCHUP_ENABLED=$(LIVE_CATCHUP_ENABLED)' \
		'CATCHUP_SPREAD_PCT=$(LIVE_CATCHUP_SPREAD_PCT)' \
		'CATCHUP_MAX_BARS=$(LIVE_CATCHUP_MAX_BARS)' \
		'CATCHUP_RETRY_SECONDS=$(LIVE_CATCHUP_RETRY_SECONDS)' \
		'' \
		'RETRAIN_ENABLED=true' \
		'RETRAIN_TIME_UTC=04:00' \
		'RETRAIN_FREQUENCY=$(LIVE_RETRAIN_FREQUENCY)' \
		'RETRAIN_LOOKBACK_DAYS=$(LIVE_RETRAIN_LOOKBACK_DAYS)' \
		'RETRAIN_TRAIN_START=$(LIVE_RETRAIN_TRAIN_START)' \
		'RETRAIN_TRAIN_END=$(LIVE_RETRAIN_TRAIN_END)' \
		'RETRAIN_ON_START=false' \
		'RETRAIN_KEEP_RUNS=10' \
		'RETRAIN_CACHE_DIR=$(LIVE_RETRAIN_CACHE_DIR)' \
		'TRAINING_RUNS_DIR=/app/state/training_runs' \
		'' \
		'TRAIN_MODEL_TYPE=$(LIVE_TRAIN_MODEL_TYPE)' \
		'TRAIN_BACKEND=$(LIVE_TRAIN_BACKEND)' \
		'TRAIN_DEVICE=$(LIVE_TRAIN_DEVICE)' \
		'TRAIN_LOOKBACK=$(LIVE_TRAIN_LOOKBACK)' \
		'TRAIN_SEQUENCE_FEATURE_SET=$(LIVE_TRAIN_SEQUENCE_FEATURE_SET)' \
		'TRAIN_EDGE=$(LIVE_TRAIN_EDGE)' \
		'TRAIN_SPLIT=$(SPLIT)' \
		'TRAIN_USE_FULL_WINDOW=$(LIVE_TRAIN_USE_FULL_WINDOW)' \
		'TRAIN_CNN_FILTERS=$(NN_CNN_FILTERS)' \
		'TRAIN_CNN_KERNEL_SIZES=$(NN_CNN_KERNEL_SIZES)' \
		'TRAIN_LSTM_HIDDEN_SIZE=$(NN_LSTM_HIDDEN_SIZE)' \
		'TRAIN_LSTM_LAYERS=$(NN_LSTM_LAYERS)' \
		'TRAIN_LSTM_DROPOUT=$(NN_LSTM_DROPOUT)' \
		'TRAIN_GRU_HIDDEN_SIZE=$(NN_GRU_HIDDEN_SIZE)' \
		'TRAIN_GRU_LAYERS=$(NN_GRU_LAYERS)' \
		'TRAIN_GRU_DROPOUT=$(NN_GRU_DROPOUT)' \
		'TRAIN_TRANSFORMER_D_MODEL=$(NN_TRANSFORMER_D_MODEL)' \
		'TRAIN_TRANSFORMER_HEADS=$(NN_TRANSFORMER_HEADS)' \
		'TRAIN_TRANSFORMER_LAYERS=$(NN_TRANSFORMER_LAYERS)' \
		'TRAIN_TRANSFORMER_FF_DIM=$(NN_TRANSFORMER_FF_DIM)' \
		'TRAIN_TRANSFORMER_DROPOUT=$(NN_TRANSFORMER_DROPOUT)' \
		'TRAIN_HIDDEN_LAYERS=$(NN_HIDDEN_LAYERS)' \
		'TRAIN_LR=$(NN_LR)' \
		'TRAIN_EPOCHS=$(NN_EPOCHS)' \
		'TRAIN_BATCH_SIZE=$(NN_BATCH_SIZE)' \
		'TRAIN_L2=$(NN_L2)' \
		'TRAIN_DECISION_THRESHOLD=$(THRESHOLD)' \
		'TRAIN_THRESHOLD_GRID=$(THRESHOLD_GRID)' \
		'TRAIN_OPTIMIZE_METRIC=$(OPTIMIZE_METRIC)' \
		'TRAIN_CLASS_WEIGHT_MODE=$(NN_CLASS_WEIGHT_MODE)' \
		'TRAIN_SEED=$(NN_SEED)' \
		'' \
		'HOME=/app/state' \
		'USER=cryptopred' \
		'LOGNAME=cryptopred' \
		'XDG_CACHE_HOME=/app/state/.cache' \
		'TORCH_HOME=/app/state/.cache/torch' \
		'TORCHINDUCTOR_CACHE_DIR=/app/state/.cache/torchinductor' \
		'TRITON_CACHE_DIR=/app/state/.cache/triton' \
		'TORCHDYNAMO_DISABLE=1' \
		'' \
		'HOST=0.0.0.0' \
		'PORT=8080' \
		'HOST_PORT=$(LIVE_HOST_PORT)' \
		'HOST_UID=$(HOST_UID)' \
		'HOST_GID=$(HOST_GID)' \
		> live_sim/.env
	@$(PYTHON) src/save_live_env_snapshot.py \
		--source-env live_sim/.env \
		--active-env $(LIVE_ENV_ACTIVE) \
		--model-env $(LIVE_MODEL_ENV) \
		--snapshot-root $(LIVE_ENV_SNAPSHOT_ROOT) \
		--model-type $(LIVE_MODEL_TYPE) \
		--data-source $(DATA_SOURCE) \
		--symbol $(SYMBOL) \
		--interval $(INTERVAL) \
		--sim-report $(LIVE_SIM_REPORT) \
		$(addprefix --env-file ,$(ENV_FILES)) \
		--param ASSET_ENV="$(ASSET_ENV)" \
		--param TRAINER_ENV="$(TRAINER_ENV)" \
		--param SYMBOL="$(SYMBOL)" \
		--param DATA_SOURCE="$(DATA_SOURCE)" \
		--param INTERVAL="$(INTERVAL)" \
		--param START="$(START)" \
		--param END="$(END)" \
		--param SPLIT="$(SPLIT)" \
		--param EDGE="$(EDGE)" \
		--param FEE="$(FEE)" \
		--param THRESHOLD="$(THRESHOLD)" \
		--param EXIT_THRESHOLD="$(EXIT_THRESHOLD)" \
		--param MAX_HOLD_BARS="$(MAX_HOLD_BARS)" \
		--param STOP_LOSS="$(STOP_LOSS)" \
		--param TAKE_PROFIT="$(TAKE_PROFIT)" \
		--param LIVE_MODEL_TYPE="$(LIVE_MODEL_TYPE)" \
		--param LIVE_MODEL_SOURCE="$(LIVE_MODEL_SOURCE)" \
		--param LIVE_RETRAIN_FREQUENCY="$(LIVE_RETRAIN_FREQUENCY)" \
		--param LIVE_RETRAIN_TRAIN_START="$(LIVE_RETRAIN_TRAIN_START)" \
		--param LIVE_RETRAIN_TRAIN_END="$(LIVE_RETRAIN_TRAIN_END)" \
		--param LIVE_RETRAIN_LOOKBACK_DAYS="$(LIVE_RETRAIN_LOOKBACK_DAYS)" \
		--param LIVE_TRAIN_MODEL_TYPE="$(LIVE_TRAIN_MODEL_TYPE)" \
		--param LIVE_TRAIN_BACKEND="$(LIVE_TRAIN_BACKEND)" \
		--param LIVE_TRAIN_DEVICE="$(LIVE_TRAIN_DEVICE)" \
		--param LIVE_TRAIN_LOOKBACK="$(LIVE_TRAIN_LOOKBACK)" \
		--param LIVE_TRAIN_SEQUENCE_FEATURE_SET="$(LIVE_TRAIN_SEQUENCE_FEATURE_SET)" \
		--param LIVE_TRAIN_EDGE="$(LIVE_TRAIN_EDGE)" \
		--param LIVE_TRAIN_USE_FULL_WINDOW="$(LIVE_TRAIN_USE_FULL_WINDOW)" \
		--param LIVE_STARTING_CASH="$(LIVE_STARTING_CASH)" \
		--param LIVE_MIN_INVEST="$(LIVE_MIN_INVEST)" \
		--param LIVE_MAX_INVEST="$(LIVE_MAX_INVEST)" \
		--param LIVE_CONFIDENCE_MULTIPLIER="$(LIVE_CONFIDENCE_MULTIPLIER)" \
		--param LIVE_SLIPPAGE="$(LIVE_SLIPPAGE)" \
		--param LIVE_CATCHUP_ENABLED="$(LIVE_CATCHUP_ENABLED)" \
		--param LIVE_CATCHUP_SPREAD_PCT="$(LIVE_CATCHUP_SPREAD_PCT)" \
		--param LIVE_CATCHUP_MAX_BARS="$(LIVE_CATCHUP_MAX_BARS)" \
		--param LIVE_CATCHUP_RETRY_SECONDS="$(LIVE_CATCHUP_RETRY_SECONDS)" \
		--param EXECUTION_MODE="$(EXECUTION_MODE)" \
		--param REAL_TRADING_ENABLED="$(REAL_TRADING_ENABLED)" \
		--param REAL_MAX_TOTAL_USD="$(REAL_MAX_TOTAL_USD)" \
		--param REAL_MAX_ORDER_USD="$(REAL_MAX_ORDER_USD)" \
		--param REAL_PORTFOLIO_MODE="$(REAL_PORTFOLIO_MODE)" \
		--param REAL_CASH_ASSET="$(REAL_CASH_ASSET)" \
		--param REAL_BASE_ASSET="$(REAL_BASE_ASSET)" \
		--param COINBASE_PRODUCT_ID="$(COINBASE_PRODUCT_ID)" \
		--param NN_MODEL_TYPE="$(NN_MODEL_TYPE)" \
		--param NN_BACKEND="$(NN_BACKEND)" \
		--param NN_DEVICE="$(NN_DEVICE)" \
		--param NN_LOOKBACK="$(NN_LOOKBACK)" \
		--param NN_SEQUENCE_FEATURE_SET="$(NN_SEQUENCE_FEATURE_SET)" \
		--param NN_CNN_FILTERS="$(NN_CNN_FILTERS)" \
		--param NN_CNN_KERNEL_SIZES="$(NN_CNN_KERNEL_SIZES)" \
		--param NN_LSTM_HIDDEN_SIZE="$(NN_LSTM_HIDDEN_SIZE)" \
		--param NN_LSTM_LAYERS="$(NN_LSTM_LAYERS)" \
		--param NN_LSTM_DROPOUT="$(NN_LSTM_DROPOUT)" \
		--param NN_GRU_HIDDEN_SIZE="$(NN_GRU_HIDDEN_SIZE)" \
		--param NN_GRU_LAYERS="$(NN_GRU_LAYERS)" \
		--param NN_GRU_DROPOUT="$(NN_GRU_DROPOUT)" \
		--param NN_TRANSFORMER_D_MODEL="$(NN_TRANSFORMER_D_MODEL)" \
		--param NN_TRANSFORMER_HEADS="$(NN_TRANSFORMER_HEADS)" \
		--param NN_TRANSFORMER_LAYERS="$(NN_TRANSFORMER_LAYERS)" \
		--param NN_TRANSFORMER_FF_DIM="$(NN_TRANSFORMER_FF_DIM)" \
		--param NN_TRANSFORMER_DROPOUT="$(NN_TRANSFORMER_DROPOUT)" \
		--param NN_HIDDEN_LAYERS="$(NN_HIDDEN_LAYERS)" \
		--param NN_LR="$(NN_LR)" \
		--param NN_EPOCHS="$(NN_EPOCHS)" \
		--param NN_BATCH_SIZE="$(NN_BATCH_SIZE)" \
		--param NN_L2="$(NN_L2)" \
		--param NN_CLASS_WEIGHT_MODE="$(NN_CLASS_WEIGHT_MODE)" \
		--param NN_SEED="$(NN_SEED)" \
		--param NN_SIM_REPORT="$(NN_SIM_REPORT)"
	@echo "Synced live_sim from main config:"
	@echo "  model:    requested $(LIVE_MODEL_SOURCE); active copy live_sim/state/model.npz"
	@echo "  env:      $(LIVE_MODEL_ENV) -> $(LIVE_ENV_ACTIVE) -> live_sim/.env"
	@echo "  snapshot: $(LIVE_ENV_SNAPSHOT_ROOT)"
	@echo "  symbol:   $(SYMBOL)"
	@echo "  interval: $(INTERVAL)"
	@echo "  catch-up: enabled=$(LIVE_CATCHUP_ENABLED), historical spread=$(LIVE_CATCHUP_SPREAD_PCT), max bars=$(LIVE_CATCHUP_MAX_BARS) (0=unlimited)"
	@echo "  real:     execution=$(EXECUTION_MODE), enabled=$(REAL_TRADING_ENABLED), mode=$(REAL_PORTFOLIO_MODE), product=$(COINBASE_PRODUCT_ID), cash=$(REAL_CASH_ASSET), base=$(REAL_BASE_ASSET)"
	@echo "  retrain:  every $(LIVE_RETRAIN_FREQUENCY), window $(LIVE_RETRAIN_TRAIN_START) to $(LIVE_RETRAIN_TRAIN_END) as rolling duration"
	@echo "  train:    $(LIVE_TRAIN_MODEL_TYPE), backend=$(LIVE_TRAIN_BACKEND), device=$(LIVE_TRAIN_DEVICE), lookback=$(LIVE_TRAIN_LOOKBACK), feature_set=$(LIVE_TRAIN_SEQUENCE_FEATURE_SET), filters=$(NN_CNN_FILTERS), gru_hidden=$(NN_GRU_HIDDEN_SIZE), lstm_hidden=$(NN_LSTM_HIDDEN_SIZE), transformer=$(NN_TRANSFORMER_D_MODEL)x$(NN_TRANSFORMER_LAYERS)/h$(NN_TRANSFORMER_HEADS), hidden=$(NN_HIDDEN_LAYERS)"

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
