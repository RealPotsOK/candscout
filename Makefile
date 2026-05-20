SHELL := /bin/bash

# ---------- Environment defaults ----------
ENV_FILES := \
	env/runtime.env \
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

GENERATED_DIRS := $(sort $(dir \
	$(RAW_DATA) \
	$(FEATURES_DATA) $(FEATURES_META) \
	$(MODEL_OUT) $(TRAIN_METRICS) $(BACKTEST_REPORT) $(PREDICTIONS_OUT) \
	$(NN_MODEL_OUT) $(NN_TRAIN_METRICS) $(NN_BACKTEST_REPORT) $(NN_PREDICTIONS_OUT) $(NN_VISUALIZATION_OUT) \
	$(DIAG_REPORT) $(DIAG_TABLE) $(DIAG_TEST_PREDICTIONS) $(SWEEP_OUTPUT_DIR)/ $(VISUALIZATION_OUT) \
	$(SIM_REPORT) $(SIM_TRADES) $(SIM_VISUALIZATION_OUT) $(NN_SIM_REPORT) $(NN_SIM_TRADES) $(NN_SIM_VISUALIZATION_OUT) \
))


.PHONY: help install dirs download nn train backtest experiment diagnostic sweep visualize sim sim-visualize sim-graph day-sim serve-reports graph serve-lan preflight run clean smoke repo-status
.PHONY: github-check github-init github-commit github-create-private github-push github-publish
.PHONY: nn-train nn-backtest nn-experiment nn-diagnostic nn-visualize nn-sim nn-sim-visualize nn-sim-graph nn-graph nn-serve-lan nn-servre-lan
.PHONY: lr-features lr-train lr-backtest lr-experiment lr-diagnostic lr-sweep lr-visualize lr-sim lr-sim-visualize lr-sim-graph lr-graph lr-serve-lan lr-preflight

run:
	$(MAKE) download
	$(MAKE) experiment
	$(MAKE) graph

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
	@echo "  make graph    - serve sequence neural net report on LAN"
	@echo "  make serve-lan    - alias for make graph"
	@echo "  make run          - run default sequence neural net pipeline -> serve LAN report"
	@echo "  make nn-graph - serve sequence neural net report on LAN"
	@echo "  make lr-experiment - run old logistic-regression feature -> train -> backtest"
	@echo "  make lr-visualize - create old logistic-regression HTML visualization"
	@echo "  make lr-sim       - simulate old logistic-regression predictions"
	@echo "  make lr-diagnostic - build old logistic-regression diagnostic report"
	@echo "  make lr-sweep     - run old logistic-regression preset/edge sweep"
	@echo "  make serve-reports - serve reports at http://127.0.0.1:8000/"
	@echo "  make smoke        - compile source files without downloading/training"
	@echo "  make repo-status  - show git state and generated artifact sizes"
	@echo "  make github-publish - create private GitHub repo and push source"
	@echo "  make preflight    - 3-day quick pipeline sanity run"
	@echo ""
	@echo "Common overrides example:"
	@echo "  make experiment NN_MODEL_TYPE=cnn NN_LOOKBACK=50 NN_CNN_FILTERS=16,32 NN_CNN_KERNEL_SIZES=5,3 NN_EPOCHS=25"
	@echo "  make experiment NN_MODEL_TYPE=cnn NN_LOOKBACK=20 NN_CNN_FILTERS=8,16 NN_CNN_KERNEL_SIZES=3,3"
	@echo "  make experiment NN_MODEL_TYPE=mlp"
	@echo "  make experiment SYMBOL=SOLUSDT INTERVAL=5m START=2026-04-18T00:00:00Z END=2026-05-18T00:00:00Z EDGE=0.0005 SPLIT=0.8 FEE=0.0001 THRESHOLD=0.55"
	@echo "  make download DATA_SOURCE=yahoo SYMBOL=AAPL INTERVAL=5m"
	@echo "  make download DATA_SOURCE=yahoo RANDOM_STOCK=1 INTERVAL=1d"
	@echo "  make sim"
	@echo "  make sim-visualize"
	@echo "  make sim SIM_START=2026-01-12 SIM_DURATION=1D"
	@echo "  make sim-visualize SIM_ACTIVITY_BUCKET=hour SIM_MARKER_SIZE_BASIS=usd"
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

sim: nn-sim

sim-visualize: nn-sim-visualize

sim-graph: nn-sim-graph

install:
	$(PIP) install -r requirements.txt

smoke:
	$(PYTHON) -m py_compile $(shell find src -name '*.py' | sort)

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
		--out $(RAW_DATA)

lr-features: dirs
	$(PYTHON) src/features.py \
		--input $(RAW_DATA) \
		--output $(FEATURES_DATA) \
		--meta-out $(FEATURES_META) \
		--edge $(EDGE) \
		--interval $(INTERVAL) \
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
		--optimize-metric $(OPTIMIZE_METRIC) \
		--class-weight-mode $(CLASS_WEIGHT_MODE)

lr-backtest: dirs
	$(PYTHON) src/backtest.py \
		--features $(FEATURES_DATA) \
		--model $(MODEL_OUT) \
		--fee $(FEE) \
		--threshold $(THRESHOLD) \
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

nn-train: dirs
	$(PYTHON) src/train_sequence_nn.py \
		--raw-data $(RAW_DATA) \
		--model-out $(NN_MODEL_OUT) \
		--metrics-out $(NN_TRAIN_METRICS) \
		--model-type $(NN_MODEL_TYPE) \
		--lookback $(NN_LOOKBACK) \
		--edge $(EDGE) \
		--split $(SPLIT) \
		--cnn-filters $(NN_CNN_FILTERS) \
		--cnn-kernel-sizes $(NN_CNN_KERNEL_SIZES) \
		--hidden-layers $(NN_HIDDEN_LAYERS) \
		--lr $(NN_LR) \
		--epochs $(NN_EPOCHS) \
		--batch-size $(NN_BATCH_SIZE) \
		--l2 $(NN_L2) \
		--decision-threshold $(DECISION_THRESHOLD) \
		--threshold-grid $(THRESHOLD_GRID) \
		--optimize-metric $(OPTIMIZE_METRIC) \
		--class-weight-mode $(NN_CLASS_WEIGHT_MODE) \
		--seed $(NN_SEED)

nn-backtest: dirs
	$(PYTHON) src/backtest_sequence_nn.py \
		--raw-data $(RAW_DATA) \
		--model $(NN_MODEL_OUT) \
		--fee $(FEE) \
		--threshold $(THRESHOLD) \
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

nn-diagnostic:
	@echo "Sequence NN diagnostics are written during train/backtest:"
	@echo "Train metrics: $(NN_TRAIN_METRICS)"
	@echo "Backtest:      $(NN_BACKTEST_REPORT)"
	@echo "Predictions:   $(NN_PREDICTIONS_OUT)"

nn-visualize: dirs
	$(PYTHON) src/visualize.py \
		--raw-data $(RAW_DATA) \
		--predictions $(NN_PREDICTIONS_OUT) \
		--output $(NN_VISUALIZATION_OUT) \
		--threshold $(THRESHOLD) \
		--fee $(FEE) \
		--starting-cash $(VIS_STARTING_CASH) \
		--title "$(SYMBOL) $(INTERVAL) Sequence NN Inspection"
	@echo "Sequence NN visualization: $(NN_VISUALIZATION_OUT)"

nn-sim: dirs
	$(PYTHON) src/daily_bank_sim.py \
		--predictions $(NN_PREDICTIONS_OUT) \
		$(SIM_WINDOW_ARGS) \
		--starting-cash $(SIM_STARTING_CASH) \
		--min-invest $(SIM_MIN_INVEST) \
		--max-invest $(SIM_MAX_INVEST) \
		--threshold $(THRESHOLD) \
		--fee $(FEE) \
		--report-out $(NN_SIM_REPORT) \
		--trades-out $(NN_SIM_TRADES)
	@echo "Sequence NN daily bank report: $(NN_SIM_REPORT)"
	@echo "Sequence NN daily bank trades: $(NN_SIM_TRADES)"

nn-sim-visualize: dirs
	$(PYTHON) src/visualize_sim.py \
		--raw-data $(RAW_DATA) \
		--trades $(NN_SIM_TRADES) \
		--report $(NN_SIM_REPORT) \
		--output $(NN_SIM_VISUALIZATION_OUT) \
		--activity-bucket $(SIM_ACTIVITY_BUCKET) \
		--marker-size-basis $(SIM_MARKER_SIZE_BASIS) \
		--title "$(SYMBOL) $(INTERVAL) Sequence NN Bank Simulation"
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
		--fee $(FEE) \
		--starting-cash $(VIS_STARTING_CASH) \
		--title "$(SYMBOL) $(INTERVAL) Model Inspection"
	@echo "Visualization: $(VISUALIZATION_OUT)"

lr-sim: dirs
	$(PYTHON) src/daily_bank_sim.py \
		--predictions $(PREDICTIONS_OUT) \
		$(SIM_WINDOW_ARGS) \
		--starting-cash $(SIM_STARTING_CASH) \
		--min-invest $(SIM_MIN_INVEST) \
		--max-invest $(SIM_MAX_INVEST) \
		--threshold $(THRESHOLD) \
		--fee $(FEE) \
		--report-out $(SIM_REPORT) \
		--trades-out $(SIM_TRADES)
	@echo "Daily bank report: $(SIM_REPORT)"
	@echo "Daily bank trades: $(SIM_TRADES)"

lr-sim-visualize: dirs
	$(PYTHON) src/visualize_sim.py \
		--raw-data $(RAW_DATA) \
		--trades $(SIM_TRADES) \
		--report $(SIM_REPORT) \
		--output $(SIM_VISUALIZATION_OUT) \
		--activity-bucket $(SIM_ACTIVITY_BUCKET) \
		--marker-size-basis $(SIM_MARKER_SIZE_BASIS) \
		--title "$(SYMBOL) $(INTERVAL) Logistic Regression Bank Simulation"
	@echo "Daily bank visualization: $(SIM_VISUALIZATION_OUT)"

day-sim: sim

serve-reports:
	cd data/reports && $(PYTHON) -m http.server $(REPORTS_PORT) --bind 127.0.0.1

nn-graph: nn-visualize
	@echo "Open this on your other laptop:"
	@echo "http://$(REPORTS_HOST):$(REPORTS_PORT)/$(NN_VISUALIZATION_URL_PATH)"
	$(PYTHON) -m http.server $(REPORTS_PORT) --bind $(REPORTS_HOST) --directory data/reports

nn-serve-lan: nn-graph

nn-servre-lan: nn-graph

graph: nn-graph

serve-lan: graph

nn-sim-graph: nn-sim-visualize
	@echo "Open this on your other laptop:"
	@echo "http://$(REPORTS_HOST):$(REPORTS_PORT)/$(NN_SIM_VISUALIZATION_URL_PATH)"
	$(PYTHON) -m http.server $(REPORTS_PORT) --bind $(REPORTS_HOST) --directory data/reports

lr-graph: lr-visualize
	@echo "Open this on your other laptop:"
	@echo "http://$(REPORTS_HOST):$(REPORTS_PORT)/$(VISUALIZATION_URL_PATH)"
	$(PYTHON) -m http.server $(REPORTS_PORT) --bind $(REPORTS_HOST) --directory data/reports

lr-serve-lan: lr-graph

lr-sim-graph: lr-sim-visualize
	@echo "Open this on your other laptop:"
	@echo "http://$(REPORTS_HOST):$(REPORTS_PORT)/$(SIM_VISUALIZATION_URL_PATH)"
	$(PYTHON) -m http.server $(REPORTS_PORT) --bind $(REPORTS_HOST) --directory data/reports

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
