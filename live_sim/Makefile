SHELL := /bin/bash

ENV_FILE ?= env/active.env
COMPOSE ?= docker compose --env-file $(ENV_FILE)
DEFAULT_MODEL ?= ../models/nn/cnn/binance/SOLUSDT/15m/model.npz
MODEL_COPY ?= state/model.npz
DB_FILE ?= state/live_sim.db

.PHONY: help setup build up down logs reset shell test config status real-check retrain-status retrain-now clear-stale-retrain-lock cache-status update-model

help:
	@echo "Targets:"
	@echo "  make setup  - create .env if missing and copy DEFAULT_MODEL into state/model.npz"
	@echo "  make build  - build Docker image"
	@echo "  make up     - start the live paper-trading server"
	@echo "  make down   - stop the server"
	@echo "  make logs   - follow server logs"
	@echo "  make reset  - stop server and remove SQLite runtime state, keeping model.npz"
	@echo "  make shell  - open shell in the running container"
	@echo "  make test   - run local unit tests"
	@echo "  make real-check - validate Coinbase config/balances without placing orders"
	@echo "  make retrain-status - show retraining scheduler status"
	@echo "  make retrain-now    - manually start retrain if ALLOW_RESET_API=true"
	@echo "  make clear-stale-retrain-lock - remove retrain.lock only when no retrain is running"
	@echo "  make cache-status - show persisted retrain candle caches"
	@echo "  make update-model   - manually retrain and activate a fresh model without enabling the API"
	@echo ""
	@echo "Open: http://127.0.0.1:8080/ or http://192.168.2.197:8080/"

setup:
	@mkdir -p state env/models
	@if [ ! -f "$(ENV_FILE)" ]; then cp .env.example "$(ENV_FILE)"; echo "Created live_sim/$(ENV_FILE)"; fi
	@cp "$(ENV_FILE)" .env
	@if [ ! -f "$(MODEL_COPY)" ]; then \
		if [ -f "$(DEFAULT_MODEL)" ]; then \
			cp "$(DEFAULT_MODEL)" "$(MODEL_COPY)"; \
			echo "Copied model to live_sim/$(MODEL_COPY)"; \
		else \
			echo "Missing default model: $(DEFAULT_MODEL)"; \
			echo "Run from repo root first: make live-sync or pass DEFAULT_MODEL=/path/to/model.npz"; \
			exit 1; \
		fi; \
	else \
		echo "Model already exists at live_sim/$(MODEL_COPY)"; \
	fi

build: setup
	$(COMPOSE) build

up: setup
	$(COMPOSE) up -d --build
	@echo "Live paper-trading dashboard: http://127.0.0.1:8080/"
	@echo "LAN dashboard: http://192.168.2.197:8080/"

config:
	$(COMPOSE) config

status:
	$(COMPOSE) ps

real-check: setup
	$(COMPOSE) run --rm live-sim python -m app.real_check

retrain-status:
	@curl -sS http://127.0.0.1:$${HOST_PORT:-8080}/api/retraining

retrain-now:
	@curl -sS -X POST http://127.0.0.1:$${HOST_PORT:-8080}/api/retrain-now

clear-stale-retrain-lock:
	@lock="state/retrain.lock"; \
	if [ ! -f "$$lock" ]; then \
		echo "No retrain lock found."; \
		exit 0; \
	fi; \
	host_port="$$(awk -F= '/^HOST_PORT=/{print $$2}' "$(ENV_FILE)" 2>/dev/null | tail -1)"; \
	host_port="$${host_port:-8080}"; \
	status_file="$$(mktemp)"; \
	api_running="unknown"; \
	if curl -fsS --max-time 2 "http://127.0.0.1:$${host_port}/api/retraining" > "$$status_file" 2>/dev/null; then \
		if grep -q '"running"[[:space:]]*:[[:space:]]*true' "$$status_file"; then \
			api_running="true"; \
		else \
			api_running="false"; \
		fi; \
	fi; \
	rm -f "$$status_file"; \
	run_containers="$$(docker ps --filter 'name=live_sim-live-sim-run' --format '{{.Names}}' 2>/dev/null || true)"; \
	if [ "$$api_running" = "true" ] || [ -n "$$run_containers" ]; then \
		echo "Retraining appears to be running; refusing to remove $$lock."; \
		echo "api_running=$$api_running"; \
		if [ -n "$$run_containers" ]; then echo "run_containers=$$run_containers"; fi; \
		exit 1; \
	fi; \
	echo "Removing stale retrain lock: $$lock"; \
	cat "$$lock" || true; \
	rm -f "$$lock"

cache-status:
	@echo "Persistent retrain candle caches under live_sim/state/downloads:"
	@if [ -d state/downloads ]; then \
		find state/downloads -type f -name 'cache.parquet' -printf '%p %k KiB\n' | sort; \
	else \
		echo "No persistent candle cache directory yet."; \
	fi

update-model: setup clear-stale-retrain-lock
	@echo "Running one-off retrain container with the current live_sim/.env..."
	$(COMPOSE) run --rm live-sim python -m app.retrain_once

down:
	$(COMPOSE) down

logs:
	$(COMPOSE) logs -f

reset: down
	@rm -f $(DB_FILE) $(DB_FILE)-shm $(DB_FILE)-wal
	@echo "Removed live_sim SQLite state. Model copy was kept."

shell:
	$(COMPOSE) exec live-sim /bin/bash

test:
	PYTHONPATH=app:../src python -m unittest discover -s tests -v
