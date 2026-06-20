# CryptoPred Live Paper-Trading Server

This folder is isolated from the training/backtest pipeline. It runs a Dockerized paper-trading simulator for `SOLUSDT` using a saved sequence model such as MLP, CNN, GRU, or LSTM.

By default it does not place real orders and does not require API keys. Optional Coinbase Advanced Trade spot execution is available only when explicitly enabled, manually armed, and capped to a small real-money limit.

## Quick Start

From the repo root, sync live_sim to the current main model/config and start Docker:

```bash
make live-up
```

This copies the selected live model into `live_sim/state/model.npz` and writes `live_sim/.env` from the main `env/*.env` values.

To sync without starting Docker:

```bash
make live-sync
```

Direct live_sim commands still work, but they use the already-written `live_sim/.env`.

```bash
cd live_sim
make setup
make up
make logs
```

Open:

```text
http://127.0.0.1:8080/
http://192.168.2.197:8080/
```

## Config

`make setup` creates `live_sim/.env` from `.env.example` and copies the saved model to `live_sim/state/model.npz`.

Important settings:

```text
STARTING_CASH=100
ENTRY_THRESHOLD=0.55
EXIT_THRESHOLD=0.48
TRADE_MODE=long_only
SHORT_ENTRY_THRESHOLD=0.55
SHORT_EXIT_THRESHOLD=0.48
FEE=0.0001
MAX_INVEST=m
MAX_SHORT_INVEST=m
MIN_INVEST=1
CONFIDENCE_MULTIPLIER=1.0
```

Short positions are paper-only and unleveraged. They enter at bid, cover at ask,
and can include `BORROW_FEE` per held bar.

`MAX_INVEST` supports only safe forms:

```text
m      all currently available cash
0.5m   half currently available cash
m/2    half currently available cash
25     fixed 25 USDT
```

`MAX_INVEST` is now the cap, not always the exact buy size. Actual live
buy size uses the same confidence sizing as the historical bank simulator:

```text
confidence = (prob_up - ENTRY_THRESHOLD) / (1 - ENTRY_THRESHOLD)
investment = MIN_INVEST + (MAX_INVEST - MIN_INVEST) * sqrt(confidence * CONFIDENCE_MULTIPLIER)
```

The result is clipped to available cash. This keeps the live bot from going
all-in on predictions that barely clear the entry threshold.

## Commands

```bash
make build   # build Docker image
make up      # start server
make down    # stop server
make logs    # follow logs
make reset   # delete SQLite state, keeping model copy
make test    # run local unit tests
```

Runtime state stays in `live_sim/state/` and is intentionally ignored by git.

## Optional Coinbase Real Spot Trading

Real trading v1 is intentionally narrow:

```text
Coinbase Advanced Trade only
SOL-USD spot only
long-only buy/sell only
no shorts, no borrowing, no margin, no leverage
hard default cap: $20 total, $5 per order
```

Safe defaults:

```text
EXECUTION_MODE=paper
REAL_TRADING_ENABLED=false
REAL_REQUIRE_MANUAL_ARM=true
REAL_MAX_TOTAL_USD=20
REAL_MAX_ORDER_USD=5
REAL_MIN_ORDER_USD=1
COINBASE_PRODUCT_ID=SOL-USD
COINBASE_API_KEY=
COINBASE_API_SECRET=
REAL_ARM_TOKEN=
```

Put Coinbase credentials only in ignored env files, such as repo-root
`env/local.env` or `live_sim/env/active.env`. Ed25519 base64 secrets are easier
to use in `.env` files than multiline PEM strings.

Preflight without placing orders:

```bash
make live-sync EXECUTION_MODE=coinbase_live REAL_TRADING_ENABLED=true
make live-real-check
```

If the check passes, start the live server and arm from the dashboard:

```bash
make live-up
```

The dashboard requires `REAL_ARM_TOKEN` and this exact confirmation text before
real orders are allowed:

```text
ARM REAL TRADING SOL-USD MAX 20
```

The bot still runs paper accounting separately. Real order attempts and fills are
stored in SQLite tables `real_trade_state` and `real_orders`.

To close only SOL that this bot tracked as bought, use the dashboard
`Flatten Bot SOL` button. It requires this exact confirmation:

```text
FLATTEN REAL SOL-USD
```

## Startup Catch-Up

When the computer or container has been offline, the server resumes from the
last persisted model decision in SQLite. Before normal polling begins, it:

1. Downloads every missing completed candle from Binance.
2. Rebuilds the model input for each candle in chronological order.
3. Replays the normal buy, hold, and sell logic.
4. Saves the missing decisions, trades, and account equity snapshots.

Historical order-book snapshots are not available from the public kline API.
Catch-up therefore approximates bid/ask execution around each candle close:

```text
CATCHUP_ENABLED=true
CATCHUP_SPREAD_PCT=0.00015
CATCHUP_MAX_BARS=0
CATCHUP_RETRY_SECONDS=60
```

`CATCHUP_MAX_BARS=0` means unlimited. Set a positive value to make startup fail
the catch-up step instead of replaying an unexpectedly large gap. The active
model at startup is used for the entire missing period. If downloading or
replay fails, normal polling stays paused and catch-up retries instead of
skipping the missing interval.

## Model Selection

From the repo root, choose which latest model artifact the live bot starts with:

```bash
make live-sync LIVE_MODEL_TYPE=cnn
make live-sync LIVE_MODEL_TYPE=gru TRAINER_ENV=env/trainers/gru_torch.env
make live-sync LIVE_MODEL_TYPE=lstm TRAINER_ENV=env/trainers/lstm_torch.env
make live-sync LIVE_MODEL_TYPE=transformer TRAINER_ENV=env/trainers/transformer_torch.env
make live-sync LIVE_MODEL_SOURCE=models/nn/lstm/binance/SOLUSDT/3m/model.npz
```

`LIVE_MODEL_SOURCE` wins when set. Otherwise the path is:

```text
models/nn/<LIVE_MODEL_TYPE>/<DATA_SOURCE>/<SYMBOL>/<INTERVAL>/model.npz
```

GRU and LSTM require PyTorch inside Docker, so `live_sim/requirements.txt` includes `torch`.

## Rolling Retraining

The container can retrain a fresh model on a rolling data window:

```text
RETRAIN_ENABLED=true
RETRAIN_TIME_UTC=04:00
RETRAIN_FREQUENCY=1d
RETRAIN_LOOKBACK_DAYS=365
RETRAIN_TRAIN_START=
RETRAIN_TRAIN_END=
TRAINING_RUNS_DIR=/app/state/training_runs
RETRAIN_CACHE_DIR=/app/state/downloads
```

`RETRAIN_FREQUENCY` supports:

```text
10h   every 10 hours
3d    every 3 days
1w    every 1 week
1m    every 1 month; m means month here, not minute
```

For training-window length, either use `RETRAIN_LOOKBACK_DAYS`, or set both date fields:

```text
RETRAIN_TRAIN_START=2025-01-01T00:00:00Z
RETRAIN_TRAIN_END=2026-01-01T00:00:00Z
```

Those dates define the duration. The scheduled retrain then downloads a rolling window of that same duration ending at retrain time. In the example above, each retrain uses the latest 365 days.

When you run `make live-sync` from the repo root and do not set live-specific dates, it writes the current model's normal `START` and `END` into `RETRAIN_TRAIN_START` and `RETRAIN_TRAIN_END`, so the live retrain window matches the individual model training window.

Live retraining uses the full selected window by default:

```text
TRAIN_USE_FULL_WINDOW=true
```

That means the activated live model is trained on the whole rolling window, not a train/test split. The metrics file still reports in-sample diagnostics for sanity only.

At the scheduled time, the server downloads Binance candles for the rolling window, trains a fresh sequence model with the `TRAIN_*` settings, archives the full run under `live_sim/state/training_runs/<timestamp>/`, then atomically replaces `live_sim/state/model.npz`.

Downloaded candles are also stored in a persistent cache:

```text
live_sim/state/downloads/binance/<SYMBOL>/<INTERVAL>/cache.parquet
```

Each training run still gets its own `candles.parquet`, but future retrains reuse the persistent cache and download only missing candle gaps. If the persistent cache does not exist yet, the retrainer tries to seed it from the newest compatible old training run before calling Binance.

Check cache files from the repo root:

```bash
make live-cache-status
```

The live bot keeps using the old model while the new model trains. After the replacement succeeds, the live bot reloads the new model automatically on the next prediction cycle.

Status endpoints:

```text
GET /api/retraining
GET /api/status
```

Manual retrain is intentionally disabled unless you set:

```text
ALLOW_RESET_API=true
```

Then call:

```bash
curl -X POST http://127.0.0.1:8080/api/retrain-now
```

The recommended manual command does not require enabling that API:

```bash
make update-model
```

From the repo root, use:

```bash
make update-model
```
