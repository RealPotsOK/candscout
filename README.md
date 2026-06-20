# cryptopred

`cryptopred` is a crypto/stock prediction research project. It downloads candle data, builds features or candle sequences, trains separate model families, backtests them, visualizes predictions, and runs bank-account simulations.

This is not live trading software. It is an offline research/backtest tool.

## Setup

```bash
python3 -m venv .venv
make install
make smoke
```

Current default workflow:

```text
symbol: SOLUSDT
interval: 3m
model: Torch sequence CNN
data source: Binance spot public candles
```

## Common Commands

Download candles:

```bash
make download
```

Downloaded requested candles are written to:

```text
data/downloads/<source>/<symbol>/<interval>/candles.parquet
```

The reusable long-term candle cache is written beside it:

```text
data/downloads/<source>/<symbol>/<interval>/cache.parquet
```

For Binance, `make download` checks that cache first and only downloads missing gaps. For example, if the cache has May through September and you request March through October, it downloads March through May and September through October, then writes the requested March-through-October range to `candles.parquet`.

Train and backtest the selected sequence model:

```bash
make experiment
```

Successful experiments are saved into `models/runs/` by default, while the latest compatibility model path is still updated.

Build the HTML visualization:

```bash
make visualize
```

Run the bank-account simulator from existing predictions:

```bash
make sim
```

Create a separate bank simulation visualization:

```bash
make sim-visualize
```

Serve the visualization on the LAN:

```bash
make graph
```

Start the main dashboard on the LAN:

```bash
make start
```

Stop the dashboard:

```bash
make stop
```

Use foreground mode only when you want server logs attached to the terminal:

```bash
make start-fg
```

Default URLs:

```text
main dashboard: http://192.168.2.197:8000/
live Docker sim: http://192.168.2.197:8080/
```

The dashboard serves existing generated report HTML files, adds Models and Compare Models pages, and includes a Live page that checks/controls the separate Docker live simulation.

The live Docker simulator persists its account in SQLite and automatically
replays missing completed candles when the computer starts after being offline.
See [live_sim/README.md](live_sim/README.md) for catch-up spread and safety
settings.

Dashboard model form settings are saved to:

```text
data/reports/dashboard_settings.json
```

Shared settings such as symbol, interval, dates, split, edge, fee, and thresholds carry across model tabs. Model-specific settings such as CNN filters or XGBoost depth are saved under that model only.

The current conservative starting profile is documented in
[`docs/recommended_settings.md`](docs/recommended_settings.md).

Create the report server home page:

```bash
make reports-index
```

Serve the simulation visualization on the LAN:

```bash
make sim-graph
```

Run the older logistic-regression pipeline:

```bash
make lr-experiment
```

Run the XGBoost tabular-feature pipeline:

```bash
make xgb-experiment
make xgb-visualize
make xgb-sim
make xgb-sim-visualize
```

If XGBoost is not installed yet, run `make install` first.

Simulation sizing is controlled by:

```bash
SIM_MIN_INVEST=100
SIM_MAX_INVEST=5000
SIM_CONFIDENCE_MULTIPLIER=3
```

The multiplier scales confidence before sizing. Higher values make near-threshold predictions use larger buys, while `SIM_MIN_INVEST` and `SIM_MAX_INVEST` still cap the trade amount.

Run rule-based strategy models:

```bash
make strategy-experiment STRATEGY_MODEL_TYPE=buy_hold
make strategy-experiment STRATEGY_MODEL_TYPE=prev_movement
make strategy-experiment STRATEGY_MODEL_TYPE=ma STRATEGY_MA_WINDOW=20
make strategy-visualize
make strategy-sim
make strategy-sim-visualize
```

`prev_movement` predicts that the next candle repeats the direction of the latest completed candle.

Models are shown in the dashboard from least complex to most complex:

```text
Buy and Hold
Previous Movement
MA Strategy
Logistic Regression
XGBoost
MLP
CNN
GRU
LSTM
Transformer
```

List saved model runs and current pointers:

```bash
make list-runs
make show-current
```

## Useful Overrides

Default settings live in section files under `env/`:

```text
env/assets/*.env
env/trainers/*.env
env/core.env
env/sequence_nn.env
env/logistic_regression.env
env/simulation.env
env/features.env
env/paths.env
env/runtime.env
```

For personal defaults, copy `env/local.env.example` to `env/local.env`. That file is ignored by git and is loaded after the tracked env files.

Command-line overrides still win:

```bash
make experiment SYMBOL=BTCUSDT NN_EPOCHS=50
```

Select asset and trainer presets:

```bash
make experiment ASSET_ENV=env/assets/btcusdt_5m.env TRAINER_ENV=env/trainers/cnn_torch.env
make experiment ASSET_ENV=env/assets/stock_yahoo_1h.env TRAINER_ENV=env/trainers/cnn_torch.env
make experiment TRAINER_ENV=env/trainers/mlp_torch.env
make experiment TRAINER_ENV=env/trainers/gru_torch.env
make experiment TRAINER_ENV=env/trainers/lstm_torch.env
make experiment TRAINER_ENV=env/trainers/transformer_torch.env
make xgb-experiment ASSET_ENV=env/assets/btcusdt_5m.env
make xgb-sim XGB_SIM_DEFAULT_TEST_FRACTION=0.05
```

Run without saving a canonical model-run snapshot:

```bash
make experiment AUTO_SAVE_RUN=0
```

Use BTC instead of SOL:

```bash
make download SYMBOL=BTCUSDT RAW_DATA=data/downloads/binance/BTCUSDT/5m/candles.parquet
make experiment SYMBOL=BTCUSDT RAW_DATA=data/downloads/binance/BTCUSDT/5m/candles.parquet
```

Try a smaller CNN:

```bash
make experiment NN_LOOKBACK=20 NN_CNN_FILTERS=8,16 NN_CNN_KERNEL_SIZES=3,3 NN_EPOCHS=50
```

Use the sequence MLP fallback:

```bash
make experiment TRAINER_ENV=env/trainers/mlp_torch.env
```

Use the GRU recurrent model:

```bash
make experiment TRAINER_ENV=env/trainers/gru_torch.env
```

GRU keeps the ordered candle sequence and uses update/reset gates. It is less complex than LSTM and usually trains somewhat faster because it has fewer gates and parameters.

Use the Transformer attention model:

```bash
make experiment TRAINER_ENV=env/trainers/transformer_torch.env
```

The Transformer uses learned positional embeddings and self-attention across the ordered candle window. The recommended preset is intentionally compact to reduce overfitting and GPU memory use.

Prediction-only research with no fee model:

```bash
make experiment EDGE=0 FEE=0
```

Simulation visualization options:

```bash
make sim-visualize SIM_ACTIVITY_BUCKET=hour SIM_MARKER_SIZE_BASIS=usd
make sim-visualize SIM_ACTIVITY_BUCKET=day SIM_MARKER_SIZE_BASIS=coin
```

## Trade Modes

`TRADE_MODE=long_only` preserves the original behavior and remains the default. Use
`short_only` or `long_short` to test unleveraged paper shorts from the same
`prob_up` output.

```bash
make nn-sim TRADE_MODE=long_short \
  THRESHOLD=0.55 EXIT_THRESHOLD=0.48 \
  SHORT_ENTRY_THRESHOLD=0.45 SHORT_EXIT_THRESHOLD=0.52
```

`THRESHOLD` and `EXIT_THRESHOLD` remain the compatible Make variable names for
long entry and long exit. Shorts reserve cash collateral, enter at bid, cover at
ask, and can include `BORROW_FEE` per held bar. `LEVERAGE=1` is enforced; live
shorting is paper simulation only.

## Model Runs

Latest-output paths remain for compatibility:

```text
models/nn/<model_type>/<source>/<symbol>/<interval>/model.npz
models/lr/<source>/<symbol>/<interval>/logreg.npz
models/xgb/<source>/<symbol>/<interval>/model.json
models/sim/xgb/<source>/<symbol>/<interval>/bank_report.json
```

Saved model history is stored under:

```text
models/runs/<run_id>/
models/current/<source>/<symbol>/<interval>/
```

Useful run-store commands:

```bash
make save-run
make lr-save-run
make xgb-save-run
make list-runs
make show-current
```

## GitHub Publishing

Generated data, reports, trained models, and `.venv/` are ignored by git. Only source, docs, requirements, workflow files, and Makefile should be committed.

Publish to a private GitHub repo named `cryptopred`:

```bash
make github-check
make github-publish
```

If `make github-check` fails, install GitHub CLI, then authenticate:

```bash
gh auth login
```

Then rerun:

```bash
make github-publish
```
