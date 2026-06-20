# Project Structure

This project intentionally keeps a simple script-based layout instead of a Python package refactor.

## Source-Controlled Files

```text
src/                         Python scripts and model code
src/model_registry/          Dashboard model tabs and command mappings
Makefile                     Main command interface
requirements.txt             Python dependencies
env/                         Sectioned Make defaults
env/assets/                  Asset/data presets
env/trainers/                Training-method presets
README.md                    Project overview and commands
docs/                        Project notes
.github/workflows/smoke.yml  GitHub smoke check
.gitignore                   Keeps generated artifacts out of git
```

## Generated Outputs

New runs use organized output paths:

```text
data/downloads/<source>/<symbol>/<interval>/candles.parquet
data/downloads/<source>/<symbol>/<interval>/cache.parquet
data/features/lr/<source>/<symbol>/<interval>/
data/reports/lr/<source>/<symbol>/<interval>/
data/reports/nn/<model_type>/<source>/<symbol>/<interval>/
data/reports/xgb/<source>/<symbol>/<interval>/
data/reports/sim/<pipeline>/<source>/<symbol>/<interval>/
models/lr/<source>/<symbol>/<interval>/
models/nn/<model_type>/<source>/<symbol>/<interval>/
models/xgb/<source>/<symbol>/<interval>/
models/sim/<pipeline>/<source>/<symbol>/<interval>/
models/runs/<run_id>/
models/current/<source>/<symbol>/<interval>/
```

These generated folders are ignored by git.

`candles.parquet` is the current requested download range used by training. `cache.parquet` is the reusable source/symbol/interval candle store used to avoid re-downloading already available Binance candles.

`models/nn/...`, `models/lr/...`, and `models/xgb/...` are compatibility paths for the latest run. They are overwritten by new experiments in that model family.

`models/runs/<run_id>/` is the canonical saved model history. Each saved run contains:

```text
artifacts/model.npz or artifacts/model.json
artifacts/train_metrics.json
artifacts/backtest_report.json
artifacts/predictions.parquet
env/
manifest.json
README.md
```

`models/current/.../*.json` points to the latest saved run for a source/symbol/interval/model combination.

## Model Families

## Environment Defaults

The Makefile loads tracked section files from `env/`:

```text
assets/*.env             Asset/data presets loaded before generic defaults
trainers/*.env           Training presets loaded before generic defaults
runtime.env              Python and project/GitHub defaults
core.env                 Symbol, interval, dates, fees, thresholds
sequence_nn.env          MLP/CNN/GRU/LSTM/Transformer sequence-model settings
logistic_regression.env  LR training settings
trainers/xgboost.env     XGBoost tree-model settings
trainers/strategy.env    Rule strategy defaults
simulation.env           Bank simulator settings and sim outputs
features.env             LR feature engineering settings
paths.env                Organized generated output paths
```

Optional personal overrides go in `env/local.env`, which is ignored by git. Command-line overrides still take precedence.

Default preset selection:

```text
ASSET_ENV=env/assets/solusdt_3m.env
TRAINER_ENV=env/trainers/cnn_torch.env
```

Examples:

```bash
make experiment ASSET_ENV=env/assets/btcusdt_5m.env
make experiment TRAINER_ENV=env/trainers/mlp_torch.env
make experiment ASSET_ENV=env/assets/stock_yahoo_1h.env TRAINER_ENV=env/trainers/cnn_torch.env
```

`nn` targets use the sequence neural-network pipeline:

```text
make train
make backtest
make experiment
make visualize
make sim
make sim-visualize
```

`lr` targets use the older feature-based logistic regression pipeline:

```text
make lr-features
make lr-train
make lr-backtest
make lr-experiment
```

`xgb` targets use XGBoost on the same leakage-safe tabular features as LR, but write to separate `xgb` paths:

```text
make xgb-features
make xgb-train
make xgb-backtest
make xgb-experiment
make xgb-visualize
make xgb-sim
make xgb-sim-visualize
```

`xgb-sim` uses the final `1 - SPLIT` fraction of the XGBoost predictions by default, so with `SPLIT=0.95` it simulates the final 5% test slice unless `SIM_START` or `XGB_SIM_DEFAULT_TEST_FRACTION` is overridden.

`strategy` targets run rule-based models in separate paths:

```text
make strategy-train
make strategy-backtest
make strategy-experiment
make strategy-visualize
make strategy-sim
make strategy-sim-visualize
```

Current strategy types are `buy_hold`, `prev_movement`, and `ma`.

Model-run commands:

```bash
make save-run
make lr-save-run
make xgb-save-run
make strategy-save-run
make list-runs
make show-current
```

`AUTO_SAVE_RUN=1` saves a canonical run after `make experiment`, `make lr-experiment`, `make xgb-experiment`, or `make strategy-experiment`. Use `AUTO_SAVE_RUN=0` for temporary runs.

## Notes

Existing generated files in older flat paths are left in place. They are not moved automatically. Future default runs write to the organized paths unless a Makefile variable override is provided.

The model visualization and simulation visualization are separate HTML pages. The simulation page uses the bank simulator CSV and JSON report to show trade markers plus active capital invested over time.

## Dashboard

`make start` runs `src/dashboard_server.py` in the background on `REPORTS_HOST:REPORTS_PORT`, default `192.168.2.197:8000`. Use `make stop` to stop the server or `make start-fg` to run it in the foreground for attached logs.

The dashboard:

```text
/               Home
/models         Registry-driven model controls ordered by complexity
/compare        Overlay simulation equity and training loss curves
/reports        Generated HTML reports from data/reports
/live           Status/control page for the separate Docker live sim
```

The Docker live simulator remains separate on `LIVE_HOST_PORT`, default `8080`. The dashboard Live page proxies its JSON APIs and embeds/links the Docker dashboard; it does not merge live runtime state into the offline research server.

Dashboard form state is generated at `data/reports/dashboard_settings.json`. Shared experiment fields are saved once and reused across model tabs; model-specific fields are saved under each registry model id.

Tracked model defaults use the `recommended_v1` profile documented in `docs/recommended_settings.md`. The dashboard settings file can override those values locally after you edit a model tab.

To add another model tab, add a JSON file under `src/model_registry/` with the model id, label, `complexity_rank`, `complexity_group`, Makefile target lists, output link templates, defaults, and UI fields. Canonical fallback ranks and labels live in `src/model_catalog.py`.
