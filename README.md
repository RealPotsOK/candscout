# cryptopred

`cryptopred` is a from-scratch crypto prediction research project. It downloads candle data, builds features or candle sequences, trains simple NumPy models, backtests them, visualizes predictions, and runs bank-account simulations.

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
interval: 5m
model: NumPy sequence CNN
data source: Binance spot public candles
```

## Common Commands

Download candles:

```bash
make download
```

Train and backtest the default sequence CNN:

```bash
make experiment
```

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

Serve the simulation visualization on the LAN:

```bash
make sim-graph
```

Run the older logistic-regression pipeline:

```bash
make lr-experiment
```

## Useful Overrides

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
make experiment NN_MODEL_TYPE=mlp
```

Prediction-only research with no fee model:

```bash
make experiment EDGE=0 FEE=0
```

Simulation visualization options:

```bash
make sim-visualize SIM_ACTIVITY_BUCKET=hour SIM_MARKER_SIZE_BASIS=usd
make sim-visualize SIM_ACTIVITY_BUCKET=day SIM_MARKER_SIZE_BASIS=coin
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
