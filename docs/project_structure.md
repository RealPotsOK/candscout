# Project Structure

This project intentionally keeps a simple script-based layout instead of a Python package refactor.

## Source-Controlled Files

```text
src/                         Python scripts and model code
Makefile                     Main command interface
requirements.txt             Python dependencies
README.md                    Project overview and commands
docs/                        Project notes
.github/workflows/smoke.yml  GitHub smoke check
.gitignore                   Keeps generated artifacts out of git
```

## Generated Outputs

New runs use organized output paths:

```text
data/downloads/<source>/<symbol>/<interval>/candles.parquet
data/features/lr/<source>/<symbol>/<interval>/
data/reports/lr/<source>/<symbol>/<interval>/
data/reports/nn/<model_type>/<source>/<symbol>/<interval>/
data/reports/sim/<pipeline>/<source>/<symbol>/<interval>/
models/lr/<source>/<symbol>/<interval>/
models/nn/<model_type>/<source>/<symbol>/<interval>/
models/sim/<pipeline>/<source>/<symbol>/<interval>/
```

These generated folders are ignored by git.

## Model Families

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

## Notes

Existing generated files in older flat paths are left in place. They are not moved automatically. Future default runs write to the organized paths unless a Makefile variable override is provided.

The model visualization and simulation visualization are separate HTML pages. The simulation page uses the bank simulator CSV and JSON report to show trade markers plus active capital invested over time.
