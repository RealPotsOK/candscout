# Recommended Starting Settings

The `recommended_v1` profile is a conservative baseline for comparing models. It is not claimed to be optimal.

## Shared Evaluation

- Chronological split: `0.90` train / `0.10` test
- Target: next-candle direction with `EDGE=0`
- Entry threshold: `0.55`
- Exit threshold: `0.48`
- Fee per side: `0.0001`
- Position mode: `hold`
- Max hold: `60` bars
- Stop loss: `0.02`
- Take profit: `0.04`
- Simulation account: `$10,000`
- Simulation trade size: `$100` to `$2,500`
- Confidence multiplier: `1.0`

The gap between entry and exit thresholds creates a hold zone. The model does not immediately sell merely because probability falls below the entry threshold.

## Model Presets

| Model | Recommended starting configuration |
| --- | --- |
| Buy and hold | Buy once and hold through the evaluation window |
| Previous movement | Follow the direction of the last completed candle |
| MA | `MA20` |
| Logistic regression | LR `0.01`, `1500` epochs, L2 `0.001`, balanced classes |
| XGBoost | `400` trees, depth `3`, LR `0.03`, row/column sample `0.8`, child weight `10`, L2 `2.0`, L1 `0.05` |
| MLP | Lookback `50`, basic channels, `64,32`, LR `0.0005`, `100` epochs, L2 `0.0005` |
| CNN | Lookback `50`, filters `16,32`, kernels `5,3`, dense `32,16`, LR `0.0005`, `100` epochs, L2 `0.0005` |
| GRU | Lookback `70`, technical channels, hidden `64`, one recurrent layer, dense `32`, LR `0.0005`, `80` epochs, L2 `0.0005` |
| LSTM | Lookback `70`, technical channels, hidden `64`, one recurrent layer, dense `32`, LR `0.0005`, `80` epochs, L2 `0.0005` |
| Transformer | Lookback `70`, technical channels, width `64`, `4` heads, `2` encoder layers, feed-forward `128`, dropout `0.1`, dense `32`, LR `0.0005`, `80` epochs, L2 `0.0005` |

Neural presets require CUDA. Training stops instead of silently falling back to CPU.

## Saved Dashboard State

The Models page persists settings in `data/reports/dashboard_settings.json`. The prior local settings were backed up under `data/reports/settings_snapshots/` before applying this profile.

Change one model tab without affecting another model's architecture settings. Shared asset values such as symbol, interval, and date range remain common across tabs.
