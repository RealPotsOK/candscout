#!/usr/bin/env python3
"""Create an interactive HTML report for candles and model predictions."""

from __future__ import annotations

import argparse
import html as html_lib
import json
from pathlib import Path

import numpy as np
import pandas as pd


TIMEFRAMES = {
    "15m": "15min",
    "1h": "1h",
    "5h": "5h",
    "1d": "1D",
    "1w": "7D",
    "1mo": "30D",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize price action and model results.")
    parser.add_argument("--raw-data", required=True, help="Raw candle Parquet path")
    parser.add_argument("--predictions", required=True, help="Predictions Parquet path from backtest.py")
    parser.add_argument("--output", default="data/reports/model_visualization_5m.html", help="Output HTML path")
    parser.add_argument("--threshold", type=float, default=0.55, help="Signal probability threshold")
    parser.add_argument("--exit-threshold", type=float, default=0.48, help="Sell/exit probability threshold")
    parser.add_argument("--trade-mode", choices=["long_only", "short_only", "long_short"], default="long_only")
    parser.add_argument("--short-entry-threshold", type=float, default=0.55)
    parser.add_argument("--short-exit-threshold", type=float, default=0.48)
    parser.add_argument("--fee", type=float, default=0.001, help="Per-side fee used for net trade return")
    parser.add_argument("--starting-cash", type=float, default=10_000.0, help="Starting cash for equity chart")
    parser.add_argument(
        "--baseline-ma-window",
        type=int,
        default=20,
        help="Trailing candle count for the simple moving-average baseline",
    )
    parser.add_argument(
        "--max-browser-points",
        type=int,
        default=60_000,
        help="Maximum points embedded per main browser series before uniform thinning",
    )
    parser.add_argument("--title", default="BTCUSDT 5m Model Inspection", help="Report title")
    parser.add_argument("--nav-home-url", default="/", help="Header home/index URL")
    parser.add_argument("--nav-model-url", default="", help="Header model visualization URL")
    parser.add_argument("--nav-sim-url", default="", help="Header simulation visualization URL")
    return parser.parse_args()


def load_candles(path: Path) -> pd.DataFrame:
    df = pd.read_parquet(path)
    required = {"open_time", "open", "high", "low", "close", "volume"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Raw data missing required columns: {sorted(missing)}")

    df = df.sort_values("open_time").drop_duplicates("open_time").reset_index(drop=True)
    df["open_time"] = pd.to_datetime(df["open_time"], utc=True)
    return df


def load_predictions(
    path: Path,
    threshold: float,
    exit_threshold: float,
    fee: float,
    trade_mode: str = "long_only",
    short_entry_threshold: float = 0.45,
) -> pd.DataFrame:
    df = pd.read_parquet(path)
    required = {"open_time", "close", "target", "forward_return", "prob_up"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Predictions data missing required columns: {sorted(missing)}")

    df = df.sort_values("open_time").drop_duplicates("open_time").reset_index(drop=True)
    df["open_time"] = pd.to_datetime(df["open_time"], utc=True)
    if "prob_down" not in df.columns:
        df["prob_down"] = 0.0
    if "target_up" not in df.columns:
        df["target_up"] = df["target"].astype(int)
    if "target_down" not in df.columns:
        df["target_down"] = (df["forward_return"] < 0.0).astype(int)
    split_counts: dict[str, int] = {}
    test_start_open_time: str | None = None
    if "dataset_split" in df.columns:
        split_values = df["dataset_split"].astype(str).str.lower()
        split_counts = {str(key): int(value) for key, value in split_values.value_counts().items()}
        test_times = df.loc[split_values == "test", "open_time"]
        if not test_times.empty:
            test_start_open_time = pd.Timestamp(test_times.iloc[0]).isoformat()

    if "signal_at_threshold" not in df.columns:
        df["signal_at_threshold"] = (df["prob_up"] >= threshold).astype(int)

    df["prediction_signal"] = (df["prob_up"] >= threshold).astype(int)
    df["short_signal"] = (
        (df["prob_down"] >= short_entry_threshold) & (trade_mode != "long_only")
    ).astype(int)
    df["probability_exit_signal"] = (df["prob_up"] < exit_threshold).astype(int)
    df["prediction_side"] = np.select(
        [df["prob_up"] >= threshold, df["prob_up"] < exit_threshold],
        ["buy", "sell"],
        default="hold",
    )
    df["buy_net_return"] = df["forward_return"] - (2.0 * fee)
    df["one_bar_net_return"] = df["buy_net_return"]
    df["net_return"] = df["buy_net_return"]
    df["entry_action"] = df["entry_signal"].astype(int) if "entry_signal" in df.columns else df["signal_at_threshold"]
    if "exit_signal" in df.columns:
        raw_exit_signal = df["exit_signal"].astype(int)
        if "exit_reason" in df.columns:
            exit_reason = df["exit_reason"].fillna("").astype(str)
            model_exit_mask = exit_reason.isin(["exit_threshold", "one_bar"])
            df["exit_action"] = (raw_exit_signal.eq(1) & model_exit_mask).astype(int)
            df["risk_exit_action"] = (raw_exit_signal.eq(1) & ~model_exit_mask).astype(int)
        else:
            df["exit_action"] = raw_exit_signal
            df["risk_exit_action"] = 0
    else:
        df["exit_action"] = df["probability_exit_signal"]
        df["risk_exit_action"] = 0
    df["hold_action"] = (
        (df["entry_action"] == 0) & (df["exit_action"] == 0) & (df["risk_exit_action"] == 0)
    ).astype(int)
    df["action_signal"] = df["entry_action"]
    if "entry_trade_net_return" in df.columns:
        df["outcome_net_return"] = df["entry_trade_net_return"]
    else:
        df["outcome_net_return"] = df["net_return"]
    df["label_hit"] = (df["target"] == 1) & (df["action_signal"] == 1)
    df["profitable"] = (df["outcome_net_return"] > 0.0) & (df["action_signal"] == 1)
    df.attrs["evaluation_split"] = "full_prediction_timeline"
    df.attrs["split_counts"] = split_counts
    df.attrs["test_start_open_time"] = test_start_open_time
    return df


def resample_candles(df: pd.DataFrame, rule: str) -> pd.DataFrame:
    indexed = df.set_index("open_time")
    out = indexed.resample(rule).agg(
        {
            "open": "first",
            "high": "max",
            "low": "min",
            "close": "last",
            "volume": "sum",
        }
    )
    return out.dropna(subset=["open", "high", "low", "close"]).reset_index()


def infer_base_interval_label(candles: pd.DataFrame) -> str:
    diffs = candles["open_time"].diff().dropna().dt.total_seconds()
    if diffs.empty:
        return "5m"

    seconds = int(round(float(diffs.median())))
    if seconds % 86_400 == 0:
        return f"{seconds // 86_400}d"
    if seconds % 3_600 == 0:
        return f"{seconds // 3_600}h"
    if seconds % 60 == 0:
        return f"{seconds // 60}m"
    return f"{seconds}s"


def to_iso_list(series: pd.Series) -> list[str]:
    return pd.to_datetime(series, utc=True).dt.strftime("%Y-%m-%dT%H:%M:%SZ").tolist()


def thin_indices(length: int, max_points: int) -> np.ndarray:
    if length <= max_points or max_points <= 0:
        return np.arange(length, dtype=np.int64)
    if max_points <= 2:
        return np.array([0, length - 1], dtype=np.int64)
    raw = np.linspace(0, length - 1, num=max_points)
    return np.unique(np.rint(raw).astype(np.int64))


def thin_frame(frame: pd.DataFrame, max_points: int) -> pd.DataFrame:
    if len(frame) <= max_points:
        return frame
    return frame.iloc[thin_indices(len(frame), max_points)].reset_index(drop=True)


def thin_equity_payload(equity: dict, indices: np.ndarray) -> dict:
    out = {"x": [equity["x"][int(i)] for i in indices]}
    for key, values in equity.items():
        if key == "x":
            continue
        out[key] = [values[int(i)] for i in indices]
    return out


def candle_payload(candles: pd.DataFrame, base_label: str, max_points: int) -> tuple[dict, str]:
    payload = {}
    actual_base = base_label
    timeframes = {base_label: None, **{k: v for k, v in TIMEFRAMES.items() if k != base_label}}
    for label, rule in timeframes.items():
        frame = candles if rule is None else resample_candles(candles, rule)
        if label == base_label and len(frame) > max_points:
            continue
        if actual_base == base_label and label != base_label:
            actual_base = label
        frame = thin_frame(frame, max_points)
        payload[label] = {
            "x": to_iso_list(frame["open_time"]),
            "open": frame["open"].round(6).tolist(),
            "high": frame["high"].round(6).tolist(),
            "low": frame["low"].round(6).tolist(),
            "close": frame["close"].round(6).tolist(),
        }
    if actual_base not in payload:
        actual_base = next(iter(payload))
    return payload, actual_base


def marker_payload(predictions: pd.DataFrame, max_points_per_group: int) -> dict:
    eps = 1e-12
    has_trade_outcomes = "entry_trade_net_return" in predictions.columns and predictions["entry_trade_net_return"].notna().any()
    buy_signal_col = "entry_action" if "entry_action" in predictions.columns else "prediction_signal"
    sell_signal_col = "exit_action" if "exit_action" in predictions.columns else "probability_exit_signal"
    buy_return_col = "outcome_net_return" if has_trade_outcomes else "buy_net_return"

    buy = predictions[predictions[buy_signal_col] == 1].copy()
    buy_returns = pd.to_numeric(buy[buy_return_col], errors="coerce")
    sell = predictions[predictions[sell_signal_col] == 1].copy()

    groups = {
        "buy_win": buy[buy_returns > eps],
        "buy_loss": buy[buy_returns < -eps],
        "sell_good": sell[sell["forward_return"] < -eps],
        "sell_bad": sell[sell["forward_return"] > eps],
        "no_loss": predictions[
            ((predictions[buy_signal_col] == 1) & (predictions[buy_return_col].abs() <= eps))
            | ((predictions[sell_signal_col] == 1) & (predictions["forward_return"].abs() <= eps))
        ],
        "short_win": predictions[
            (predictions["short_signal"] == 1) & (predictions["forward_return"] < -eps)
        ],
        "short_loss": predictions[
            (predictions["short_signal"] == 1) & (predictions["forward_return"] > eps)
        ],
    }

    out = {}
    for name, frame in groups.items():
        frame = thin_frame(frame, max_points_per_group)
        out[name] = {
            "x": to_iso_list(frame["open_time"]),
            "close": frame["close"].round(6).tolist(),
            "prob_up": frame["prob_up"].round(6).tolist(),
            "prob_down": frame["prob_down"].round(6).tolist(),
        }
    return out


def predictions_payload(predictions: pd.DataFrame) -> dict:
    return {
        "x": to_iso_list(predictions["open_time"]),
        "prob_up": predictions["prob_up"].round(6).tolist(),
        "prob_down": predictions["prob_down"].round(6).tolist(),
    }


def mean_or_zero(series: pd.Series) -> float:
    return float(series.mean()) if len(series) else 0.0


def model_return_series(predictions: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    if "realized_trade_return" in predictions.columns:
        returns = pd.to_numeric(predictions["realized_trade_return"], errors="coerce")
        executed = returns.notna()
        return returns.fillna(0.0), executed

    if "entry_trade_net_return" in predictions.columns:
        returns = pd.to_numeric(predictions["entry_trade_net_return"], errors="coerce")
        executed = returns.notna()
        return returns.fillna(0.0), executed

    returns = predictions["buy_net_return"].where(predictions["prediction_signal"] == 1, 0.0)
    executed = predictions["prediction_signal"] == 1
    return returns.fillna(0.0), executed


def moving_average_baseline_equity(
    predictions: pd.DataFrame,
    starting_cash: float,
    fee: float,
    ma_window: int,
) -> list[float]:
    if ma_window < 1:
        raise ValueError("--baseline-ma-window must be at least 1")

    closes = predictions["close"].to_numpy(dtype=np.float64)
    ma = predictions["close"].rolling(window=ma_window, min_periods=ma_window).mean().shift(1)

    cash = float(starting_cash)
    coin = 0.0
    equity: list[float] = []

    for close, ma_value in zip(closes, ma.to_numpy(dtype=np.float64)):
        if np.isfinite(ma_value) and close > 0.0:
            if close > ma_value and coin <= 0.0 and cash > 0.0:
                entry_fee = cash * fee
                coin = max(0.0, cash - entry_fee) / close
                cash = 0.0
            elif close < ma_value and coin > 0.0:
                gross_exit = coin * close
                exit_fee = gross_exit * fee
                cash = max(0.0, gross_exit - exit_fee)
                coin = 0.0
        equity.append(cash + coin * close)

    return equity


def equity_payload(
    predictions: pd.DataFrame,
    starting_cash: float,
    fee: float,
    baseline_ma_window: int,
) -> tuple[dict, dict]:
    if starting_cash <= 0.0:
        raise ValueError("--starting-cash must be positive")

    model_returns, executed = model_return_series(predictions)
    closes = predictions["close"].to_numpy(dtype=np.float64)
    if len(closes) == 0:
        raise ValueError("Need at least one prediction row for equity chart")

    model_equity: list[float] = []
    fees_paid = 0.0
    cash = float(starting_cash)

    for row_return, is_executed in zip(model_returns.to_numpy(dtype=np.float64), executed.to_numpy(dtype=bool)):
        if is_executed:
            # Backtest returns subtract 2*fee from each completed trade.
            fees_paid += cash * 2.0 * fee
            cash *= 1.0 + float(row_return)
        model_equity.append(cash)

    first_close = float(closes[0])
    if first_close <= 0.0:
        buy_hold_equity = np.full(len(closes), float(starting_cash), dtype=np.float64)
    else:
        buy_hold_equity = float(starting_cash) * (closes / first_close)

    ma_baseline_equity = moving_average_baseline_equity(
        predictions=predictions,
        starting_cash=starting_cash,
        fee=fee,
        ma_window=baseline_ma_window,
    )

    equity = {
        "x": to_iso_list(predictions["open_time"]),
        "model": [round(x, 6) for x in model_equity],
        "buy_hold": [round(float(x), 6) for x in buy_hold_equity],
        "ma_baseline": [round(float(x), 6) for x in ma_baseline_equity],
    }
    summary = {
        "starting_cash": float(starting_cash),
        "model_ending_cash": float(model_equity[-1]),
        "buy_hold_ending_cash": float(buy_hold_equity[-1]),
        "ma_baseline_ending_cash": float(ma_baseline_equity[-1]),
        "model_net_profit_cash": float(model_equity[-1] - starting_cash),
        "buy_hold_net_profit_cash": float(buy_hold_equity[-1] - starting_cash),
        "ma_baseline_net_profit_cash": float(ma_baseline_equity[-1] - starting_cash),
        "fees_paid_cash": float(fees_paid),
        "ma_baseline_window": int(baseline_ma_window),
    }
    return equity, summary


def infer_position_mode(predictions: pd.DataFrame) -> str:
    if "exit_reason" not in predictions.columns:
        return "prediction_only"

    exit_reasons = predictions["exit_reason"].fillna("").astype(str)
    nonempty = exit_reasons[exit_reasons != ""]
    if nonempty.empty:
        return "prediction_only"
    if set(nonempty.unique()) == {"one_bar"}:
        return "one_bar"
    return "hold"


def outcome_summary(predictions: pd.DataFrame, fee: float, equity_summary: dict) -> dict:
    eps = 1e-12
    has_trade_outcomes = "entry_trade_net_return" in predictions.columns and predictions["entry_trade_net_return"].notna().any()
    buy_signal_col = "entry_action" if "entry_action" in predictions.columns else "prediction_signal"
    sell_signal_col = "exit_action" if "exit_action" in predictions.columns else "probability_exit_signal"
    buy_return_col = "outcome_net_return" if has_trade_outcomes else "buy_net_return"

    buy_predictions = predictions[predictions[buy_signal_col] == 1].copy()
    buy_returns = pd.to_numeric(buy_predictions[buy_return_col], errors="coerce")
    sell_predictions = predictions[predictions[sell_signal_col] == 1]
    hold_predictions = predictions[(predictions[buy_signal_col] == 0) & (predictions[sell_signal_col] == 0)]
    risk_exit_predictions = predictions[predictions.get("risk_exit_action", 0) == 1]

    buy_wins = buy_predictions[buy_returns > eps]
    buy_losses = buy_predictions[buy_returns < -eps]
    sell_good = sell_predictions[sell_predictions["forward_return"] < -eps]
    sell_bad = sell_predictions[sell_predictions["forward_return"] > eps]

    executed_trade_returns, executed = model_return_series(predictions)
    trade_count = int(executed.sum())
    net_profit_return_sum = float(executed_trade_returns.loc[executed].sum()) if trade_count else 0.0

    no_loss = int(
        (
            ((predictions[buy_signal_col] == 1) & (predictions[buy_return_col].abs() <= eps))
            | ((predictions[sell_signal_col] == 1) & (predictions["forward_return"].abs() <= eps))
        ).sum()
    )

    return {
        "evaluation_split": str(predictions.attrs.get("evaluation_split", "unknown")),
        "train_prediction_rows": int(predictions.attrs.get("split_counts", {}).get("train", 0)),
        "test_prediction_rows": int(predictions.attrs.get("split_counts", {}).get("test", 0)),
        "test_start_open_time": predictions.attrs.get("test_start_open_time"),
        "prediction_rows": int(len(predictions)),
        "buy_predictions": int(len(buy_predictions)),
        "sell_predictions": int(len(sell_predictions)),
        "hold_predictions": int(len(hold_predictions)),
        "risk_exit_predictions": int(len(risk_exit_predictions)),
        "buy_win_rate": float(len(buy_wins) / len(buy_predictions)) if len(buy_predictions) else 0.0,
        "sell_good_rate": float(len(sell_good) / len(sell_predictions)) if len(sell_predictions) else 0.0,
        "no_loss_predictions": no_loss,
        "average_buy_win": mean_or_zero(buy_wins[buy_return_col]),
        "average_buy_loss": mean_or_zero(buy_losses[buy_return_col]),
        "average_sell_good_gain": mean_or_zero(-sell_good["forward_return"]),
        "average_sell_bad_missed_gain": mean_or_zero(sell_bad["forward_return"]),
        # Sell markers are visual diagnostics. A sell is not treated as an extra
        # loss here; actual trade PnL is already represented by executed trades.
        "average_sell_bad_loss": 0.0,
        "sell_marker_metric": "visual_only_not_pnl_penalty",
        "executed_trade_count": trade_count,
        "fees_paid": float(equity_summary["fees_paid_cash"]),
        "net_profit": float(equity_summary["model_net_profit_cash"]),
        "net_profit_return_sum": net_profit_return_sum,
        "starting_cash": float(equity_summary["starting_cash"]),
        "model_ending_cash": float(equity_summary["model_ending_cash"]),
        "buy_hold_ending_cash": float(equity_summary["buy_hold_ending_cash"]),
        "ma_baseline_ending_cash": float(equity_summary["ma_baseline_ending_cash"]),
        "buy_hold_net_profit_cash": float(equity_summary["buy_hold_net_profit_cash"]),
        "ma_baseline_net_profit_cash": float(equity_summary["ma_baseline_net_profit_cash"]),
        "ma_baseline_window": int(equity_summary["ma_baseline_window"]),
        "position_mode": infer_position_mode(predictions),
        "buy_metric": "executed_trade_outcome" if has_trade_outcomes else "one_bar_prediction_return",
    }


def nav_html(home_url: str, model_url: str, sim_url: str, active: str) -> str:
    items = [
        ("Dashboard", "/", "dashboard"),
        ("Models", "/models", "models"),
        ("Compare", "/compare", "compare"),
        ("Reports", home_url, "home"),
        ("Model", model_url, "model"),
        ("Simulation", sim_url, "sim"),
        ("Live", "/live", "live"),
    ]
    links = []
    for label, url, key in items:
        if not url:
            continue
        css_class = "active" if key == active else ""
        links.append(
            f'<a class="{css_class}" href="{html_lib.escape(url, quote=True)}">'
            f"{html_lib.escape(label)}</a>"
        )
    if not links:
        return ""
    return '<nav class="site-nav">' + "".join(links) + "</nav>"


def html_template(title: str, data: dict, nav: str) -> str:
    payload = json.dumps(data, separators=(",", ":"))
    title_text = html_lib.escape(title)
    template = """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>__TITLE__</title>
  <style>
    body {{
      margin: 0;
      background: #f7f7f2;
      color: #17201a;
      font-family: Georgia, "Times New Roman", serif;
    }}
    header {{
      padding: 18px 24px 10px;
      border-bottom: 1px solid #d8d8ce;
    }}
    h1 {{
      margin: 0 0 8px;
      font-size: 24px;
      font-weight: 700;
      letter-spacing: 0;
    }}
    .site-nav {{
      display: flex;
      flex-wrap: wrap;
      gap: 8px;
      margin: 0 0 10px;
      font-family: "Courier New", monospace;
      font-size: 13px;
    }}
    .site-nav a {{
      color: #17201a;
      text-decoration: none;
      border: 1px solid #c8c8bc;
      background: #fffdf3;
      padding: 6px 9px;
      border-radius: 4px;
    }}
    .site-nav a:hover,
    .site-nav a.active {{
      color: #ffffff;
      background: #17201a;
      border-color: #17201a;
    }}
    .summary {{
      display: flex;
      flex-wrap: wrap;
      gap: 8px;
      font-family: "Courier New", monospace;
      font-size: 13px;
    }}
    .summary span {{
      border: 1px solid #d8d8ce;
      background: #ffffff;
      padding: 5px 8px;
      border-radius: 4px;
    }}
    main {{
      padding: 14px 16px 24px;
    }}
    .chart {{
      position: relative;
      width: 100%;
      height: 48vh;
      min-height: 360px;
      border: 1px solid #d8d8ce;
      background: #ffffff;
      cursor: crosshair;
    }}
    .chart-controls {{
      position: absolute;
      top: 8px;
      right: 8px;
      z-index: 4;
      display: flex;
      gap: 6px;
    }}
    .chart-controls button {{
      border: 1px solid #c8c8bc;
      background: rgba(255, 253, 243, 0.94);
      color: #17201a;
      padding: 6px 9px;
      border-radius: 4px;
      cursor: pointer;
      font: 12px "Courier New", monospace;
    }}
    .chart.is-collapsed {{
      height: 48px !important;
      min-height: 48px !important;
      overflow: hidden;
      cursor: default;
    }}
    .chart.is-collapsed canvas {{ display: none; }}
    .chart.is-expanded {{
      position: fixed;
      inset: 12px;
      z-index: 50;
      width: auto;
      height: auto !important;
      min-height: 0 !important;
      margin: 0 !important;
      box-shadow: 0 24px 80px rgba(23, 32, 26, 0.28);
    }}
    body.chart-overlay-open {{ overflow: hidden; }}
    #performance {{
      height: 34vh;
      min-height: 280px;
      margin-top: 12px;
    }}
    #equity {{
      height: 34vh;
      min-height: 280px;
      margin-top: 12px;
    }}
    canvas {{
      display: block;
      width: 100%;
      height: 100%;
    }}
    .hint {{
      margin-top: 8px;
      font-family: "Courier New", monospace;
      font-size: 12px;
      color: #4f594f;
    }}
    .tooltip {{
      position: fixed;
      pointer-events: none;
      display: none;
      z-index: 10;
      background: rgba(255, 255, 255, 0.96);
      border: 1px solid #c9c9bd;
      color: #17201a;
      font: 12px "Courier New", monospace;
      padding: 6px 8px;
      box-shadow: 0 4px 14px rgba(0, 0, 0, 0.12);
      max-width: 320px;
      white-space: nowrap;
    }}
  </style>
  <link rel="stylesheet" href="/assets/candscout.css">
</head>
<body>
  <header>
    <h1>__TITLE__</h1>
    __NAV__
    <div class="summary" id="summary"></div>
    <div class="hint">Wheel to zoom, drag to pan, double-click to reset. The full train + test prediction timeline is shown; train rows are in-sample and the test divider marks out-of-sample data. Candles automatically regroup as the visible window changes.</div>
  </header>
  <main>
    <div id="price" class="chart"><div class="chart-controls"><button data-chart-action="collapse">Collapse</button><button data-chart-action="expand">Expand</button></div><canvas id="priceCanvas"></canvas></div>
    <div id="performance" class="chart"><div class="chart-controls"><button data-chart-action="collapse">Collapse</button><button data-chart-action="expand">Expand</button></div><canvas id="perfCanvas"></canvas></div>
    <div id="equity" class="chart"><div class="chart-controls"><button data-chart-action="collapse">Collapse</button><button data-chart-action="expand">Expand</button></div><canvas id="equityCanvas"></canvas></div>
  </main>
  <div class="tooltip" id="tooltip"></div>
  <script>
    const report = __PAYLOAD__;
    const priceCanvas = document.getElementById("priceCanvas");
    const perfCanvas = document.getElementById("perfCanvas");
    const equityCanvas = document.getElementById("equityCanvas");
    const tooltip = document.getElementById("tooltip");
    const colors = {
      green: "#17803a",
      greenSoft: "#d9f0df",
      red: "#a82727",
      redSoft: "#f5d9d5",
      grid: "#ecece4",
      axis: "#6f6f65",
      text: "#17201a",
      prob: "#2f5d7c",
      probDown: "#8e3f72",
      modelEquity: "#173f6b",
      buyHoldEquity: "#b06b18",
      maBaselineEquity: "#2d7c64",
      threshold: "#6f5b2d",
      exitThreshold: "#8a6bb8",
      shortEntryThreshold: "#7b4ab3",
      shortExitThreshold: "#d5a20f",
      buyWin: "#10843f",
      buyLoss: "#bf2f2f",
      sellGood: "#247d8f",
      sellBad: "#e64b9b",
      noLoss: "#f28c18",
      shortWin: "#7b4ab3",
      shortLoss: "#e0ad16"
    };

    function setupChartControls() {{
      for (const chart of document.querySelectorAll(".chart")) {{
        const collapse = chart.querySelector('[data-chart-action="collapse"]');
        const expand = chart.querySelector('[data-chart-action="expand"]');
        collapse.addEventListener("click", (event) => {{
          event.stopPropagation();
          if (chart.classList.contains("is-expanded")) {{
            chart.classList.remove("is-expanded");
            document.body.classList.remove("chart-overlay-open");
            expand.textContent = "Expand";
          }}
          chart.classList.toggle("is-collapsed");
          collapse.textContent = chart.classList.contains("is-collapsed") ? "Show" : "Collapse";
          scheduleRedraw();
        }});
        expand.addEventListener("click", (event) => {{
          event.stopPropagation();
          chart.classList.remove("is-collapsed");
          collapse.textContent = "Collapse";
          const expanded = chart.classList.toggle("is-expanded");
          document.body.classList.toggle("chart-overlay-open", expanded);
          expand.textContent = expanded ? "Unexpand" : "Expand";
          scheduleRedraw();
        }});
      }}
      document.addEventListener("keydown", (event) => {{
        if (event.key !== "Escape") return;
        const expanded = document.querySelector(".chart.is-expanded");
        if (!expanded) return;
        expanded.classList.remove("is-expanded");
        expanded.querySelector('[data-chart-action="expand"]').textContent = "Expand";
        document.body.classList.remove("chart-overlay-open");
        scheduleRedraw();
      }});
    }}

    for (const frame of Object.values(report.candles)) {
      frame.t = frame.x.map((x) => new Date(x).getTime());
    }
    report.predictions.t = report.predictions.x.map((x) => new Date(x).getTime());
    report.equity.t = report.equity.x.map((x) => new Date(x).getTime());
    for (const group of Object.values(report.markers)) {
      group.t = group.x.map((x) => new Date(x).getTime());
    }
    report.test_start_t = report.test_start ? new Date(report.test_start).getTime() : null;

    const baseFrame = report.base_frame || Object.keys(report.candles)[0];
    const fullRange = [
      report.candles[baseFrame].t[0],
      report.candles[baseFrame].t[report.candles[baseFrame].t.length - 1]
    ];
    let xRange = [...fullRange];
    let activeFrame = "1d";
    let drag = null;
    let lastHover = null;
    let equityHoverIndex = null;
    let redrawPending = false;

    function chooseFrame(start, end, plotWidth) {{
      const targetBars = Math.max(120, Math.min(500, Math.floor(plotWidth / 2.5)));
      const frames = Object.entries(report.candles)
        .filter(([, frame]) => frame.t.length)
        .sort((a, b) => {{
          const aSpan = a[1].t.length > 1 ? a[1].t[a[1].t.length - 1] - a[1].t[0] : Infinity;
          const bSpan = b[1].t.length > 1 ? b[1].t[b[1].t.length - 1] - b[1].t[0] : Infinity;
          return (aSpan / Math.max(1, a[1].t.length - 1)) - (bSpan / Math.max(1, b[1].t.length - 1));
        }});
      let selected = frames[frames.length - 1]?.[0] || baseFrame;
      for (const [name, frame] of frames) {{
        const visible = upperBound(frame.t, end) - lowerBound(frame.t, start);
        if (visible <= targetBars) {{
          selected = name;
          break;
        }}
      }}
      return selected;
    }}

    function updateSummary() {{
      const s = report.summary;
      const fmtPct = (value) => (100 * value).toFixed(3) + "%";
      document.getElementById("summary").innerHTML = [
        "active candles=" + activeFrame,
        "timeline=full train+test",
        "train_rows=" + s.train_prediction_rows,
        "test_rows=" + s.test_prediction_rows,
        "position_mode=" + s.position_mode,
        "buy_metric=" + s.buy_metric,
        "starting_cash=" + formatMoney(s.starting_cash),
        "model_end=" + formatMoney(s.model_ending_cash),
        "buy_hold_end=" + formatMoney(s.buy_hold_ending_cash),
        "ma" + s.ma_baseline_window + "_end=" + formatMoney(s.ma_baseline_ending_cash),
        "rows=" + s.prediction_rows,
        "buy_preds=" + s.buy_predictions,
        "sell_preds=" + s.sell_predictions,
        "hold_preds=" + s.hold_predictions,
        "risk_exits=" + s.risk_exit_predictions,
        "buy_win_rate=" + (100 * s.buy_win_rate).toFixed(3) + "%",
        "sell_good_rate=" + (100 * s.sell_good_rate).toFixed(3) + "%",
        "average_buy_win=" + fmtPct(s.average_buy_win),
        "average_buy_loss=" + fmtPct(s.average_buy_loss),
        "average_sell_good_gain=" + fmtPct(s.average_sell_good_gain),
        "average_sell_bad_missed_gain=" + fmtPct(s.average_sell_bad_missed_gain || 0),
        "sell_markers=" + (s.sell_marker_metric || "visual_only"),
        "executed_trades=" + s.executed_trade_count,
        "fees_paid=" + formatMoney(s.fees_paid),
        "net_profit=" + formatMoney(s.net_profit),
        "flat=" + s.no_loss_predictions,
        "threshold=" + report.threshold,
        "exit_threshold=" + report.exit_threshold
      ].map((x) => "<span>" + x + "</span>").join("");
    }}

    function formatMoney(value) {
      return "$" + Number(value).toLocaleString(undefined, {
        minimumFractionDigits: 2,
        maximumFractionDigits: 2
      });
    }

    function setupCanvas(canvas) {{
      const ratio = window.devicePixelRatio || 1;
      const rect = canvas.parentElement.getBoundingClientRect();
      canvas.width = Math.max(1, Math.floor(rect.width * ratio));
      canvas.height = Math.max(1, Math.floor(rect.height * ratio));
      const ctx = canvas.getContext("2d");
      ctx.setTransform(ratio, 0, 0, ratio, 0, 0);
      return { ctx, width: rect.width, height: rect.height };
    }}

    function visibleIndexes(times) {{
      const start = Math.max(0, lowerBound(times, xRange[0]) - 2);
      const end = Math.min(times.length, upperBound(times, xRange[1]) + 2);
      const out = [];
      for (let i = start; i < end; i++) out.push(i);
      return out;
    }}

    function scaleX(t, plot) {{
      return plot.left + ((t - xRange[0]) / (xRange[1] - xRange[0])) * plot.width;
    }}

    function scaleY(v, min, max, plot) {{
      if (max === min) return plot.top + plot.height / 2;
      return plot.top + (1 - ((v - min) / (max - min))) * plot.height;
    }}

    function drawGrid(ctx, plot, yMin, yMax, yLabel) {{
      ctx.strokeStyle = colors.grid;
      ctx.fillStyle = colors.axis;
      ctx.lineWidth = 1;
      ctx.font = "12px Courier New";
      ctx.textBaseline = "middle";

      for (let i = 0; i <= 4; i++) {
        const y = plot.top + (plot.height * i) / 4;
        ctx.beginPath();
        ctx.moveTo(plot.left, y);
        ctx.lineTo(plot.left + plot.width, y);
        ctx.stroke();
        const value = yMax - ((yMax - yMin) * i) / 4;
        ctx.fillText(value.toFixed(yLabel === "prob" ? 2 : 0), 8, y);
      }

      for (let i = 0; i <= 5; i++) {
        const x = plot.left + (plot.width * i) / 5;
        const t = new Date(xRange[0] + ((xRange[1] - xRange[0]) * i) / 5);
        ctx.beginPath();
        ctx.moveTo(x, plot.top);
        ctx.lineTo(x, plot.top + plot.height);
        ctx.stroke();
        ctx.save();
        ctx.textAlign = "center";
        ctx.textBaseline = "top";
        ctx.fillText(t.toISOString().slice(5, 16).replace("T", " "), x, plot.top + plot.height + 8);
        ctx.restore();
      }
    }}

    function drawTestBoundary(ctx, plot) {{
      const t = report.test_start_t;
      if (!Number.isFinite(t) || t < xRange[0] || t > xRange[1]) return;
      const x = scaleX(t, plot);
      ctx.save();
      ctx.strokeStyle = "#7d5a17";
      ctx.fillStyle = "#7d5a17";
      ctx.lineWidth = 1.5;
      ctx.setLineDash([7, 5]);
      ctx.beginPath();
      ctx.moveTo(x, plot.top);
      ctx.lineTo(x, plot.top + plot.height);
      ctx.stroke();
      ctx.setLineDash([]);
      ctx.font = "11px Courier New";
      ctx.textAlign = x > plot.left + plot.width - 90 ? "right" : "left";
      ctx.textBaseline = "bottom";
      ctx.fillText("test starts", x + (ctx.textAlign === "right" ? -5 : 5), plot.top + plot.height - 4);
      ctx.restore();
    }}

    function drawLegend(ctx, items, x, y, maxWidth) {
      ctx.font = "12px Courier New";
      ctx.textBaseline = "top";
      let offset = 0;
      let row = 0;
      const rowHeight = 17;
      for (const item of items) {
        const itemWidth = item.width || Math.max(76, ctx.measureText(item.label).width + 24);
        if (offset > 0 && offset + itemWidth > maxWidth) {
          offset = 0;
          row += 1;
        }
        const itemX = x + offset;
        const itemY = y + row * rowHeight;
        ctx.fillStyle = item.color;
        ctx.fillRect(itemX, itemY, 10, 10);
        ctx.fillStyle = colors.text;
        ctx.fillText(item.label, itemX + 14, itemY - 1);
        offset += itemWidth;
      }
    }}

    function markerGroups() {
      return [
        ["buy_win", colors.buyWin, "buy + won money"],
        ["buy_loss", colors.buyLoss, "buy + lost money"],
        ["sell_good", colors.sellGood, "sell + avoided loss"],
        ["sell_bad", colors.sellBad, "sell + price went up"],
        ["no_loss", colors.noLoss, "buy/sell + flat"],
        ["short_win", colors.shortWin, "short + won money"],
        ["short_loss", colors.shortLoss, "short + lost money"]
      ];
    }}

    const MAX_INDIVIDUAL_MARKERS = 600;
    const MIN_MARKER_SPACING_PX = 8;

    function markerBucketPx() {
      const visibleSpan = Math.max(1, xRange[1] - xRange[0]);
      const fullSpan = Math.max(visibleSpan, fullRange[1] - fullRange[0]);
      const zoom = Math.max(1, fullSpan / visibleSpan);
      if (zoom < 2) return 28;
      if (zoom < 8) return 20;
      if (zoom < 32) return 14;
      return 9;
    }

    function lowerBound(values, target) {
      let lo = 0;
      let hi = values.length;
      while (lo < hi) {
        const mid = Math.floor((lo + hi) / 2);
        if (values[mid] < target) lo = mid + 1;
        else hi = mid;
      }
      return lo;
    }

    function upperBound(values, target) {
      let lo = 0;
      let hi = values.length;
      while (lo < hi) {
        const mid = Math.floor((lo + hi) / 2);
        if (values[mid] <= target) lo = mid + 1;
        else hi = mid;
      }
      return lo;
    }

    function visibleSlice(times) {
      return [lowerBound(times, xRange[0]), upperBound(times, xRange[1])];
    }

    function drawIndividualMarkers(ctx, item, plot, yForPoint, withStroke) {
      const g = item.group;
      ctx.fillStyle = item.color;
      if (withStroke) ctx.strokeStyle = "#111";

      for (let i = item.start; i < item.end; i++) {
        const x = scaleX(g.t[i], plot);
        const y = yForPoint(g, i);
        ctx.beginPath();
        ctx.arc(x, y, 3.5, 0, Math.PI * 2);
        ctx.fill();
        if (withStroke) ctx.stroke();
      }

      return item.end - item.start;
    }

    function drawClusteredMarkers(ctx, item, plot, yForPoint, withStroke) {
      const g = item.group;
      const clusters = new Map();
      const bucketPx = markerBucketPx();

      for (let i = item.start; i < item.end; i++) {
        const x = scaleX(g.t[i], plot);
        if (x < plot.left - bucketPx || x > plot.left + plot.width + bucketPx) continue;
        const y = yForPoint(g, i);
        const bucket = Math.floor((x - plot.left) / bucketPx);
        let cluster = clusters.get(bucket);
        if (!cluster) {
          cluster = { x: 0, y: 0, count: 0 };
          clusters.set(bucket, cluster);
        }
        cluster.x += x;
        cluster.y += y;
        cluster.count += 1;
      }

      ctx.fillStyle = item.color;
      if (withStroke) ctx.strokeStyle = "#111";
      ctx.globalAlpha = 0.82;

      for (const cluster of clusters.values()) {
        const x = cluster.x / cluster.count;
        const y = cluster.y / cluster.count;
        const radius = Math.min(10, 3.5 + Math.log2(cluster.count + 1) * 1.15);
        ctx.beginPath();
        ctx.arc(x, y, radius, 0, Math.PI * 2);
        ctx.fill();
        if (withStroke) ctx.stroke();
      }

      ctx.globalAlpha = 1;
      return clusters.size;
    }

    function drawMarkerGroups(ctx, plot, yForPoint, withStroke) {
      const items = [];
      let totalVisible = 0;

      for (const [key, color, label] of markerGroups()) {
        const group = report.markers[key];
        const [start, end] = visibleSlice(group.t);
        const visibleCount = end - start;
        totalVisible += visibleCount;
        items.push({ key, color, label, group, start, end, visibleCount });
      }

      const spacing = plot.width / Math.max(1, totalVisible);
      const clusterAll = totalVisible > MAX_INDIVIDUAL_MARKERS || spacing < MIN_MARKER_SPACING_PX;
      let drawnMarkers = 0;
      let usedClusters = false;

      for (const item of items) {
        if (item.visibleCount <= 0) continue;
        const groupSpacing = plot.width / Math.max(1, item.visibleCount);
        const shouldCluster = clusterAll || item.visibleCount > MAX_INDIVIDUAL_MARKERS || groupSpacing < MIN_MARKER_SPACING_PX;

        if (shouldCluster) {
          drawnMarkers += drawClusteredMarkers(ctx, item, plot, yForPoint, withStroke);
          usedClusters = true;
        } else {
          drawnMarkers += drawIndividualMarkers(ctx, item, plot, yForPoint, withStroke);
        }
      }

      return { totalVisible, drawnMarkers, usedClusters };
    }

    function drawMarkerNote(ctx, plot, stats) {
      if (!stats.usedClusters) return;
      ctx.save();
      ctx.fillStyle = colors.axis;
      ctx.font = "12px Courier New";
      ctx.textAlign = "right";
      ctx.textBaseline = "top";
      ctx.fillText(
        "markers clustered: " + stats.drawnMarkers + " dots from " + stats.totalVisible + " events",
        plot.left + plot.width,
        plot.top + 4
      );
      ctx.restore();
    }

    function detailTarget(plot) {
      const visibleSpan = Math.max(1, xRange[1] - xRange[0]);
      const fullSpan = Math.max(visibleSpan, fullRange[1] - fullRange[0]);
      const zoom = Math.max(1, fullSpan / visibleSpan);
      const zoomBoost = Math.pow(Math.min(512, zoom), 0.45);
      const base = Math.max(70, Math.round(plot.width * 0.12));
      return Math.max(60, Math.min(2500, Math.round(base * zoomBoost)));
    }

    function decimateIndexesByMinMax(indexes, values, target) {
      if (indexes.length <= target) return indexes;
      if (target <= 3) return [indexes[0], indexes[indexes.length - 1]];
      const bucketCount = Math.max(1, Math.floor((target - 2) / 2));
      const bucketSize = (indexes.length - 2) / bucketCount;
      const out = [indexes[0]];
      for (let b = 0; b < bucketCount; b++) {
        const start = 1 + Math.floor(b * bucketSize);
        const end = Math.min(indexes.length - 1, 1 + Math.floor((b + 1) * bucketSize));
        if (start >= end) continue;
        let minIdx = indexes[start];
        let maxIdx = indexes[start];
        for (let n = start + 1; n < end; n++) {
          const idx = indexes[n];
          if ((values[idx] ?? 0) < (values[minIdx] ?? 0)) minIdx = idx;
          if ((values[idx] ?? 0) > (values[maxIdx] ?? 0)) maxIdx = idx;
        }
        if (minIdx === maxIdx) out.push(minIdx);
        else if (minIdx < maxIdx) out.push(minIdx, maxIdx);
        else out.push(maxIdx, minIdx);
      }
      const last = indexes[indexes.length - 1];
      if (out[out.length - 1] !== last) out.push(last);
      return out;
    }

    function drawSeriesLine(ctx, times, values, indexes, plot, yMin, yMax, color, lineWidth) {
      if (!indexes.length) return;
      const sampled = decimateIndexesByMinMax(indexes, values, detailTarget(plot));

      ctx.strokeStyle = color;
      ctx.lineWidth = lineWidth;
      ctx.beginPath();
      let started = false;

      for (const i of sampled) {
        if (!Number.isFinite(values[i])) continue;
        const x = scaleX(times[i], plot);
        const y = scaleY(values[i], yMin, yMax, plot);
        if (!started) {
          ctx.moveTo(x, y);
          started = true;
        } else {
          ctx.lineTo(x, y);
        }
      }

      ctx.stroke();
    }

    function nearestIndex(times, target) {
      if (!times.length) return null;
      const right = lowerBound(times, target);
      const left = Math.max(0, right - 1);
      if (right >= times.length) return left;
      return Math.abs(times[left] - target) <= Math.abs(times[right] - target) ? left : right;
    }

    function equityPlotFromRect(rect) {
      return { left: 62, top: 28, width: rect.width - 78, height: rect.height - 70 };
    }

    function equityYScale(indexes) {
      const e = report.equity;
      let yMin = Infinity;
      let yMax = -Infinity;
      for (const i of indexes) {
        const values = [e.model[i], e.buy_hold[i]];
        if (e.ma_baseline) values.push(e.ma_baseline[i]);
        for (const value of values) {
          if (!Number.isFinite(value)) continue;
          if (value < yMin) yMin = value;
          if (value > yMax) yMax = value;
        }
      }
      if (!Number.isFinite(yMin) || !Number.isFinite(yMax)) {
        yMin = 0;
        yMax = 1;
      }
      const pad = (yMax - yMin) * 0.08 || 1;
      return { yMin: yMin - pad, yMax: yMax + pad };
    }

    function hideTooltip() {
      tooltip.style.display = "none";
    }

    function updateEquityTooltip(event) {
      if (drag) {
        equityHoverIndex = null;
        hideTooltip();
        return;
      }

      const rect = equityCanvas.parentElement.getBoundingClientRect();
      const plot = equityPlotFromRect(rect);
      const x = event.clientX - rect.left;
      const y = event.clientY - rect.top;

      if (x < plot.left || x > plot.left + plot.width || y < plot.top || y > plot.top + plot.height) {
        equityHoverIndex = null;
        hideTooltip();
        redraw();
        return;
      }

      const targetTime = xRange[0] + ((x - plot.left) / plot.width) * (xRange[1] - xRange[0]);
      const idx = nearestIndex(report.equity.t, targetTime);
      if (idx === null || report.equity.t[idx] < xRange[0] || report.equity.t[idx] > xRange[1]) {
        equityHoverIndex = null;
        hideTooltip();
        redraw();
        return;
      }

      equityHoverIndex = idx;
      const modelValue = report.equity.model[idx];
      const buyHoldValue = report.equity.buy_hold[idx];
      const maValue = report.equity.ma_baseline ? report.equity.ma_baseline[idx] : null;
      const diff = modelValue - buyHoldValue;
      const maDiff = maValue === null ? null : modelValue - maValue;
      const when = new Date(report.equity.t[idx]).toISOString().replace("T", " ").slice(0, 16) + " UTC";

      const rows = [
        "<strong>" + when + "</strong>",
        "model equity: " + formatMoney(modelValue),
        "buy & hold: " + formatMoney(buyHoldValue),
        "model - buy hold: " + formatMoney(diff)
      ];
      if (maValue !== null) {
        rows.push("MA" + report.baseline_ma_window + " baseline: " + formatMoney(maValue));
        rows.push("model - MA baseline: " + formatMoney(maDiff));
      }
      tooltip.innerHTML = rows.join("<br>");
      tooltip.style.display = "block";
      tooltip.style.left = Math.min(event.clientX + 14, window.innerWidth - 300) + "px";
      tooltip.style.top = Math.min(event.clientY + 14, window.innerHeight - 110) + "px";
      redraw();
    }

    function drawPrice() {{
      const { ctx, width, height } = setupCanvas(priceCanvas);
      const plot = { left: 62, top: 28, width: width - 78, height: height - 70 };
      ctx.clearRect(0, 0, width, height);
      activeFrame = chooseFrame(xRange[0], xRange[1], plot.width);
      updateSummary();

      const c = report.candles[activeFrame];
      const indexes = visibleIndexes(c.t);
      if (!indexes.length) return;

      const highs = indexes.map((i) => c.high[i]);
      const lows = indexes.map((i) => c.low[i]);
      let yMin = Math.min(...lows);
      let yMax = Math.max(...highs);
      const pad = (yMax - yMin) * 0.08 || 1;
      yMin -= pad;
      yMax += pad;

      drawGrid(ctx, plot, yMin, yMax, "price");
      ctx.fillStyle = colors.text;
      ctx.font = "14px Georgia";
      ctx.fillText("Price + model buy/sell predictions", plot.left, 18);

      const candleWidth = Math.max(2, Math.min(12, (plot.width / indexes.length) * 0.7));
      for (const i of indexes) {
        const x = scaleX(c.t[i], plot);
        const yOpen = scaleY(c.open[i], yMin, yMax, plot);
        const yClose = scaleY(c.close[i], yMin, yMax, plot);
        const yHigh = scaleY(c.high[i], yMin, yMax, plot);
        const yLow = scaleY(c.low[i], yMin, yMax, plot);
        const up = c.close[i] >= c.open[i];
        ctx.strokeStyle = up ? colors.green : colors.red;
        ctx.fillStyle = up ? colors.greenSoft : colors.redSoft;
        ctx.beginPath();
        ctx.moveTo(x, yHigh);
        ctx.lineTo(x, yLow);
        ctx.stroke();
        const top = Math.min(yOpen, yClose);
        const bodyHeight = Math.max(1, Math.abs(yClose - yOpen));
        ctx.fillRect(x - candleWidth / 2, top, candleWidth, bodyHeight);
        ctx.strokeRect(x - candleWidth / 2, top, candleWidth, bodyHeight);
      }

      const markerStats = drawMarkerGroups(
        ctx,
        plot,
        (group, i) => scaleY(group.close[i], yMin, yMax, plot),
        true
      );
      drawMarkerNote(ctx, plot, markerStats);
      drawTestBoundary(ctx, plot);
      drawLegend(ctx, [
        { color: colors.buyWin, label: "buy won", width: 92 },
        { color: colors.buyLoss, label: "buy lost", width: 92 },
        { color: colors.sellGood, label: "sell good", width: 105 },
        { color: colors.sellBad, label: "sell bad", width: 100 },
        { color: colors.noLoss, label: "flat", width: 65 }
      ], plot.left + 4, plot.top + 8, plot.width - 8);
    }}

    function drawEquity() {{
      const { ctx, width, height } = setupCanvas(equityCanvas);
      const plot = { left: 62, top: 28, width: width - 78, height: height - 70 };
      ctx.clearRect(0, 0, width, height);

      const e = report.equity;
      const indexes = visibleIndexes(e.t);
      if (!indexes.length) return;

      const { yMin, yMax } = equityYScale(indexes);

      drawGrid(ctx, plot, yMin, yMax, "money");
      ctx.fillStyle = colors.text;
      ctx.font = "14px Georgia";
      ctx.fillText("Full-timeline equity: in-sample train + out-of-sample test", plot.left, 18);

      drawSeriesLine(ctx, e.t, e.model, indexes, plot, yMin, yMax, colors.modelEquity, 2);
      drawSeriesLine(ctx, e.t, e.buy_hold, indexes, plot, yMin, yMax, colors.buyHoldEquity, 2);
      if (e.ma_baseline) {
        drawSeriesLine(ctx, e.t, e.ma_baseline, indexes, plot, yMin, yMax, colors.maBaselineEquity, 2);
      }
      drawTestBoundary(ctx, plot);
      drawLegend(ctx, [
        { color: colors.modelEquity, label: "model equity", width: 125 },
        { color: colors.buyHoldEquity, label: "buy and hold", width: 130 },
        { color: colors.maBaselineEquity, label: "MA" + report.baseline_ma_window + " baseline", width: 135 }
      ], plot.left + 4, plot.top + 8, plot.width - 8);

      if (equityHoverIndex !== null && e.t[equityHoverIndex] >= xRange[0] && e.t[equityHoverIndex] <= xRange[1]) {
        const x = scaleX(e.t[equityHoverIndex], plot);
        const yModel = scaleY(e.model[equityHoverIndex], yMin, yMax, plot);
        const yBuyHold = scaleY(e.buy_hold[equityHoverIndex], yMin, yMax, plot);
        const yMa = e.ma_baseline ? scaleY(e.ma_baseline[equityHoverIndex], yMin, yMax, plot) : null;

        ctx.save();
        ctx.strokeStyle = "#3d443d";
        ctx.setLineDash([4, 4]);
        ctx.beginPath();
        ctx.moveTo(x, plot.top);
        ctx.lineTo(x, plot.top + plot.height);
        ctx.stroke();
        ctx.setLineDash([]);

        ctx.fillStyle = colors.modelEquity;
        ctx.strokeStyle = "#ffffff";
        ctx.lineWidth = 2;
        ctx.beginPath();
        ctx.arc(x, yModel, 5, 0, Math.PI * 2);
        ctx.fill();
        ctx.stroke();

        ctx.fillStyle = colors.buyHoldEquity;
        ctx.beginPath();
        ctx.arc(x, yBuyHold, 5, 0, Math.PI * 2);
        ctx.fill();
        ctx.stroke();

        if (yMa !== null) {
          ctx.fillStyle = colors.maBaselineEquity;
          ctx.beginPath();
          ctx.arc(x, yMa, 5, 0, Math.PI * 2);
          ctx.fill();
          ctx.stroke();
        }
        ctx.restore();
      }
    }}

    function drawPerformance() {{
      const { ctx, width, height } = setupCanvas(perfCanvas);
      const plot = { left: 62, top: 28, width: width - 78, height: height - 70 };
      ctx.clearRect(0, 0, width, height);
      drawGrid(ctx, plot, 0, 1, "prob");
      ctx.fillStyle = colors.text;
      ctx.font = "14px Georgia";
      ctx.fillText("Model probability + buy/sell outcomes", plot.left, 18);

      const p = report.predictions;
      const indexes = visibleIndexes(p.t);
      if (indexes.length) {
        const sampled = decimateIndexesByMinMax(indexes, p.prob_up, detailTarget(plot));
        ctx.strokeStyle = colors.prob;
        ctx.lineWidth = 1;
        ctx.beginPath();
        let started = false;
        for (const i of sampled) {
          const x = scaleX(p.t[i], plot);
          const y = scaleY(p.prob_up[i], 0, 1, plot);
          if (!started) {
            ctx.moveTo(x, y);
            started = true;
          } else {
            ctx.lineTo(x, y);
          }
        }
        ctx.stroke();
        if (p.prob_down && p.prob_down.length) {
          const sampledDown = decimateIndexesByMinMax(indexes, p.prob_down, detailTarget(plot));
          ctx.strokeStyle = colors.probDown;
          ctx.lineWidth = 1;
          ctx.beginPath();
          let downStarted = false;
          for (const i of sampledDown) {
            const x = scaleX(p.t[i], plot);
            const y = scaleY(p.prob_down[i], 0, 1, plot);
            if (!downStarted) {
              ctx.moveTo(x, y);
              downStarted = true;
            } else {
              ctx.lineTo(x, y);
            }
          }
          ctx.stroke();
        }
      }

      const thresholdY = scaleY(report.threshold, 0, 1, plot);
      ctx.strokeStyle = colors.threshold;
      ctx.setLineDash([6, 5]);
      ctx.beginPath();
      ctx.moveTo(plot.left, thresholdY);
      ctx.lineTo(plot.left + plot.width, thresholdY);
      ctx.stroke();
      ctx.setLineDash([]);

      if (report.trade_mode !== "long_only") {
        for (const [value, color, dash] of [
          [report.short_entry_threshold, colors.shortEntryThreshold, [7, 3]],
          [report.short_exit_threshold, colors.shortExitThreshold, [2, 4]]
        ]) {
          const y = scaleY(value, 0, 1, plot);
          ctx.strokeStyle = color; ctx.setLineDash(dash); ctx.beginPath();
          ctx.moveTo(plot.left, y); ctx.lineTo(plot.left + plot.width, y); ctx.stroke();
        }
        ctx.setLineDash([]);
      }

      const exitThresholdY = scaleY(report.exit_threshold, 0, 1, plot);
      ctx.strokeStyle = colors.exitThreshold;
      ctx.setLineDash([3, 5]);
      ctx.beginPath();
      ctx.moveTo(plot.left, exitThresholdY);
      ctx.lineTo(plot.left + plot.width, exitThresholdY);
      ctx.stroke();
      ctx.setLineDash([]);

      const markerStats = drawMarkerGroups(
        ctx,
        plot,
        (group, i) => scaleY(group.prob_down?.[i] ?? group.prob_up[i], 0, 1, plot),
        false
      );
      drawMarkerNote(ctx, plot, markerStats);
      drawTestBoundary(ctx, plot);
      drawLegend(ctx, [
        { color: colors.prob, label: "prob_up", width: 86 },
        { color: colors.probDown, label: "prob_down", width: 110 },
        { color: colors.threshold, label: "long entry", width: 105 },
        { color: colors.exitThreshold, label: "long exit", width: 100 },
        ...(report.trade_mode !== "long_only" ? [
          { color: colors.shortEntryThreshold, label: "short entry", width: 110 },
          { color: colors.shortExitThreshold, label: "short cover", width: 112 }
        ] : []),
        { color: colors.buyWin, label: "buy won", width: 92 },
        { color: colors.buyLoss, label: "buy lost", width: 92 },
        { color: colors.sellGood, label: "sell good", width: 105 },
        { color: colors.sellBad, label: "sell bad", width: 100 },
        { color: colors.noLoss, label: "flat", width: 65 }
      ], plot.left + 4, plot.top + 8, plot.width - 8);
    }}

    function redraw() {{
      drawPrice();
      drawPerformance();
      drawEquity();
    }}

    function scheduleRedraw() {{
      if (redrawPending) return;
      redrawPending = true;
      requestAnimationFrame(() => {
        redrawPending = false;
        redraw();
      });
    }}

    function clampRange() {{
      const minSpan = 30 * 60 * 1000;
      let span = xRange[1] - xRange[0];
      if (span < minSpan) {
        const mid = (xRange[0] + xRange[1]) / 2;
        xRange = [mid - minSpan / 2, mid + minSpan / 2];
      }
      span = xRange[1] - xRange[0];
      if (xRange[0] < fullRange[0]) xRange = [fullRange[0], fullRange[0] + span];
      if (xRange[1] > fullRange[1]) xRange = [fullRange[1] - span, fullRange[1]];
      if (xRange[0] < fullRange[0]) xRange[0] = fullRange[0];
      if (xRange[1] > fullRange[1]) xRange[1] = fullRange[1];
    }}

    function attachInteraction(canvas) {{
      canvas.addEventListener("wheel", (event) => {
        event.preventDefault();
        const rect = canvas.getBoundingClientRect();
        const ratio = (event.clientX - rect.left) / rect.width;
        const anchor = xRange[0] + ratio * (xRange[1] - xRange[0]);
        const factor = event.deltaY < 0 ? 0.75 : 1.35;
        const left = anchor - (anchor - xRange[0]) * factor;
        const right = anchor + (xRange[1] - anchor) * factor;
        xRange = [left, right];
        clampRange();
        scheduleRedraw();
      }, { passive: false });

      canvas.addEventListener("pointerdown", (event) => {
        drag = { x: event.clientX, range: [...xRange] };
        canvas.setPointerCapture(event.pointerId);
      });

      canvas.addEventListener("pointermove", (event) => {
        if (drag) {
          const rect = canvas.getBoundingClientRect();
          const dx = event.clientX - drag.x;
          const span = drag.range[1] - drag.range[0];
          const shift = -(dx / rect.width) * span;
          xRange = [drag.range[0] + shift, drag.range[1] + shift];
          clampRange();
          scheduleRedraw();
        }
      });

      canvas.addEventListener("pointerup", () => {
        drag = null;
      });

      canvas.addEventListener("dblclick", () => {
        xRange = [...fullRange];
        scheduleRedraw();
      });
    }}

    attachInteraction(priceCanvas);
    attachInteraction(perfCanvas);
    attachInteraction(equityCanvas);
    equityCanvas.addEventListener("pointermove", updateEquityTooltip);
    equityCanvas.addEventListener("pointerleave", () => {
      equityHoverIndex = null;
      hideTooltip();
      redraw();
    });
    window.addEventListener("resize", redraw);
    setupChartControls();
    redraw();
  </script>
</body>
</html>
"""
    template = template.replace("{{", "{").replace("}}", "}")
    return template.replace("__TITLE__", title_text).replace("__NAV__", nav).replace("__PAYLOAD__", payload)


def build_report(
    raw_data: Path,
    predictions_path: Path,
    output: Path,
    threshold: float,
    exit_threshold: float,
    trade_mode: str,
    short_entry_threshold: float,
    short_exit_threshold: float,
    fee: float,
    starting_cash: float,
    title: str,
    nav_home_url: str,
    nav_model_url: str,
    nav_sim_url: str,
    baseline_ma_window: int,
    max_browser_points: int,
) -> None:
    candles = load_candles(raw_data)
    predictions = load_predictions(
        predictions_path,
        threshold=threshold,
        exit_threshold=exit_threshold,
        fee=fee,
        trade_mode=trade_mode,
        short_entry_threshold=short_entry_threshold,
    )
    requested_base_frame = infer_base_interval_label(candles)
    candles_browser, base_frame = candle_payload(candles, requested_base_frame, max_browser_points)
    equity, equity_summary = equity_payload(
        predictions,
        starting_cash=starting_cash,
        fee=fee,
        baseline_ma_window=baseline_ma_window,
    )
    browser_indices = thin_indices(len(predictions), max_browser_points)
    predictions_browser = predictions.iloc[browser_indices].reset_index(drop=True)
    equity_browser = thin_equity_payload(equity, browser_indices)

    data = {
        "threshold": threshold,
        "exit_threshold": exit_threshold,
        "trade_mode": trade_mode,
        "short_entry_threshold": short_entry_threshold,
        "short_exit_threshold": short_exit_threshold,
        "fee": fee,
        "starting_cash": starting_cash,
        "baseline_ma_window": baseline_ma_window,
        "base_frame": base_frame,
        "test_start": predictions.attrs.get("test_start_open_time"),
        "candles": candles_browser,
        "predictions": predictions_payload(predictions_browser),
        "equity": equity_browser,
        "markers": marker_payload(predictions, max(1_000, max_browser_points // 4)),
        "summary": outcome_summary(predictions, fee=fee, equity_summary=equity_summary),
    }

    output.parent.mkdir(parents=True, exist_ok=True)
    nav = nav_html(nav_home_url, nav_model_url, nav_sim_url, active="model")
    output.write_text(html_template(title, data, nav), encoding="utf-8")


def main() -> None:
    args = parse_args()
    build_report(
        raw_data=Path(args.raw_data),
        predictions_path=Path(args.predictions),
        output=Path(args.output),
        threshold=args.threshold,
        exit_threshold=args.exit_threshold,
        trade_mode=args.trade_mode,
        short_entry_threshold=args.short_entry_threshold,
        short_exit_threshold=args.short_exit_threshold,
        fee=args.fee,
        starting_cash=args.starting_cash,
        title=args.title,
        nav_home_url=args.nav_home_url,
        nav_model_url=args.nav_model_url,
        nav_sim_url=args.nav_sim_url,
        baseline_ma_window=args.baseline_ma_window,
        max_browser_points=args.max_browser_points,
    )
    print(f"Saved visualization to {args.output}")


if __name__ == "__main__":
    main()
