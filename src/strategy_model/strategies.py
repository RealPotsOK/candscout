"""Shared helpers for rule-based strategy models."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

SRC_ROOT = Path(__file__).resolve().parents[1]
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from backtest import annualized_return, max_drawdown  # noqa: E402

STRATEGY_LABELS = {
    "buy_hold": "Buy and Hold",
    "prev_movement": "Previous Movement",
    "ma": "Moving Average",
    "counter_ma": "Counter MA",
    "counter_ma_opt": "Optimized Counter MA",
}


def load_raw_candles(path: Path, edge: float) -> pd.DataFrame:
    df = pd.read_parquet(path)
    required = {"open_time", "open", "high", "low", "close", "volume"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Raw candle file missing required columns: {sorted(missing)}")

    df = df.sort_values("open_time").drop_duplicates("open_time").reset_index(drop=True)
    df["open_time"] = pd.to_datetime(df["open_time"], utc=True)
    df["forward_return"] = df["close"].shift(-1) / df["close"] - 1.0
    df = df.dropna(subset=["forward_return"]).reset_index(drop=True)
    df["target_up"] = (df["forward_return"] > edge).astype(int)
    df["target_down"] = (df["forward_return"] < -edge).astype(int)
    df["target"] = df["target_up"]
    return df


def infer_bar_seconds(open_times: pd.Series) -> float:
    times = pd.to_datetime(open_times, utc=True)
    diffs = times.diff().dropna().dt.total_seconds()
    if diffs.empty:
        return 300.0
    return float(diffs.median())


def bars_per_year_from_seconds(bar_seconds: float) -> int:
    if bar_seconds <= 0.0:
        return 105_120
    return int(round((365.0 * 24.0 * 60.0 * 60.0) / bar_seconds))


def moving_average(close: pd.Series, window: int) -> pd.Series:
    if window < 1:
        raise ValueError("MA window must be >= 1")
    # Shift by one candle so the score only uses current close plus prior MA.
    return close.rolling(window=window, min_periods=window).mean().shift(1)


def ma_scores(df: pd.DataFrame, window: int) -> np.ndarray:
    close = df["close"].astype(float)
    ma = moving_average(close, window)
    scores = np.full(len(df), 0.5, dtype=np.float64)
    valid = ma.notna() & (ma > 0.0)
    scores[valid & (close > ma)] = 0.8
    scores[valid & (close < ma)] = 0.2
    return scores


def previous_movement_scores(df: pd.DataFrame) -> np.ndarray:
    returns = df["close"].astype(float).pct_change()
    scores = np.full(len(df), 0.5, dtype=np.float64)
    scores[returns > 0.0] = 0.8
    scores[returns < 0.0] = 0.2
    return scores


def counter_ma_scores(df: pd.DataFrame, window: int, band: float) -> np.ndarray:
    close = df["close"].astype(float)
    ma = moving_average(close, window)
    distance = close / ma - 1.0
    distance_delta = distance.diff()
    scores = np.full(len(df), 0.5, dtype=np.float64)
    valid = ma.notna() & np.isfinite(distance) & np.isfinite(distance_delta)
    if band > 0.0:
        valid = valid & (distance.abs() <= band)

    # Counter-MA logic: try to front-run MA crowd behavior near the average.
    # Buy before a bullish MA cross when price is below MA but moving upward.
    # Exit/avoid before a bearish MA cross when price is above MA but moving down.
    buy = valid & (distance < 0.0) & (distance_delta > 0.0)
    sell = valid & (distance > 0.0) & (distance_delta < 0.0)
    scores[buy] = 0.8
    scores[sell] = 0.2
    return scores


def strategy_scores(df: pd.DataFrame, model_type: str, window: int, counter_band: float) -> np.ndarray:
    if model_type == "buy_hold":
        return np.ones(len(df), dtype=np.float64)
    if model_type == "prev_movement":
        return previous_movement_scores(df)
    if model_type == "ma":
        return ma_scores(df, window)
    if model_type in {"counter_ma", "counter_ma_opt"}:
        return counter_ma_scores(df, window, counter_band)
    raise ValueError(f"Unsupported strategy model type: {model_type}")


def strategy_down_scores(df: pd.DataFrame, model_type: str, window: int, counter_band: float) -> np.ndarray:
    if model_type == "buy_hold":
        return np.zeros(len(df), dtype=np.float64)
    scores = strategy_scores(df, model_type, window, counter_band)
    return 1.0 - scores


def split_frame(df: pd.DataFrame, split: float) -> tuple[pd.DataFrame, pd.DataFrame, int]:
    if not (0.0 < split < 1.0):
        raise ValueError("--split must be between 0 and 1")
    split_idx = int(len(df) * split)
    split_idx = max(1, min(len(df) - 1, split_idx))
    return df.iloc[:split_idx].copy(), df.iloc[split_idx:].copy(), split_idx


def read_model(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_model(path: Path, model: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(model, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def buy_hold_arrays(closes: np.ndarray, targets: np.ndarray, fee: float, bars_per_year: int) -> tuple[dict, dict[str, np.ndarray]]:
    n_rows = len(closes)
    position = np.ones(n_rows, dtype=np.int64)
    entry_signal = np.zeros(n_rows, dtype=np.int64)
    exit_signal = np.zeros(n_rows, dtype=np.int64)
    trade_id = np.ones(n_rows, dtype=np.int64)
    realized_trade_return = np.full(n_rows, np.nan, dtype=np.float64)
    entry_trade_net_return = np.full(n_rows, np.nan, dtype=np.float64)
    bars_held_at_exit = np.full(n_rows, np.nan, dtype=np.float64)
    exit_reason = np.array([""] * n_rows, dtype=object)

    if n_rows == 0:
        return empty_trade_report(), {
            "position": position,
            "entry_signal": entry_signal,
            "exit_signal": exit_signal,
            "trade_id": trade_id,
            "realized_trade_return": realized_trade_return,
            "entry_trade_net_return": entry_trade_net_return,
            "bars_held_at_exit": bars_held_at_exit,
            "exit_reason": exit_reason,
        }

    entry_signal[0] = 1
    exit_signal[-1] = 1
    gross_return = float(closes[-1] / closes[0] - 1.0) if closes[0] > 0.0 else 0.0
    net_return = gross_return - (2.0 * fee)
    realized_trade_return[-1] = net_return
    entry_trade_net_return[0] = net_return
    bars_held_at_exit[-1] = max(0, n_rows - 1)
    exit_reason[-1] = "end_of_data"

    mark_to_market = closes / closes[0] if closes[0] > 0.0 else np.ones(n_rows, dtype=np.float64)
    equity_curve = mark_to_market.copy()
    equity_curve[-1] = max(1e-12, equity_curve[-1] - (2.0 * fee))
    total_return = float(net_return)
    report = {
        "trade_count": 1,
        "hit_rate": float(net_return > 0.0),
        "label_hit_rate": float(int(targets[0] == 1)) if len(targets) else 0.0,
        "profitable_trade_rate": float(net_return > 0.0),
        "avg_gross_return_per_trade": gross_return,
        "avg_net_return_per_trade": net_return,
        "best_net_return": net_return,
        "worst_net_return": net_return,
        "avg_bars_held": float(max(0, n_rows - 1)),
        "total_return": total_return,
        "annualized_return_proxy": annualized_return(total_return, n_rows, bars_per_year),
        "max_drawdown": max_drawdown(equity_curve) if len(equity_curve) else 0.0,
        "exit_reason_counts": {"end_of_data": 1},
    }
    arrays = {
        "position": position,
        "entry_signal": entry_signal,
        "exit_signal": exit_signal,
        "trade_id": trade_id,
        "realized_trade_return": realized_trade_return,
        "entry_trade_net_return": entry_trade_net_return,
        "bars_held_at_exit": bars_held_at_exit,
        "exit_reason": exit_reason,
    }
    return report, arrays


def empty_trade_report() -> dict:
    return {
        "trade_count": 0,
        "hit_rate": 0.0,
        "label_hit_rate": 0.0,
        "profitable_trade_rate": 0.0,
        "avg_gross_return_per_trade": 0.0,
        "avg_net_return_per_trade": 0.0,
        "avg_bars_held": 0.0,
        "total_return": 0.0,
        "annualized_return_proxy": 0.0,
        "max_drawdown": 0.0,
        "exit_reason_counts": {},
    }


def prediction_frame(
    df: pd.DataFrame,
    scores: np.ndarray,
    down_scores: np.ndarray,
    threshold: float,
    arrays: dict[str, np.ndarray],
) -> pd.DataFrame:
    predictions = pd.DataFrame(
        {
            "open_time": df["open_time"],
            "close": df["close"],
            "target": df["target"].astype(int),
            "target_up": df["target_up"].astype(int),
            "target_down": df["target_down"].astype(int),
            "forward_return": df["forward_return"],
            "prob_up": scores,
            "prob_down": down_scores,
            "predicted_class_at_0_50": (scores >= 0.50).astype(int),
            "signal_at_threshold": (scores >= threshold).astype(int),
        }
    )
    for key, values in arrays.items():
        predictions[key] = values
    return predictions
