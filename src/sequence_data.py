#!/usr/bin/env python3
"""Build leakage-safe candle sequence tensors for sequence models."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


BASIC_SEQUENCE_CHANNELS = [
    "open_to_prev_close",
    "high_to_prev_close",
    "low_to_prev_close",
    "close_to_prev_close",
    "log_volume",
]

TECHNICAL_SEQUENCE_CHANNELS = [
    *BASIC_SEQUENCE_CHANNELS,
    "return_1bar",
    "return_3bar",
    "return_5bar",
    "return_10bar",
    "return_20bar",
    "volatility_10bar",
    "volatility_20bar",
    "sma_10_ratio",
    "sma_20_ratio",
    "sma_50_ratio",
    "ema_12_ratio",
    "ema_26_ratio",
    "macd_pct",
    "rsi_14",
    "volume_sma_ratio_20",
    "close_position_in_range_20",
]

SEQUENCE_CHANNELS = BASIC_SEQUENCE_CHANNELS


def load_raw_candles(path: Path) -> pd.DataFrame:
    df = pd.read_parquet(path)
    required = {"open_time", "open", "high", "low", "close", "volume"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Raw data missing required columns: {sorted(missing)}")

    df = df.sort_values("open_time").drop_duplicates("open_time").reset_index(drop=True)
    df["open_time"] = pd.to_datetime(df["open_time"], utc=True)
    return df


def channel_names_for_feature_set(feature_set: str) -> list[str]:
    if feature_set == "basic":
        return list(BASIC_SEQUENCE_CHANNELS)
    if feature_set == "technical":
        return list(TECHNICAL_SEQUENCE_CHANNELS)
    raise ValueError("--sequence-feature-set must be basic or technical")


def build_candle_feature_frame(df: pd.DataFrame) -> tuple[pd.DataFrame, np.ndarray]:
    open_ = df["open"].to_numpy(dtype=np.float64)
    high = df["high"].to_numpy(dtype=np.float64)
    low = df["low"].to_numpy(dtype=np.float64)
    close = df["close"].to_numpy(dtype=np.float64)
    volume = df["volume"].to_numpy(dtype=np.float64)

    prev_close = np.roll(close, 1)
    prev_close[0] = np.nan
    invalid_prev = ~np.isfinite(prev_close) | (prev_close <= 0.0)

    one_bar_return = close / prev_close - 1.0
    one_bar_return[invalid_prev] = np.nan

    close_s = pd.Series(close)
    high_s = pd.Series(high)
    low_s = pd.Series(low)
    volume_s = pd.Series(volume)
    return_s = pd.Series(one_bar_return)

    ema_12 = close_s.ewm(span=12, adjust=False, min_periods=12).mean()
    ema_26 = close_s.ewm(span=26, adjust=False, min_periods=26).mean()
    avg_gain = return_s.clip(lower=0.0).rolling(14).mean()
    avg_loss = (-return_s.clip(upper=0.0)).rolling(14).mean()
    rs = avg_gain / avg_loss.replace(0.0, np.nan)
    rsi = 100.0 - (100.0 / (1.0 + rs))
    rsi = rsi.where(avg_loss != 0.0, 100.0)
    range_high_20 = high_s.rolling(20).max()
    range_low_20 = low_s.rolling(20).min()
    range_width_20 = (range_high_20 - range_low_20).replace(0.0, np.nan)
    volume_sma_20 = volume_s.rolling(20).mean().replace(0.0, np.nan)

    features = pd.DataFrame(
        {
            "open_to_prev_close": open_ / prev_close - 1.0,
            "high_to_prev_close": high / prev_close - 1.0,
            "low_to_prev_close": low / prev_close - 1.0,
            "close_to_prev_close": close / prev_close - 1.0,
            "log_volume": np.log1p(np.maximum(volume, 0.0)),
            "return_1bar": one_bar_return,
            "return_3bar": close_s / close_s.shift(3) - 1.0,
            "return_5bar": close_s / close_s.shift(5) - 1.0,
            "return_10bar": close_s / close_s.shift(10) - 1.0,
            "return_20bar": close_s / close_s.shift(20) - 1.0,
            "volatility_10bar": return_s.rolling(10).std(ddof=0),
            "volatility_20bar": return_s.rolling(20).std(ddof=0),
            "sma_10_ratio": close_s / close_s.rolling(10).mean() - 1.0,
            "sma_20_ratio": close_s / close_s.rolling(20).mean() - 1.0,
            "sma_50_ratio": close_s / close_s.rolling(50).mean() - 1.0,
            "ema_12_ratio": close_s / ema_12 - 1.0,
            "ema_26_ratio": close_s / ema_26 - 1.0,
            "macd_pct": (ema_12 - ema_26) / close_s,
            "rsi_14": (rsi / 100.0) - 0.5,
            "volume_sma_ratio_20": volume_s / volume_sma_20 - 1.0,
            "close_position_in_range_20": ((close_s - range_low_20) / range_width_20) - 0.5,
        }
    )
    features.loc[invalid_prev, ["open_to_prev_close", "high_to_prev_close", "low_to_prev_close", "close_to_prev_close"]] = np.nan
    return features.astype(np.float32), one_bar_return.astype(np.float64)


def build_sequence_dataset(
    df: pd.DataFrame,
    lookback: int,
    edge: float,
    short_edge: float | None = None,
    feature_set: str = "basic",
    channel_names: list[str] | None = None,
) -> tuple[np.ndarray, np.ndarray, pd.DataFrame, list[str]]:
    if lookback < 2:
        raise ValueError("--lookback must be at least 2")

    n_rows = len(df)
    if n_rows < lookback + 2:
        raise ValueError(f"Need at least {lookback + 2} raw candles, found {n_rows}")

    requested_channels = list(channel_names) if channel_names is not None else channel_names_for_feature_set(feature_set)
    short_edge_value = edge if short_edge is None else short_edge
    feature_frame, one_bar_return = build_candle_feature_frame(df)
    missing_channels = set(requested_channels) - set(feature_frame.columns)
    if missing_channels:
        raise ValueError(f"Unknown sequence channels requested: {sorted(missing_channels)}")

    channels = feature_frame[requested_channels].to_numpy(dtype=np.float32)
    close = df["close"].to_numpy(dtype=np.float64)
    forward_return = np.roll(close, -1) / close - 1.0
    forward_return[-1] = np.nan

    sma_short = pd.Series(close).rolling(5).mean().to_numpy()
    sma_long = pd.Series(close).rolling(20).mean().to_numpy()
    sma_spread = sma_short / sma_long - 1.0

    # Example ending at candle i uses rows i-lookback+1..i, then predicts i+1.
    row_indices = np.arange(lookback, n_rows - 1)
    n_examples = len(row_indices)
    n_channels = channels.shape[1]
    x = np.empty((n_examples, lookback, n_channels), dtype=np.float32)

    for out_idx, candle_idx in enumerate(row_indices):
        start = candle_idx - lookback + 1
        end = candle_idx + 1
        x[out_idx] = channels[start:end]

    meta = pd.DataFrame(
        {
            "open_time": df.loc[row_indices, "open_time"].to_numpy(),
            "sequence_start_time": df.loc[row_indices - lookback + 1, "open_time"].to_numpy(),
            "close": close[row_indices],
            "forward_return": forward_return[row_indices],
            "target_up": (forward_return[row_indices] > edge).astype(np.int64),
            "target_down": (forward_return[row_indices] < -short_edge_value).astype(np.int64),
            "return_1bar": one_bar_return[row_indices],
            "sma_spread": sma_spread[row_indices],
        }
    )
    meta["target"] = meta["target_up"]

    valid_mask = np.isfinite(x.reshape(n_examples, -1)).all(axis=1)
    valid_mask &= np.isfinite(meta["forward_return"].to_numpy(dtype=np.float64))
    valid_mask &= np.isfinite(meta["return_1bar"].to_numpy(dtype=np.float64))
    valid_mask &= np.isfinite(meta["sma_spread"].to_numpy(dtype=np.float64))

    x = x[valid_mask]
    meta = meta.loc[valid_mask].reset_index(drop=True)
    y = meta["target"].to_numpy(dtype=np.float32)

    return x, y, meta, requested_channels


def flatten_sequences(x: np.ndarray) -> np.ndarray:
    return x.reshape(x.shape[0], -1).astype(np.float32, copy=False)
