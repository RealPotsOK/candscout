#!/usr/bin/env python3
"""Build leakage-safe candle sequence tensors for sequence models."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


SEQUENCE_CHANNELS = [
    "open_to_prev_close",
    "high_to_prev_close",
    "low_to_prev_close",
    "close_to_prev_close",
    "log_volume",
]


def load_raw_candles(path: Path) -> pd.DataFrame:
    df = pd.read_parquet(path)
    required = {"open_time", "open", "high", "low", "close", "volume"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Raw data missing required columns: {sorted(missing)}")

    df = df.sort_values("open_time").drop_duplicates("open_time").reset_index(drop=True)
    df["open_time"] = pd.to_datetime(df["open_time"], utc=True)
    return df


def build_candle_channel_matrix(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    open_ = df["open"].to_numpy(dtype=np.float64)
    high = df["high"].to_numpy(dtype=np.float64)
    low = df["low"].to_numpy(dtype=np.float64)
    close = df["close"].to_numpy(dtype=np.float64)
    volume = df["volume"].to_numpy(dtype=np.float64)

    prev_close = np.roll(close, 1)
    prev_close[0] = np.nan
    invalid_prev = ~np.isfinite(prev_close) | (prev_close <= 0.0)

    channels = np.column_stack(
        [
            open_ / prev_close - 1.0,
            high / prev_close - 1.0,
            low / prev_close - 1.0,
            close / prev_close - 1.0,
            np.log1p(np.maximum(volume, 0.0)),
        ]
    ).astype(np.float32)
    channels[invalid_prev, :4] = np.nan

    one_bar_return = close / prev_close - 1.0
    one_bar_return[invalid_prev] = np.nan
    return channels, one_bar_return.astype(np.float64)


def build_sequence_dataset(
    df: pd.DataFrame,
    lookback: int,
    edge: float,
) -> tuple[np.ndarray, np.ndarray, pd.DataFrame, list[str]]:
    if lookback < 2:
        raise ValueError("--lookback must be at least 2")

    n_rows = len(df)
    if n_rows < lookback + 2:
        raise ValueError(f"Need at least {lookback + 2} raw candles, found {n_rows}")

    channels, one_bar_return = build_candle_channel_matrix(df)
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
            "target": (forward_return[row_indices] > edge).astype(np.int64),
            "return_1bar": one_bar_return[row_indices],
            "sma_spread": sma_spread[row_indices],
        }
    )

    valid_mask = np.isfinite(x.reshape(n_examples, -1)).all(axis=1)
    valid_mask &= np.isfinite(meta["forward_return"].to_numpy(dtype=np.float64))
    valid_mask &= np.isfinite(meta["return_1bar"].to_numpy(dtype=np.float64))
    valid_mask &= np.isfinite(meta["sma_spread"].to_numpy(dtype=np.float64))

    x = x[valid_mask]
    meta = meta.loc[valid_mask].reset_index(drop=True)
    y = meta["target"].to_numpy(dtype=np.float32)

    return x, y, meta, list(SEQUENCE_CHANNELS)


def flatten_sequences(x: np.ndarray) -> np.ndarray:
    return x.reshape(x.shape[0], -1).astype(np.float32, copy=False)

