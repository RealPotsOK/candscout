"""Live sequence-model loading and feature construction."""

from __future__ import annotations

import threading
from pathlib import Path

import numpy as np
import pandas as pd

from sequence_data import BASIC_SEQUENCE_CHANNELS, build_candle_feature_frame
from sequence_nn import load_sequence_model, predict_loaded_sequence_model

from .market import Candle

FEATURE_WARMUP_BY_CHANNEL = {
    "open_to_prev_close": 1,
    "high_to_prev_close": 1,
    "low_to_prev_close": 1,
    "close_to_prev_close": 1,
    "log_volume": 0,
    "return_1bar": 1,
    "return_3bar": 3,
    "return_5bar": 5,
    "return_10bar": 10,
    "return_20bar": 20,
    "volatility_10bar": 10,
    "volatility_20bar": 20,
    "sma_10_ratio": 10,
    "sma_20_ratio": 20,
    "sma_50_ratio": 50,
    "ema_12_ratio": 12,
    "ema_26_ratio": 26,
    "macd_pct": 26,
    "rsi_14": 14,
    "volume_sma_ratio_20": 20,
    "close_position_in_range_20": 20,
}


class LiveModel:
    def __init__(self, model_path: str) -> None:
        path = Path(model_path)
        if not path.exists():
            raise FileNotFoundError(f"Model artifact not found: {path}")
        self.path = path
        self.lock = threading.RLock()
        self.model: dict | None = None
        self.model_mtime_ns = 0
        self.lookback = 0
        self.model_type = ""
        self.channel_names: list[str] = []
        self.load()

    def load(self) -> None:
        model = load_sequence_model(self.path)
        channel_names = list(model["channel_names"])
        missing_channels = sorted(set(channel_names) - set(FEATURE_WARMUP_BY_CHANNEL))
        if missing_channels:
            raise ValueError(f"Saved model uses unsupported live channels: {missing_channels}")
        stat = self.path.stat()
        self.model = model
        self.model_mtime_ns = stat.st_mtime_ns
        self.lookback = int(model["lookback"])
        self.model_type = str(model["model_type"])
        self.channel_names = channel_names

    def maybe_reload(self) -> bool:
        with self.lock:
            current_mtime = self.path.stat().st_mtime_ns
            if current_mtime == self.model_mtime_ns:
                return False
            self.load()
            print(f"Reloaded active model from {self.path}", flush=True)
            return True

    def info(self) -> dict:
        with self.lock:
            return {
                "path": str(self.path),
                "model_type": self.model_type,
                "lookback": self.lookback,
                "channels": self.channel_names,
                "edge": None if self.model is None else self.model.get("edge"),
                "mtime_ns": self.model_mtime_ns,
            }

    def required_candles(self, buffer: int = 8) -> int:
        with self.lock:
            warmup = max((FEATURE_WARMUP_BY_CHANNEL.get(name, 0) for name in self.channel_names), default=1)
            return self.lookback + warmup + max(buffer, 1)

    def predict(self, candles: list[Candle]) -> tuple[float, Candle]:
        with self.lock:
            self.maybe_reload()
            if self.model is None:
                raise RuntimeError("Model is not loaded")
            x_seq, last_candle = build_sequence_input(candles, self.lookback, self.channel_names)
            prob = float(predict_loaded_sequence_model(self.model, x_seq, batch_size=1)[0])
            return prob, last_candle


def build_sequence_input(
    candles: list[Candle],
    lookback: int,
    channel_names: list[str] | None = None,
) -> tuple[np.ndarray, Candle]:
    if lookback < 2:
        raise ValueError("lookback must be at least 2")
    requested_channels = list(channel_names or BASIC_SEQUENCE_CHANNELS)
    warmup = max((FEATURE_WARMUP_BY_CHANNEL.get(name, 0) for name in requested_channels), default=1)
    min_candles = lookback + warmup
    if len(candles) < min_candles:
        raise ValueError(f"Need at least {min_candles} completed candles, found {len(candles)}")

    frame = candles_to_frame(candles)
    feature_frame, _one_bar_return = build_candle_feature_frame(frame)
    missing_channels = sorted(set(requested_channels) - set(feature_frame.columns))
    if missing_channels:
        raise ValueError(f"Unsupported live sequence channels: {missing_channels}")

    values = feature_frame[requested_channels].to_numpy(dtype=np.float32)
    x_window = values[-lookback:]
    x_seq = x_window.reshape(1, lookback, len(requested_channels))
    if not np.isfinite(x_seq).all():
        raise ValueError("Live sequence contains NaN or infinite values")
    return x_seq, candles[-1]


def candles_to_frame(candles: list[Candle]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "open_time": [c.open_time for c in candles],
            "open": [c.open for c in candles],
            "high": [c.high for c in candles],
            "low": [c.low for c in candles],
            "close": [c.close for c in candles],
            "volume": [c.volume for c in candles],
        }
    )
