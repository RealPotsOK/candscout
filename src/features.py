#!/usr/bin/env python3
"""Generate leakage-safe features and labels for candle-level crypto prediction."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def parse_int_list(raw: str) -> list[int]:
    values = [int(x.strip()) for x in raw.split(",") if x.strip()]
    if not values:
        raise ValueError("Expected at least one integer value")
    if any(v <= 0 for v in values):
        raise ValueError("All window values must be positive")
    return values


def parse_optional_int_list(raw: str) -> list[int]:
    values = [int(x.strip()) for x in raw.split(",") if x.strip()]
    if any(v <= 0 for v in values):
        raise ValueError("All window values must be positive")
    return values


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build features and labels from raw candle data.")
    parser.add_argument("--input", required=True, help="Input raw candle Parquet path")
    parser.add_argument("--output", default="data/features_5m.parquet", help="Output feature Parquet path")
    parser.add_argument(
        "--meta-out",
        default=None,
        help="Optional metadata JSON output path (default: same as output with .meta.json)",
    )
    parser.add_argument(
        "--edge",
        type=float,
        default=0.002,
        help="Positive class edge on next-candle return (default: 0.002 = 0.20%%)",
    )
    parser.add_argument(
        "--short-edge",
        type=float,
        default=None,
        help="Short class edge on next-candle return. Defaults to --edge.",
    )
    parser.add_argument(
        "--interval",
        default="5m",
        help="Candle interval label used for metadata (default: 5m)",
    )
    parser.add_argument(
        "--return-windows",
        default="1,3,5,15,30,60",
        help="Comma-separated return windows in candles/bars",
    )
    parser.add_argument(
        "--vol-windows",
        default="5,15,30,60",
        help="Comma-separated volatility windows in candles/bars",
    )
    parser.add_argument("--sma-short-window", type=int, default=5, help="Short SMA window")
    parser.add_argument("--sma-long-window", type=int, default=20, help="Long SMA window")
    parser.add_argument(
        "--extra-sma-windows",
        default="50,100",
        help="Comma-separated extra SMA ratio windows in candles/bars",
    )
    parser.add_argument("--volume-z-window", type=int, default=20, help="Rolling window for volume z-score")
    parser.add_argument(
        "--volume-ratio-windows",
        default="20,60",
        help="Comma-separated volume/SMA ratio windows in candles/bars",
    )
    parser.add_argument(
        "--include-time-features",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Include UTC hour/day cyclical time features",
    )
    return parser.parse_args()


def load_raw_data(path: Path) -> pd.DataFrame:
    df = pd.read_parquet(path)
    required = {
        "open_time",
        "open",
        "high",
        "low",
        "close",
        "volume",
    }
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns in raw data: {sorted(missing)}")

    df = df.sort_values("open_time").drop_duplicates(subset=["open_time"]).reset_index(drop=True)
    return df


def build_features(
    df: pd.DataFrame,
    edge: float,
    short_edge: float | None,
    return_windows: list[int],
    vol_windows: list[int],
    sma_short_window: int,
    sma_long_window: int,
    extra_sma_windows: list[int],
    volume_z_window: int,
    volume_ratio_windows: list[int],
    include_time_features: bool,
) -> tuple[pd.DataFrame, list[str], dict]:
    if sma_short_window <= 0 or sma_long_window <= 0 or volume_z_window <= 0:
        raise ValueError("SMA and volume z-score windows must be positive")
    if sma_short_window >= sma_long_window:
        raise ValueError("--sma-short-window must be smaller than --sma-long-window")

    out = df.copy()
    short_edge_value = edge if short_edge is None else short_edge

    # Baseline helper columns expected by train/backtest logic.
    out["return_1bar"] = out["close"].pct_change(1)

    feature_columns: list[str] = []

    for window in return_windows:
        col = f"return_{window}bar"
        if window == 1:
            out[col] = out["return_1bar"]
        else:
            out[col] = out["close"].pct_change(window)
        feature_columns.append(col)

    for window in vol_windows:
        col = f"volatility_{window}bar"
        out[col] = out["return_1bar"].rolling(window).std()
        feature_columns.append(col)

    sma_short = out["close"].rolling(sma_short_window).mean()
    sma_long = out["close"].rolling(sma_long_window).mean()

    sma_short_ratio_col = f"sma_{sma_short_window}_ratio"
    sma_long_ratio_col = f"sma_{sma_long_window}_ratio"
    out[sma_short_ratio_col] = out["close"] / sma_short - 1.0
    out[sma_long_ratio_col] = out["close"] / sma_long - 1.0
    feature_columns.extend([sma_short_ratio_col, sma_long_ratio_col])

    # Keep a stable name for baseline MA direction logic.
    out["sma_spread"] = sma_short / sma_long - 1.0
    feature_columns.append("sma_spread")

    for window in extra_sma_windows:
        col = f"sma_{window}_ratio"
        sma = out["close"].rolling(window).mean()
        out[col] = out["close"] / sma - 1.0
        feature_columns.append(col)

    out["volume_change_1bar"] = out["volume"].pct_change(1)
    feature_columns.append("volume_change_1bar")

    volume_mean = out["volume"].rolling(volume_z_window).mean()
    volume_std = out["volume"].rolling(volume_z_window).std()
    volume_z_col = f"volume_zscore_{volume_z_window}"
    out[volume_z_col] = (out["volume"] - volume_mean) / (volume_std + 1e-12)
    feature_columns.append(volume_z_col)

    for window in volume_ratio_windows:
        col = f"volume_sma_ratio_{window}"
        volume_sma = out["volume"].rolling(window).mean()
        out[col] = out["volume"] / (volume_sma + 1e-12) - 1.0
        feature_columns.append(col)

    out["high_low_range"] = (out["high"] - out["low"]) / out["close"]
    out["close_open_range"] = (out["close"] - out["open"]) / out["open"]
    feature_columns.extend(["high_low_range", "close_open_range"])

    candle_range = out["high"] - out["low"]
    candle_range_safe = candle_range + 1e-12
    candle_body = out["close"] - out["open"]
    out["candle_body_pct"] = candle_body / out["open"]
    out["candle_body_abs_pct"] = candle_body.abs() / out["open"]
    out["upper_wick_pct"] = (out["high"] - out[["open", "close"]].max(axis=1)) / out["open"]
    out["lower_wick_pct"] = (out[["open", "close"]].min(axis=1) - out["low"]) / out["open"]
    out["close_position_in_range"] = (out["close"] - out["low"]) / candle_range_safe
    feature_columns.extend(
        [
            "candle_body_pct",
            "candle_body_abs_pct",
            "upper_wick_pct",
            "lower_wick_pct",
            "close_position_in_range",
        ]
    )

    if include_time_features:
        open_time = pd.to_datetime(out["open_time"], utc=True)
        hour_fraction = (
            open_time.dt.hour
            + open_time.dt.minute / 60.0
            + open_time.dt.second / 3600.0
        ) / 24.0
        day_fraction = open_time.dt.dayofweek / 7.0
        out["hour_sin"] = np.sin(2.0 * np.pi * hour_fraction)
        out["hour_cos"] = np.cos(2.0 * np.pi * hour_fraction)
        out["day_of_week_sin"] = np.sin(2.0 * np.pi * day_fraction)
        out["day_of_week_cos"] = np.cos(2.0 * np.pi * day_fraction)
        feature_columns.extend(["hour_sin", "hour_cos", "day_of_week_sin", "day_of_week_cos"])

    # Forward-looking value used only for label and backtest PnL construction.
    out["forward_return"] = out["close"].shift(-1) / out["close"] - 1.0
    out["target_up"] = (out["forward_return"] > edge).astype(int)
    out["target_down"] = (out["forward_return"] < -short_edge_value).astype(int)
    out["target"] = out["target_up"]

    # De-duplicate feature name list if any overlap occurs.
    feature_columns = list(dict.fromkeys(feature_columns))

    required_cols = feature_columns + [
        "forward_return",
        "target",
        "target_up",
        "target_down",
        "return_1bar",
        "sma_spread",
    ]
    out = out.replace([np.inf, -np.inf], np.nan)
    out = out.dropna(subset=required_cols).reset_index(drop=True)

    feature_config = {
        "return_windows": return_windows,
        "vol_windows": vol_windows,
        "sma_short_window": sma_short_window,
        "sma_long_window": sma_long_window,
        "extra_sma_windows": extra_sma_windows,
        "volume_z_window": volume_z_window,
        "volume_ratio_windows": volume_ratio_windows,
        "include_time_features": include_time_features,
    }

    return out, feature_columns, feature_config


def default_meta_path(output_path: Path) -> Path:
    if output_path.suffix:
        return output_path.with_suffix(".meta.json")
    return output_path.with_name(output_path.name + ".meta.json")


def main() -> None:
    args = parse_args()
    input_path = Path(args.input)
    output_path = Path(args.output)

    return_windows = parse_int_list(args.return_windows)
    vol_windows = parse_int_list(args.vol_windows)
    extra_sma_windows = parse_optional_int_list(args.extra_sma_windows)
    volume_ratio_windows = parse_optional_int_list(args.volume_ratio_windows)

    df = load_raw_data(input_path)
    features_df, feature_columns, feature_config = build_features(
        df,
        edge=args.edge,
        short_edge=args.short_edge,
        return_windows=return_windows,
        vol_windows=vol_windows,
        sma_short_window=args.sma_short_window,
        sma_long_window=args.sma_long_window,
        extra_sma_windows=extra_sma_windows,
        volume_z_window=args.volume_z_window,
        volume_ratio_windows=volume_ratio_windows,
        include_time_features=args.include_time_features,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    features_df.to_parquet(output_path, index=False)

    meta_path = Path(args.meta_out) if args.meta_out else default_meta_path(output_path)
    meta_path.parent.mkdir(parents=True, exist_ok=True)
    metadata = {
        "edge": float(args.edge),
        "short_edge": float(args.edge if args.short_edge is None else args.short_edge),
        "interval": str(args.interval),
        "target_horizon_bars": 1,
        "target_definition": "target/target_up=1 when next-candle forward_return > edge, else 0",
        "target_down_definition": "target_down=1 when next-candle forward_return < -short_edge, else 0",
        "feature_columns": feature_columns,
        "feature_config": feature_config,
    }
    meta_path.write_text(json.dumps(metadata, indent=2))

    positive_rate = float(features_df["target"].mean())
    down_rate = float(features_df["target_down"].mean())
    print(f"Saved {len(features_df)} feature rows to {output_path}")
    print(f"Saved feature metadata to {meta_path}")
    print(f"Positive class rate (y=1): {positive_rate:.6f}")
    print(f"Negative class rate (y=0): {1.0 - positive_rate:.6f}")
    print(f"Down class rate (target_down=1): {down_rate:.6f}")
    print(f"Feature columns used ({len(feature_columns)}): {feature_columns}")


if __name__ == "__main__":
    main()
