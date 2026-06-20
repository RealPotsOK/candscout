#!/usr/bin/env python3
"""Backtest XGBoost probabilities with the same long/cash rules as other models."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

SRC_ROOT = Path(__file__).resolve().parents[1]
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from backtest import (  # noqa: E402
    annualized_return,
    bars_per_year_from_seconds,
    chronological_split_index,
    dataset_split_labels,
    infer_bar_seconds,
    max_drawdown,
    resolve_column,
    simulate_hold_positions,
    simulate_independent_one_bar_trades,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Backtest XGBoost long-or-cash model signals.")
    parser.add_argument("--features", required=True, help="Feature Parquet path")
    parser.add_argument("--model", required=True, help="Trained XGBoost model path")
    parser.add_argument("--fee", type=float, default=0.001, help="Per-side fee rate")
    parser.add_argument("--threshold", type=float, default=0.55, help="Entry probability threshold")
    parser.add_argument("--split", type=float, default=0.8, help="Chronological train fraction")
    parser.add_argument(
        "--position-mode",
        choices=["one_bar", "hold"],
        default="hold",
        help="Backtest logic: independent one-bar trades or real hold/exit position logic",
    )
    parser.add_argument("--exit-threshold", type=float, default=0.50, help="Exit when score drops below this value")
    parser.add_argument("--max-hold-bars", type=int, default=60, help="Max bars to hold a position; 0 disables")
    parser.add_argument("--stop-loss", type=float, default=0.002, help="Gross stop loss from entry; 0 disables")
    parser.add_argument("--take-profit", type=float, default=0.004, help="Gross take profit from entry; 0 disables")
    parser.add_argument("--report-out", default="models/xgb/backtest_report.json", help="Backtest report JSON output")
    parser.add_argument("--predictions-out", default="data/reports/xgb/predictions.parquet", help="Predictions Parquet output")
    return parser.parse_args()


def import_xgboost():
    try:
        import xgboost as xgb  # type: ignore
    except ModuleNotFoundError as exc:
        raise SystemExit(
            "Missing dependency: xgboost. Install it with `make install` or `./.venv/bin/pip install xgboost`."
        ) from exc
    return xgb


def load_model_and_features(model_path: Path):
    xgb = import_xgboost()
    booster = xgb.Booster()
    booster.load_model(model_path)
    raw_features = booster.attr("feature_columns")
    if raw_features:
        feature_names = [str(x) for x in json.loads(raw_features)]
    elif booster.feature_names:
        feature_names = [str(x) for x in booster.feature_names]
    else:
        raise ValueError("XGBoost model does not contain feature column metadata.")
    return xgb, booster, feature_names


def load_optional_down_model(xgb, model_path: Path, feature_names: list[str]):
    down_path = model_path.with_suffix(".down.json")
    if not down_path.exists():
        return None
    booster = xgb.Booster()
    booster.load_model(down_path)
    raw_features = booster.attr("feature_columns")
    down_features = [str(x) for x in json.loads(raw_features)] if raw_features else list(booster.feature_names or [])
    if down_features != feature_names:
        raise ValueError(f"Down XGBoost feature mismatch. Up={feature_names}, down={down_features}")
    return booster


def one_bar_equity(forward_returns: np.ndarray, signals: np.ndarray, fee: float) -> np.ndarray:
    returns = np.where(signals == 1, forward_returns - (2.0 * fee), 0.0)
    return np.cumprod(1.0 + returns) if len(returns) else np.array([], dtype=np.float64)


def main() -> None:
    args = parse_args()
    features_df = pd.read_parquet(args.features).sort_values("open_time").reset_index(drop=True)
    required_cols = {"open_time", "close", "target", "sma_spread"}
    missing = required_cols - set(features_df.columns)
    if missing:
        raise ValueError(f"Feature file missing required columns: {sorted(missing)}")

    forward_return_col = resolve_column(features_df, ["forward_return", "forward_return_1m"], "next-candle forward return")
    one_bar_return_col = resolve_column(features_df, ["return_1bar", "return_1m"], "previous-candle direction baseline")
    bar_seconds = infer_bar_seconds(features_df["open_time"])
    bars_per_year = bars_per_year_from_seconds(bar_seconds)

    model_path = Path(args.model)
    xgb, booster, feature_names = load_model_and_features(model_path)
    booster_down = load_optional_down_model(xgb, model_path, feature_names)
    missing_model_cols = set(feature_names) - set(features_df.columns)
    if missing_model_cols:
        raise ValueError(f"Feature file missing model columns: {sorted(missing_model_cols)}")

    matrix = xgb.DMatrix(features_df[feature_names], feature_names=feature_names)
    probs = booster.predict(matrix).astype(np.float64)
    probs_down = booster_down.predict(matrix).astype(np.float64) if booster_down is not None else np.zeros(len(features_df), dtype=np.float64)
    model_signal = (probs >= args.threshold).astype(int)
    predicted_class_at_050 = (probs >= 0.50).astype(int)
    forward_returns = features_df[forward_return_col].to_numpy(dtype=np.float64)
    targets = features_df["target"].to_numpy(dtype=np.int64)
    target_up = features_df["target_up"].to_numpy(dtype=np.int64) if "target_up" in features_df.columns else targets
    target_down = features_df["target_down"].to_numpy(dtype=np.int64) if "target_down" in features_df.columns else np.zeros(len(features_df), dtype=np.int64)
    closes = features_df["close"].to_numpy(dtype=np.float64)
    split_idx = chronological_split_index(len(features_df), args.split)
    eval_slice = slice(split_idx, None)

    scores = {
        "xgboost": probs,
        "always_positive": np.ones(len(features_df), dtype=np.float64),
        "always_negative": np.zeros(len(features_df), dtype=np.float64),
        "prev_candle_direction": (features_df[one_bar_return_col].to_numpy() > 0.0).astype(float),
        "ma_direction": (features_df["sma_spread"].to_numpy() > 0.0).astype(float),
    }
    signals = {name: (score_array >= args.threshold).astype(int) for name, score_array in scores.items()}

    report = {
        "assumptions": {
            "model_type": "xgboost",
            "fee_per_side": float(args.fee),
            "entry_threshold": float(args.threshold),
            "position_mode": args.position_mode,
            "exit_threshold": float(args.exit_threshold),
            "max_hold_bars": int(args.max_hold_bars),
            "stop_loss": float(args.stop_loss),
            "take_profit": float(args.take_profit),
            "trade_return_definition": "gross close-to-close return minus round-trip fees",
            "bar_seconds": float(bar_seconds),
            "bars_per_year": int(bars_per_year),
            "forward_return_column": forward_return_col,
            "one_bar_return_column": one_bar_return_col,
            "dual_model": bool(booster_down is not None),
        },
        "rows": int(len(features_df)),
        "evaluation_rows": int(len(features_df) - split_idx),
        "evaluation_split": "chronological_test",
        "split_fraction": float(args.split),
        "test_start_open_time": str(features_df["open_time"].iloc[split_idx]),
        "backtest": {},
    }

    model_position_arrays: dict[str, np.ndarray] | None = None
    if args.position_mode == "one_bar":
        report["assumptions"]["position_logic"] = (
            "independent one-bar trades: if signal=1 at t, enter close[t], exit close[t+1], subtract round-trip fees"
        )
        for name, signal_array in signals.items():
            report["backtest"][name] = simulate_independent_one_bar_trades(
                signals=signal_array[eval_slice],
                targets=targets[eval_slice],
                forward_returns=forward_returns[eval_slice],
                fee=args.fee,
                bars_per_year=bars_per_year,
            )
    else:
        report["assumptions"]["position_logic"] = (
            "hold one long position after entry until exit threshold, stop loss, take profit, max hold, or end of data"
        )
        for name, score_array in scores.items():
            result, _arrays = simulate_hold_positions(
                scores=score_array[eval_slice].astype(np.float64),
                closes=closes[eval_slice],
                targets=targets[eval_slice],
                fee=args.fee,
                entry_threshold=args.threshold,
                exit_threshold=args.exit_threshold,
                max_hold_bars=args.max_hold_bars,
                stop_loss=args.stop_loss,
                take_profit=args.take_profit,
                bars_per_year=bars_per_year,
            )
            report["backtest"][name] = result
        _, model_position_arrays = simulate_hold_positions(
            scores=probs,
            closes=closes,
            targets=targets,
            fee=args.fee,
            entry_threshold=args.threshold,
            exit_threshold=args.exit_threshold,
            max_hold_bars=args.max_hold_bars,
            stop_loss=args.stop_loss,
            take_profit=args.take_profit,
            bars_per_year=bars_per_year,
        )

    report_out = Path(args.report_out)
    report_out.parent.mkdir(parents=True, exist_ok=True)
    report_out.write_text(json.dumps(report, indent=2))

    predictions_path = Path(args.predictions_out)
    predictions_path.parent.mkdir(parents=True, exist_ok=True)
    predictions_df = pd.DataFrame(
        {
            "open_time": features_df["open_time"],
            "close": features_df["close"],
            "target": features_df["target"].astype(int),
            "target_up": target_up.astype(int),
            "target_down": target_down.astype(int),
            "forward_return": features_df[forward_return_col],
            "prob_up": probs,
            "prob_down": probs_down,
            "predicted_class_at_0_50": predicted_class_at_050,
            "signal_at_threshold": model_signal,
            "dataset_split": dataset_split_labels(len(features_df), split_idx),
        }
    )

    if args.position_mode == "one_bar":
        predictions_df["entry_signal"] = model_signal
        predictions_df["exit_signal"] = model_signal
        predictions_df["position"] = model_signal
        predictions_df["trade_id"] = np.arange(1, len(predictions_df) + 1) * model_signal
        one_bar_returns = np.where(model_signal == 1, forward_returns - (2.0 * args.fee), np.nan)
        predictions_df["realized_trade_return"] = one_bar_returns
        predictions_df["entry_trade_net_return"] = one_bar_returns
        predictions_df["bars_held_at_exit"] = np.where(model_signal == 1, 1, np.nan)
        predictions_df["exit_reason"] = np.where(model_signal == 1, "one_bar", "")
    elif model_position_arrays is not None:
        for key, values in model_position_arrays.items():
            predictions_df[key] = values

    predictions_df.to_parquet(predictions_path, index=False)

    print(f"Saved XGBoost backtest report to {report_out}")
    print(f"Saved predictions to {predictions_path}")
    print("XGBoost backtest summary:")
    print(json.dumps(report["backtest"]["xgboost"], indent=2))


if __name__ == "__main__":
    main()
