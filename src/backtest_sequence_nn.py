#!/usr/bin/env python3
"""Backtest sequence neural net models and write prediction rows."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from backtest import (
    bars_per_year_from_seconds,
    infer_bar_seconds,
    simulate_hold_positions,
    simulate_independent_one_bar_trades,
)
from sequence_data import build_sequence_dataset, load_raw_candles
from sequence_nn import load_sequence_model, predict_loaded_sequence_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Backtest a sequence neural net model.")
    parser.add_argument("--raw-data", required=True, help="Raw candle Parquet path")
    parser.add_argument("--model", required=True, help="Trained sequence model .npz path")
    parser.add_argument("--fee", type=float, default=0.001, help="Per-side fee rate")
    parser.add_argument("--threshold", type=float, default=0.55, help="Entry probability threshold")
    parser.add_argument(
        "--position-mode",
        choices=["one_bar", "hold"],
        default="hold",
        help="Backtest logic: independent one-bar trades or hold/exit position logic",
    )
    parser.add_argument("--exit-threshold", type=float, default=0.45, help="Exit when score drops below this value")
    parser.add_argument("--max-hold-bars", type=int, default=60, help="Max bars to hold; 0 disables")
    parser.add_argument("--stop-loss", type=float, default=0.002, help="Gross stop loss from entry; 0 disables")
    parser.add_argument("--take-profit", type=float, default=0.004, help="Gross take profit from entry; 0 disables")
    parser.add_argument("--report-out", default="models/seq_nn_backtest_report_5m.json", help="Backtest report JSON")
    parser.add_argument(
        "--predictions-out",
        default="data/reports/seq_nn_predictions_5m.parquet",
        help="Per-row prediction Parquet output",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model = load_sequence_model(Path(args.model))
    model_key = model["model_type"]
    if model_key not in {"sequence_mlp", "sequence_cnn"}:
        raise ValueError(f"Unsupported sequence model type: {model_key}")

    candles = load_raw_candles(Path(args.raw_data))
    x_seq, _y, meta, channel_names = build_sequence_dataset(
        candles,
        lookback=model["lookback"],
        edge=model["edge"],
    )

    if channel_names != model["channel_names"]:
        raise ValueError(f"Sequence channel mismatch. Data={channel_names}, model={model['channel_names']}")

    probs = predict_loaded_sequence_model(model, x_seq)

    bar_seconds = infer_bar_seconds(meta["open_time"])
    bars_per_year = bars_per_year_from_seconds(bar_seconds)
    model_signal = (probs >= args.threshold).astype(int)
    predicted_class_at_050 = (probs >= 0.50).astype(int)
    forward_returns = meta["forward_return"].to_numpy(dtype=np.float64)
    targets = meta["target"].to_numpy(dtype=np.int64)
    closes = meta["close"].to_numpy(dtype=np.float64)

    scores = {
        model_key: probs,
        "always_positive": np.ones(len(meta), dtype=int),
        "always_negative": np.zeros(len(meta), dtype=int),
        "prev_candle_direction": (meta["return_1bar"].to_numpy(dtype=np.float64) > 0.0).astype(int),
        "ma_direction": (meta["sma_spread"].to_numpy(dtype=np.float64) > 0.0).astype(int),
    }
    signals = {name: (score_array >= args.threshold).astype(int) for name, score_array in scores.items()}

    architecture: dict[str, object] = {
        "model_type": model_key,
        "lookback": int(model["lookback"]),
        "sequence_channels": model["channel_names"],
        "edge": float(model["edge"]),
    }
    if model_key == "sequence_cnn":
        architecture.update(
            {
                "cnn_filters": model["cnn_filters"],
                "cnn_kernel_sizes": model["cnn_kernel_sizes"],
                "hidden_layers": model["hidden_layers"],
            }
        )
    else:
        architecture.update({"hidden_layers": model["hidden_layers"]})

    report = {
        "assumptions": {
            **architecture,
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
        },
        "rows": int(len(meta)),
        "backtest": {},
    }

    model_position_arrays: dict[str, np.ndarray] | None = None

    if args.position_mode == "one_bar":
        report["assumptions"]["position_logic"] = (
            "independent one-bar trades: if signal=1 at t, enter close[t], "
            "exit close[t+1], subtract round-trip fees"
        )
        for name, signal_array in signals.items():
            report["backtest"][name] = simulate_independent_one_bar_trades(
                signals=signal_array,
                targets=targets,
                forward_returns=forward_returns,
                fee=args.fee,
                bars_per_year=bars_per_year,
            )
    else:
        report["assumptions"]["position_logic"] = (
            "hold one long position after entry until exit threshold, stop loss, "
            "take profit, max hold, or end of data"
        )
        for name, score_array in scores.items():
            result, arrays = simulate_hold_positions(
                scores=score_array.astype(np.float64),
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
            report["backtest"][name] = result
            if name == model_key:
                model_position_arrays = arrays

    out_path = Path(args.report_out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2))

    predictions_df = pd.DataFrame(
        {
            "open_time": meta["open_time"],
            "close": meta["close"],
            "target": meta["target"].astype(int),
            "forward_return": meta["forward_return"],
            "prob_up": probs,
            "predicted_class_at_0_50": predicted_class_at_050,
            "signal_at_threshold": model_signal,
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

    predictions_path = Path(args.predictions_out)
    predictions_path.parent.mkdir(parents=True, exist_ok=True)
    predictions_df.to_parquet(predictions_path, index=False)

    print(f"Saved sequence backtest report to {out_path}")
    print(f"Saved sequence predictions to {predictions_path}")
    print(f"{model_key} backtest summary:")
    print(json.dumps(report["backtest"][model_key], indent=2))


if __name__ == "__main__":
    main()
