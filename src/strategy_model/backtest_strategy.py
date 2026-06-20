#!/usr/bin/env python3
"""Backtest rule-based strategy models and write compatible prediction files."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

SRC_ROOT = Path(__file__).resolve().parents[1]
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from backtest import (  # noqa: E402
    bars_per_year_from_seconds,
    chronological_split_index,
    dataset_split_labels,
    infer_bar_seconds,
    simulate_hold_positions,
    simulate_independent_one_bar_trades,
)
from strategy_model.strategies import (  # noqa: E402
    buy_hold_arrays,
    load_raw_candles,
    prediction_frame,
    read_model,
    strategy_scores,
    strategy_down_scores,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Backtest a rule-based strategy model.")
    parser.add_argument("--raw-data", required=True, help="Raw candle Parquet path")
    parser.add_argument("--model", required=True, help="Strategy model JSON path")
    parser.add_argument("--fee", type=float, default=0.0001, help="Per-side fee")
    parser.add_argument("--threshold", type=float, default=0.52, help="Entry score threshold")
    parser.add_argument("--split", type=float, default=0.95, help="Chronological train fraction")
    parser.add_argument(
        "--position-mode",
        choices=["one_bar", "hold"],
        default="hold",
        help="Backtest logic for non-buy-hold strategies",
    )
    parser.add_argument("--exit-threshold", type=float, default=0.50, help="Exit score threshold")
    parser.add_argument("--max-hold-bars", type=int, default=60, help="Max bars to hold; 0 disables")
    parser.add_argument("--stop-loss", type=float, default=0.0, help="Stop loss; 0 disables")
    parser.add_argument("--take-profit", type=float, default=0.0, help="Take profit; 0 disables")
    parser.add_argument("--report-out", required=True, help="Backtest report JSON output")
    parser.add_argument("--predictions-out", required=True, help="Predictions Parquet output")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model = read_model(Path(args.model))
    model_type = str(model["model_type"])
    edge = float(model.get("edge", 0.0))
    selected_window = int(model.get("selected_window", model.get("ma_window", 20)))
    counter_band = float(model.get("counter_band", 0.015))

    df = load_raw_candles(Path(args.raw_data), edge=edge)
    bar_seconds = infer_bar_seconds(df["open_time"])
    bars_per_year = bars_per_year_from_seconds(bar_seconds)
    scores = strategy_scores(df, model_type, selected_window, counter_band)
    down_scores = strategy_down_scores(df, model_type, selected_window, counter_band)
    signals = (scores >= args.threshold).astype(int)
    targets = df["target"].to_numpy(dtype=np.int64)
    closes = df["close"].to_numpy(dtype=np.float64)
    forward_returns = df["forward_return"].to_numpy(dtype=np.float64)
    split_idx = chronological_split_index(len(df), args.split)
    eval_slice = slice(split_idx, None)

    report = {
        "assumptions": {
            "model_type": f"strategy_{model_type}",
            "strategy_model_type": model_type,
            "selected_window": selected_window,
            "counter_band": counter_band,
            "fee_per_side": float(args.fee),
            "entry_threshold": float(args.threshold),
            "position_mode": "hold" if model_type == "buy_hold" else args.position_mode,
            "exit_threshold": float(args.exit_threshold),
            "max_hold_bars": int(args.max_hold_bars),
            "stop_loss": float(args.stop_loss),
            "take_profit": float(args.take_profit),
            "bar_seconds": float(bar_seconds),
            "bars_per_year": int(bars_per_year),
        },
        "rows": int(len(df)),
        "evaluation_rows": int(len(df) - split_idx),
        "evaluation_split": "chronological_test",
        "split_fraction": float(args.split),
        "test_start_open_time": str(df["open_time"].iloc[split_idx]),
        "backtest": {},
    }

    if model_type == "buy_hold":
        result, _ = buy_hold_arrays(
            closes=closes[eval_slice],
            targets=targets[eval_slice],
            fee=args.fee,
            bars_per_year=bars_per_year,
        )
        _, arrays = buy_hold_arrays(closes=closes, targets=targets, fee=args.fee, bars_per_year=bars_per_year)
        report["assumptions"]["position_logic"] = "buy once at the first row and sell once at end of data"
        report["backtest"][model_type] = result
    elif args.position_mode == "one_bar":
        result = simulate_independent_one_bar_trades(
            signals=signals[eval_slice],
            targets=targets[eval_slice],
            forward_returns=forward_returns[eval_slice],
            fee=args.fee,
            bars_per_year=bars_per_year,
        )
        report["assumptions"]["position_logic"] = "independent one-bar strategy trades"
        report["backtest"][model_type] = result
        arrays = {
            "position": signals,
            "entry_signal": signals,
            "exit_signal": signals,
            "trade_id": np.arange(1, len(df) + 1) * signals,
            "realized_trade_return": np.where(signals == 1, forward_returns - (2.0 * args.fee), np.nan),
            "entry_trade_net_return": np.where(signals == 1, forward_returns - (2.0 * args.fee), np.nan),
            "bars_held_at_exit": np.where(signals == 1, 1, np.nan),
            "exit_reason": np.where(signals == 1, "one_bar", ""),
        }
    else:
        result, _ = simulate_hold_positions(
            scores=scores[eval_slice],
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
        _, arrays = simulate_hold_positions(
            scores=scores,
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
        report["assumptions"]["position_logic"] = "hold one long position until score exit, risk exit, max hold, or end"
        report["backtest"][model_type] = result

    report_path = Path(args.report_out)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    predictions = prediction_frame(df, scores=scores, down_scores=down_scores, threshold=args.threshold, arrays=arrays)
    predictions["dataset_split"] = dataset_split_labels(len(df), split_idx)
    predictions_path = Path(args.predictions_out)
    predictions_path.parent.mkdir(parents=True, exist_ok=True)
    predictions.to_parquet(predictions_path, index=False)

    print(f"Saved strategy backtest report to {report_path}")
    print(f"Saved strategy predictions to {predictions_path}")
    print(json.dumps(report["backtest"][model_type], indent=2))


if __name__ == "__main__":
    main()
