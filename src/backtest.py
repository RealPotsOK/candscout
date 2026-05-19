#!/usr/bin/env python3
"""Backtest model and baseline signals with one-bar or hold/exit position logic."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Backtest long-or-cash model signals.")
    parser.add_argument("--features", required=True, help="Feature Parquet path")
    parser.add_argument("--model", required=True, help="Trained model .npz path")
    parser.add_argument("--fee", type=float, default=0.001, help="Per-side fee rate (default: 0.001 = 0.10%%)")
    parser.add_argument("--threshold", type=float, default=0.55, help="Probability threshold for long signal")
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
    parser.add_argument("--report-out", default="models/backtest_report_5m.json", help="Backtest report JSON output")
    parser.add_argument(
        "--predictions-out",
        default="data/reports/predictions_5m.parquet",
        help="Per-row predictions Parquet output path",
    )
    return parser.parse_args()


def sigmoid(z: np.ndarray) -> np.ndarray:
    z = np.clip(z, -500.0, 500.0)
    return 1.0 / (1.0 + np.exp(-z))


def max_drawdown(equity_curve: np.ndarray) -> float:
    running_max = np.maximum.accumulate(equity_curve)
    drawdowns = equity_curve / running_max - 1.0
    return float(np.min(drawdowns))


def resolve_column(df: pd.DataFrame, candidates: list[str], purpose: str) -> str:
    for col in candidates:
        if col in df.columns:
            return col
    raise ValueError(f"Missing required column for {purpose}. Tried: {candidates}")


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


def annualized_return(total_return: float, bars: int, bars_per_year: int) -> float:
    if bars <= 0:
        return 0.0
    if total_return <= -1.0:
        return -1.0
    return float((1.0 + total_return) ** (bars_per_year / bars) - 1.0)


def simulate_independent_one_bar_trades(
    signals: np.ndarray,
    targets: np.ndarray,
    forward_returns: np.ndarray,
    fee: float,
    bars_per_year: int,
) -> dict:
    signals = signals.astype(int)
    targets = targets.astype(int)
    forward_returns = forward_returns.astype(np.float64)

    # Independent one-bar trades: when signal=1 at t, enter at close[t],
    # exit at close[t+1], subtract round-trip fees for that trade.
    per_bar_strategy_returns = np.where(signals == 1, forward_returns - (2.0 * fee), 0.0)

    equity_curve = np.cumprod(1.0 + per_bar_strategy_returns)
    ending_equity = float(equity_curve[-1]) if len(equity_curve) else 1.0
    total_return = ending_equity - 1.0

    trade_mask = signals == 1
    trade_count = int(np.sum(trade_mask))
    trade_targets = targets[trade_mask]
    trade_forward_returns = forward_returns[trade_mask]
    trade_returns = per_bar_strategy_returns[trade_mask]
    profitable_trade_rate = float(np.mean(trade_returns > 0.0)) if trade_count else 0.0
    label_hit_rate = float(np.mean(trade_targets == 1)) if trade_count else 0.0

    return {
        "trade_count": trade_count,
        # Legacy field retained for compatibility; equals profitable_trade_rate.
        "hit_rate": profitable_trade_rate,
        "label_hit_rate": label_hit_rate,
        "profitable_trade_rate": profitable_trade_rate,
        "avg_forward_return_per_trade": float(np.mean(trade_forward_returns)) if trade_count else 0.0,
        "avg_net_return_per_trade": float(np.mean(trade_returns)) if trade_count else 0.0,
        "total_return": float(total_return),
        "annualized_return_proxy": annualized_return(total_return, len(per_bar_strategy_returns), bars_per_year),
        "max_drawdown": max_drawdown(equity_curve) if len(equity_curve) else 0.0,
    }


def exit_reason_counts(trades: list[dict]) -> dict:
    counts: dict[str, int] = {}
    for trade in trades:
        reason = str(trade["exit_reason"])
        counts[reason] = counts.get(reason, 0) + 1
    return counts


def trade_report(trades: list[dict], equity_curve: np.ndarray, bars_per_year: int) -> dict:
    if not trades:
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

    net_returns = np.array([trade["net_return"] for trade in trades], dtype=np.float64)
    gross_returns = np.array([trade["gross_return"] for trade in trades], dtype=np.float64)
    entry_targets = np.array([trade["entry_target"] for trade in trades], dtype=np.int64)
    bars_held = np.array([trade["bars_held"] for trade in trades], dtype=np.float64)
    total_return = float(np.prod(1.0 + net_returns) - 1.0)
    profitable_trade_rate = float(np.mean(net_returns > 0.0))

    return {
        "trade_count": int(len(trades)),
        # Legacy field retained for compatibility; equals profitable_trade_rate.
        "hit_rate": profitable_trade_rate,
        "label_hit_rate": float(np.mean(entry_targets == 1)),
        "profitable_trade_rate": profitable_trade_rate,
        "avg_gross_return_per_trade": float(np.mean(gross_returns)),
        "avg_net_return_per_trade": float(np.mean(net_returns)),
        "best_net_return": float(np.max(net_returns)),
        "worst_net_return": float(np.min(net_returns)),
        "avg_bars_held": float(np.mean(bars_held)),
        "total_return": total_return,
        "annualized_return_proxy": annualized_return(total_return, len(equity_curve), bars_per_year),
        "max_drawdown": max_drawdown(equity_curve) if len(equity_curve) else 0.0,
        "exit_reason_counts": exit_reason_counts(trades),
    }


def simulate_hold_positions(
    scores: np.ndarray,
    closes: np.ndarray,
    targets: np.ndarray,
    fee: float,
    entry_threshold: float,
    exit_threshold: float,
    max_hold_bars: int,
    stop_loss: float,
    take_profit: float,
    bars_per_year: int,
) -> tuple[dict, dict[str, np.ndarray]]:
    scores = scores.astype(np.float64)
    closes = closes.astype(np.float64)
    targets = targets.astype(np.int64)
    n_rows = len(scores)

    position = np.zeros(n_rows, dtype=np.int64)
    entry_signal = np.zeros(n_rows, dtype=np.int64)
    exit_signal = np.zeros(n_rows, dtype=np.int64)
    trade_id = np.zeros(n_rows, dtype=np.int64)
    realized_trade_return = np.full(n_rows, np.nan, dtype=np.float64)
    entry_trade_net_return = np.full(n_rows, np.nan, dtype=np.float64)
    bars_held_at_exit = np.full(n_rows, np.nan, dtype=np.float64)
    exit_reason = np.array([""] * n_rows, dtype=object)
    equity_curve = np.ones(n_rows, dtype=np.float64)

    in_position = False
    active_trade_id = 0
    entry_idx = -1
    entry_price = 0.0
    current_equity = 1.0
    trades: list[dict] = []

    for i in range(n_rows):
        if in_position:
            position[i] = 1
            trade_id[i] = active_trade_id

            if i > entry_idx:
                gross_return = closes[i] / entry_price - 1.0
                bars_held = i - entry_idx
                reason = ""

                if stop_loss > 0.0 and gross_return <= -stop_loss:
                    reason = "stop_loss"
                elif take_profit > 0.0 and gross_return >= take_profit:
                    reason = "take_profit"
                elif scores[i] < exit_threshold:
                    reason = "exit_threshold"
                elif max_hold_bars > 0 and bars_held >= max_hold_bars:
                    reason = "max_hold"
                elif i == n_rows - 1:
                    reason = "end_of_data"

                if reason:
                    net_return = gross_return - (2.0 * fee)
                    exit_signal[i] = 1
                    realized_trade_return[i] = net_return
                    entry_trade_net_return[entry_idx] = net_return
                    bars_held_at_exit[i] = bars_held
                    exit_reason[i] = reason
                    current_equity *= 1.0 + net_return
                    trades.append(
                        {
                            "trade_id": active_trade_id,
                            "entry_idx": entry_idx,
                            "exit_idx": i,
                            "entry_score": float(scores[entry_idx]),
                            "exit_score": float(scores[i]),
                            "entry_target": int(targets[entry_idx]),
                            "entry_price": float(entry_price),
                            "exit_price": float(closes[i]),
                            "gross_return": float(gross_return),
                            "net_return": float(net_return),
                            "bars_held": int(bars_held),
                            "exit_reason": reason,
                        }
                    )
                    in_position = False

        equity_curve[i] = current_equity

        if not in_position and i < n_rows - 1 and scores[i] >= entry_threshold:
            active_trade_id += 1
            in_position = True
            entry_idx = i
            entry_price = closes[i]
            entry_signal[i] = 1
            position[i] = 1
            trade_id[i] = active_trade_id

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
    return trade_report(trades, equity_curve, bars_per_year), arrays


def main() -> None:
    args = parse_args()

    features_df = pd.read_parquet(args.features).sort_values("open_time").reset_index(drop=True)
    required_feature_cols = {"open_time", "close", "target", "sma_spread"}
    missing_base = required_feature_cols - set(features_df.columns)
    if missing_base:
        raise ValueError(f"Feature file missing required columns: {sorted(missing_base)}")
    forward_return_col = resolve_column(
        features_df,
        ["forward_return", "forward_return_1m"],
        "next-candle forward return",
    )
    one_bar_return_col = resolve_column(
        features_df,
        ["return_1bar", "return_1m"],
        "previous-candle direction baseline",
    )
    bar_seconds = infer_bar_seconds(features_df["open_time"])
    bars_per_year = bars_per_year_from_seconds(bar_seconds)

    model = np.load(args.model)
    weights = model["weights"]
    bias = float(model["bias"][0])
    feature_names = [str(x) for x in model["feature_names"]]
    mean = model["mean"]
    std = model["std"]

    missing_model_cols = set(feature_names) - set(features_df.columns)
    if missing_model_cols:
        raise ValueError(f"Feature file missing model columns: {sorted(missing_model_cols)}")

    x = features_df[feature_names].to_numpy(dtype=np.float64)
    x_norm = (x - mean) / std
    probs = sigmoid(x_norm @ weights + bias)

    model_signal = (probs >= args.threshold).astype(int)
    predicted_class_at_050 = (probs >= 0.50).astype(int)
    forward_returns = features_df[forward_return_col].to_numpy(dtype=np.float64)
    targets = features_df["target"].to_numpy(dtype=np.int64)
    closes = features_df["close"].to_numpy(dtype=np.float64)

    scores = {
        "logistic_regression": probs,
        "always_positive": np.ones(len(features_df), dtype=int),
        "always_negative": np.zeros(len(features_df), dtype=int),
        "prev_candle_direction": (features_df[one_bar_return_col].to_numpy() > 0.0).astype(int),
        "ma_direction": (features_df["sma_spread"].to_numpy() > 0.0).astype(int),
    }
    signals = {name: (score_array >= args.threshold).astype(int) for name, score_array in scores.items()}

    report = {
        "assumptions": {
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
        },
        "rows": int(len(features_df)),
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
            if name == "logistic_regression":
                model_position_arrays = arrays

    out_path = Path(args.report_out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2))

    predictions_path = Path(args.predictions_out)
    predictions_path.parent.mkdir(parents=True, exist_ok=True)
    predictions_df = pd.DataFrame(
        {
            "open_time": features_df["open_time"],
            "close": features_df["close"],
            "target": features_df["target"].astype(int),
            "forward_return": features_df[forward_return_col],
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

    predictions_df.to_parquet(predictions_path, index=False)

    print(f"Saved backtest report to {out_path}")
    print(f"Saved predictions to {predictions_path}")
    print("Logistic regression backtest summary:")
    print(json.dumps(report["backtest"]["logistic_regression"], indent=2))


if __name__ == "__main__":
    main()
