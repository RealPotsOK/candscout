#!/usr/bin/env python3
"""Simulate account-level one-bar trades from model predictions over a UTC window."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Simulate bank account trades from predictions.")
    parser.add_argument("--predictions", default="data/reports/predictions_5m.parquet", help="Predictions Parquet path")
    parser.add_argument(
        "--start",
        default=None,
        help="UTC simulation start date/time. Default: start of final --default-test-fraction of predictions.",
    )
    parser.add_argument(
        "--date",
        default=None,
        help="Backward-compatible alias for --start when simulating from a UTC date",
    )
    parser.add_argument(
        "--duration",
        default=None,
        help="Simulation duration, for example 1D, 12H, 6H, or 90min. Default: until predictions end.",
    )
    parser.add_argument(
        "--default-test-fraction",
        type=float,
        default=0.2,
        help="When --start is omitted, simulate the final fraction of predictions (default: 0.2)",
    )
    parser.add_argument("--starting-cash", type=float, default=10_000.0, help="Starting cash balance")
    parser.add_argument("--min-invest", type=float, default=100.0, help="Minimum investment per trade")
    parser.add_argument("--max-invest", type=float, default=500.0, help="Maximum investment per trade")
    parser.add_argument("--threshold", type=float, default=0.55, help="Probability threshold for entering a trade")
    parser.add_argument("--fee", type=float, default=0.0, help="Per-side fee rate")
    parser.add_argument("--report-out", default="models/daily_bank_report_5m.json", help="JSON report output path")
    parser.add_argument("--trades-out", default="data/reports/daily_bank_trades_5m.csv", help="Trade CSV output path")
    return parser.parse_args()


def load_predictions(path: Path) -> pd.DataFrame:
    df = pd.read_parquet(path)
    required = {"open_time", "close", "prob_up", "forward_return"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Predictions file missing required columns: {sorted(missing)}")

    df = df.sort_values("open_time").drop_duplicates("open_time").reset_index(drop=True)
    df["open_time"] = pd.to_datetime(df["open_time"], utc=True)
    return df


def parse_utc_timestamp(raw: str) -> pd.Timestamp:
    if raw.endswith("Z"):
        raw = raw[:-1] + "+00:00"
    ts = pd.Timestamp(raw)
    if ts.tzinfo is None:
        return ts.tz_localize("UTC")
    return ts.tz_convert("UTC")


def filter_window(df: pd.DataFrame, start_raw: str, duration_raw: str) -> tuple[pd.DataFrame, pd.Timestamp, pd.Timestamp]:
    start = parse_utc_timestamp(start_raw)
    duration = pd.Timedelta(duration_raw.strip().lower())
    if duration <= pd.Timedelta(0):
        raise ValueError("--duration must be greater than zero")
    end = start + duration
    out = df[(df["open_time"] >= start) & (df["open_time"] < end)].copy().reset_index(drop=True)
    if out.empty:
        available_start = df["open_time"].min()
        available_end = df["open_time"].max()
        raise ValueError(
            "No prediction rows found for requested simulation window.\n"
            f"Requested: {start} to {end}\n"
            f"Available predictions: {available_start} to {available_end}\n"
            "Use SIM_START/SIM_DURATION inside that available range, or regenerate predictions "
            "with make experiment START=... END=... first."
        )
    return out, start, end


def default_test_window(df: pd.DataFrame, test_fraction: float) -> tuple[pd.DataFrame, pd.Timestamp, pd.Timestamp]:
    if not (0.0 < test_fraction < 1.0):
        raise ValueError("--default-test-fraction must be between 0 and 1")
    split_idx = int(len(df) * (1.0 - test_fraction))
    split_idx = max(0, min(len(df) - 1, split_idx))
    out = df.iloc[split_idx:].copy().reset_index(drop=True)
    start = pd.Timestamp(out["open_time"].iloc[0])
    end = pd.Timestamp(out["open_time"].iloc[-1]) + infer_bar_delta(out)
    return out, start, end


def infer_bar_delta(df: pd.DataFrame) -> pd.Timedelta:
    diffs = df["open_time"].diff().dropna()
    if diffs.empty:
        return pd.Timedelta(minutes=5)
    return pd.Timedelta(diffs.median())


def investment_size(prob_up: float, threshold: float, min_invest: float, max_invest: float) -> float:
    confidence = (prob_up - threshold) / (1.0 - threshold)
    confidence = min(max(confidence, 0.0), 1.0)
    return min_invest + (max_invest - min_invest) * math.sqrt(confidence)


def empty_best_worst() -> dict:
    return {
        "entry_time": None,
        "exit_time": None,
        "prob_up": 0.0,
        "investment": 0.0,
        "gross_return": 0.0,
        "net_profit": 0.0,
    }


def summarize_trade(row: pd.Series) -> dict:
    return {
        "entry_time": str(row["entry_time"]),
        "exit_time": str(row["exit_time"]),
        "prob_up": float(row["prob_up"]),
        "investment": float(row["investment"]),
        "gross_return": float(row["gross_return"]),
        "net_profit": float(row["net_profit"]),
    }


def simulate_day(
    df: pd.DataFrame,
    start: pd.Timestamp,
    end: pd.Timestamp,
    starting_cash: float,
    min_invest: float,
    max_invest: float,
    threshold: float,
    fee: float,
) -> tuple[dict, pd.DataFrame]:
    if starting_cash <= 0.0:
        raise ValueError("--starting-cash must be positive")
    if min_invest <= 0.0:
        raise ValueError("--min-invest must be positive")
    if max_invest < min_invest:
        raise ValueError("--max-invest must be greater than or equal to --min-invest")
    if not (0.0 <= fee < 1.0):
        raise ValueError("--fee must be >= 0 and < 1")
    if not (0.0 < threshold < 1.0):
        raise ValueError("--threshold must be between 0 and 1")

    cash = float(starting_cash)
    max_net_value = cash
    min_net_value = cash
    total_invested = 0.0
    total_fees = 0.0
    skipped_signals = 0
    rows: list[dict] = []

    for i, row in df.iterrows():
        prob_up = float(row["prob_up"])
        if prob_up < threshold:
            continue

        required_cash = min_invest * (1.0 + fee)
        if cash < required_cash:
            skipped_signals += 1
            continue

        planned_investment = investment_size(prob_up, threshold, min_invest, max_invest)
        investment = min(planned_investment, cash / (1.0 + fee))
        if investment < min_invest:
            skipped_signals += 1
            continue

        entry_fee = investment * fee
        gross_return = float(row["forward_return"])
        gross_exit_value = investment * (1.0 + gross_return)
        exit_fee = gross_exit_value * fee
        net_profit = gross_exit_value - exit_fee - investment - entry_fee
        cash = cash + net_profit

        total_invested += investment
        total_fees += entry_fee + exit_fee
        max_net_value = max(max_net_value, cash)
        min_net_value = min(min_net_value, cash)

        entry_time = pd.Timestamp(row["open_time"])
        exit_time = df["open_time"].iloc[i + 1] if i + 1 < len(df) else entry_time + pd.Timedelta(minutes=5)
        entry_price = float(row["close"])
        exit_price = entry_price * (1.0 + gross_return)

        rows.append(
            {
                "entry_time": entry_time.isoformat(),
                "exit_time": pd.Timestamp(exit_time).isoformat(),
                "entry_price": entry_price,
                "exit_price": exit_price,
                "prob_up": prob_up,
                "investment": investment,
                "gross_return": gross_return,
                "net_profit": net_profit,
                "cash_after_trade": cash,
                "account_value_after_trade": cash,
            }
        )

    trades_df = pd.DataFrame(
        rows,
        columns=[
            "entry_time",
            "exit_time",
            "entry_price",
            "exit_price",
            "prob_up",
            "investment",
            "gross_return",
            "net_profit",
            "cash_after_trade",
            "account_value_after_trade",
        ],
    )

    trade_count = len(trades_df)
    if trade_count:
        winning_trades = int((trades_df["net_profit"] > 0.0).sum())
        losing_trades = int((trades_df["net_profit"] <= 0.0).sum())
        profitable_trade_rate = float(winning_trades / trade_count)
        best_invest = summarize_trade(trades_df.loc[trades_df["net_profit"].idxmax()])
        worst_invest = summarize_trade(trades_df.loc[trades_df["net_profit"].idxmin()])
    else:
        winning_trades = 0
        losing_trades = 0
        profitable_trade_rate = 0.0
        best_invest = empty_best_worst()
        worst_invest = empty_best_worst()

    ending_cash = float(cash)
    total_profit = ending_cash - starting_cash
    btc_start_price = float(df["close"].iloc[0])
    btc_end_price = float(df["close"].iloc[-1])
    btc_return_pct = (btc_end_price / btc_start_price - 1.0) * 100.0
    entry_count = int(trade_count)
    exit_count = int(trade_count)

    report = {
        "start_utc": str(start),
        "end_utc": str(end),
        "duration": str(end - start),
        "total_days": float((end - start) / pd.Timedelta(days=1)),
        "btc_start_price": btc_start_price,
        "btc_end_price": btc_end_price,
        "btc_return_pct": float(btc_return_pct),
        "predictions": {
            "rows_for_window": int(len(df)),
            "first_open_time": str(df["open_time"].iloc[0]),
            "last_open_time": str(df["open_time"].iloc[-1]),
        },
        "assumptions": {
            "logic": "independent one-bar trades from prediction row t to t+1",
            "starting_cash": float(starting_cash),
            "min_invest": float(min_invest),
            "max_invest": float(max_invest),
            "threshold": float(threshold),
            "fee_per_side": float(fee),
            "position_overlap": "none",
            "slippage": 0.0,
        },
        "starting_cash": float(starting_cash),
        "ending_cash": ending_cash,
        "total_profit": float(total_profit),
        "total_return_pct": float(total_profit / starting_cash * 100.0),
        "max_net_value": float(max_net_value),
        "min_net_value": float(min_net_value),
        "trade_count": int(trade_count),
        "entry_count": entry_count,
        "exit_count": exit_count,
        "winning_trades": int(winning_trades),
        "losing_trades": int(losing_trades),
        "profitable_trade_rate": float(profitable_trade_rate),
        "total_invested": float(total_invested),
        "total_fees": float(total_fees),
        "skipped_signals": int(skipped_signals),
        "best_invest": best_invest,
        "worst_invest": worst_invest,
    }

    return report, trades_df


def main() -> None:
    args = parse_args()
    start_raw = args.start or args.date

    predictions = load_predictions(Path(args.predictions))
    if start_raw:
        duration_raw = args.duration or "1D"
        window_predictions, start, end = filter_window(predictions, start_raw, duration_raw)
    else:
        window_predictions, start, end = default_test_window(predictions, args.default_test_fraction)
    report, trades_df = simulate_day(
        df=window_predictions,
        start=start,
        end=end,
        starting_cash=args.starting_cash,
        min_invest=args.min_invest,
        max_invest=args.max_invest,
        threshold=args.threshold,
        fee=args.fee,
    )

    report_path = Path(args.report_out)
    trades_path = Path(args.trades_out)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    trades_path.parent.mkdir(parents=True, exist_ok=True)

    report_path.write_text(json.dumps(report, indent=2))
    trades_df.to_csv(trades_path, index=False)

    print(f"Saved daily bank report to {report_path}")
    print(f"Saved daily trade log to {trades_path}")
    print(json.dumps({k: report[k] for k in [
        "start_utc",
        "end_utc",
        "total_days",
        "btc_start_price",
        "btc_end_price",
        "starting_cash",
        "ending_cash",
        "total_profit",
        "total_return_pct",
        "max_net_value",
        "min_net_value",
        "trade_count",
        "entry_count",
        "exit_count",
        "profitable_trade_rate",
    ]}, indent=2))


if __name__ == "__main__":
    main()
