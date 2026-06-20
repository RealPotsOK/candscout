#!/usr/bin/env python3
"""Simulate account-level hold trades from strategy/model probabilities."""

from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path

import pandas as pd

_NUMERIC_RE = re.compile(r"^(?:\d+(?:\.\d*)?|\.\d+)$")
_COEFF_M_RE = re.compile(r"^((?:\d+(?:\.\d*)?|\.\d+))\s*\*?\s*m$")
_M_DIV_RE = re.compile(r"^m\s*/\s*((?:\d+(?:\.\d*)?|\.\d+))$")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Simulate one-position account trading from prediction probabilities.")
    parser.add_argument("--predictions", required=True, help="Predictions Parquet path")
    parser.add_argument("--model-type", default="", help="Optional strategy type, e.g. buy_hold")
    parser.add_argument("--start", default=None, help="UTC simulation start. Default: final test fraction")
    parser.add_argument("--duration", default=None, help="Simulation duration, e.g. 10D, 12H")
    parser.add_argument("--default-test-fraction", type=float, default=0.05)
    parser.add_argument("--starting-cash", type=float, default=10_000.0)
    parser.add_argument("--min-invest", type=float, default=1.0)
    parser.add_argument("--max-invest", default="m")
    parser.add_argument(
        "--confidence-multiplier",
        type=float,
        default=1.0,
        help="Multiplies confidence before sizing; >1 sizes near-threshold buys more aggressively",
    )
    parser.add_argument("--threshold", type=float, default=0.52)
    parser.add_argument("--exit-threshold", type=float, default=0.50)
    parser.add_argument("--fee", type=float, default=0.0)
    parser.add_argument("--max-hold-bars", type=int, default=0, help="0 disables")
    parser.add_argument("--stop-loss", type=float, default=0.0, help="0 disables")
    parser.add_argument("--take-profit", type=float, default=0.0, help="0 disables")
    parser.add_argument("--report-out", required=True, help="JSON report output")
    parser.add_argument("--trades-out", required=True, help="Trade CSV output")
    return parser.parse_args()


def load_predictions(path: Path) -> pd.DataFrame:
    df = pd.read_parquet(path)
    required = {"open_time", "close", "prob_up"}
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


def infer_bar_delta(df: pd.DataFrame) -> pd.Timedelta:
    diffs = df["open_time"].diff().dropna()
    if diffs.empty:
        return pd.Timedelta(minutes=5)
    return pd.Timedelta(diffs.median())


def filter_window(df: pd.DataFrame, start_raw: str, duration_raw: str) -> tuple[pd.DataFrame, pd.Timestamp, pd.Timestamp]:
    start = parse_utc_timestamp(start_raw)
    duration = pd.Timedelta(duration_raw.strip().lower())
    if duration <= pd.Timedelta(0):
        raise ValueError("--duration must be greater than zero")
    end = start + duration
    out = df[(df["open_time"] >= start) & (df["open_time"] < end)].copy().reset_index(drop=True)
    if out.empty:
        raise ValueError(f"No prediction rows found from {start} to {end}")
    return out, start, end


def default_test_window(df: pd.DataFrame, test_fraction: float) -> tuple[pd.DataFrame, pd.Timestamp, pd.Timestamp]:
    if "dataset_split" in df.columns:
        split_values = df["dataset_split"].astype(str).str.lower()
        out = df[split_values == "test"].copy().reset_index(drop=True)
        if out.empty:
            raise ValueError("Predictions contain dataset_split but no test rows")
        start = pd.Timestamp(out["open_time"].iloc[0])
        end = pd.Timestamp(out["open_time"].iloc[-1]) + infer_bar_delta(out)
        return out, start, end

    if not (0.0 < test_fraction < 1.0):
        raise ValueError("--default-test-fraction must be between 0 and 1")
    split_idx = int(len(df) * (1.0 - test_fraction))
    split_idx = max(0, min(len(df) - 1, split_idx))
    out = df.iloc[split_idx:].copy().reset_index(drop=True)
    start = pd.Timestamp(out["open_time"].iloc[0])
    end = pd.Timestamp(out["open_time"].iloc[-1]) + infer_bar_delta(out)
    return out, start, end


def investment_size(
    prob_up: float,
    threshold: float,
    min_invest: float,
    max_invest: float,
    confidence_multiplier: float,
) -> float:
    confidence = (prob_up - threshold) / (1.0 - threshold)
    confidence *= confidence_multiplier
    confidence = min(max(confidence, 0.0), 1.0)
    return min_invest + (max_invest - min_invest) * math.sqrt(confidence)


def parse_max_invest(expr: str, available_cash: float) -> float:
    raw = str(expr).strip().lower().replace(" ", "")
    if raw == "m":
        return available_cash
    if _NUMERIC_RE.match(raw):
        return float(raw)
    coeff_match = _COEFF_M_RE.match(raw)
    if coeff_match:
        return float(coeff_match.group(1)) * available_cash
    div_match = _M_DIV_RE.match(raw)
    if div_match:
        divisor = float(div_match.group(1))
        if divisor <= 0.0:
            raise ValueError("--max-invest divisor must be positive")
        return available_cash / divisor
    raise ValueError(f"Unsupported --max-invest expression: {expr!r}")


def empty_trade_columns() -> list[str]:
    return [
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
    ]


def summarize_trade(row: pd.Series) -> dict:
    return {
        "entry_time": str(row["entry_time"]),
        "exit_time": str(row["exit_time"]),
        "prob_up": float(row["prob_up"]),
        "investment": float(row["investment"]),
        "gross_return": float(row["gross_return"]),
        "net_profit": float(row["net_profit"]),
    }


def simulate(df: pd.DataFrame, args: argparse.Namespace, start: pd.Timestamp, end: pd.Timestamp) -> tuple[dict, pd.DataFrame]:
    if args.starting_cash <= 0.0:
        raise ValueError("--starting-cash must be positive")
    if args.min_invest <= 0.0:
        raise ValueError("--min-invest must be positive")
    if args.confidence_multiplier <= 0.0:
        raise ValueError("--confidence-multiplier must be positive")
    if not (0.0 <= args.fee < 1.0):
        raise ValueError("--fee must be >= 0 and < 1")

    cash = float(args.starting_cash)
    coin = 0.0
    entry_price = 0.0
    entry_time: pd.Timestamp | None = None
    entry_prob = 0.0
    investment = 0.0
    entry_fee = 0.0
    bars_held = 0
    completed_buy_hold = False
    skipped_signals = 0
    rows: list[dict] = []
    max_net_value = cash
    min_net_value = cash
    total_invested = 0.0
    total_fees = 0.0

    for idx, row in df.iterrows():
        close = float(row["close"])
        prob_up = float(row["prob_up"])
        last_row = idx == len(df) - 1

        if coin > 0.0:
            bars_held += 1
            gross_return = close / entry_price - 1.0 if entry_price > 0.0 else 0.0
            account_value = cash + coin * close
            max_net_value = max(max_net_value, account_value)
            min_net_value = min(min_net_value, account_value)
            should_exit = False
            if args.model_type == "buy_hold":
                should_exit = last_row
            elif last_row:
                should_exit = True
            elif prob_up < args.exit_threshold:
                should_exit = True
            elif args.stop_loss > 0.0 and gross_return <= -args.stop_loss:
                should_exit = True
            elif args.take_profit > 0.0 and gross_return >= args.take_profit:
                should_exit = True
            elif args.max_hold_bars > 0 and bars_held >= args.max_hold_bars:
                should_exit = True

            if should_exit:
                gross_exit_value = coin * close
                exit_fee = gross_exit_value * args.fee
                exit_proceeds = gross_exit_value - exit_fee
                cash += exit_proceeds
                net_profit = exit_proceeds - investment - entry_fee
                total_fees += exit_fee
                max_net_value = max(max_net_value, cash)
                min_net_value = min(min_net_value, cash)
                rows.append(
                    {
                        "entry_time": entry_time.isoformat() if entry_time is not None else str(row["open_time"]),
                        "exit_time": pd.Timestamp(row["open_time"]).isoformat(),
                        "entry_price": entry_price,
                        "exit_price": close,
                        "prob_up": entry_prob,
                        "investment": investment,
                        "gross_return": gross_return,
                        "net_profit": net_profit,
                        "cash_after_trade": cash,
                        "account_value_after_trade": cash,
                    }
                )
                coin = 0.0
                completed_buy_hold = args.model_type == "buy_hold"
                investment = 0.0
                entry_fee = 0.0
                bars_held = 0
                continue

        if coin <= 0.0 and not completed_buy_hold and not last_row and prob_up >= args.threshold:
            required_cash = args.min_invest * (1.0 + args.fee)
            if cash < required_cash:
                skipped_signals += 1
                continue
            max_cap = min(parse_max_invest(args.max_invest, cash), cash / (1.0 + args.fee))
            if max_cap < args.min_invest:
                skipped_signals += 1
                continue
            planned = investment_size(
                prob_up,
                args.threshold,
                args.min_invest,
                max_cap,
                args.confidence_multiplier,
            )
            investment = min(planned, cash / (1.0 + args.fee))
            if investment < args.min_invest:
                skipped_signals += 1
                continue
            entry_fee = investment * args.fee
            cash -= investment + entry_fee
            coin = investment / close if close > 0.0 else 0.0
            entry_price = close
            entry_time = pd.Timestamp(row["open_time"])
            entry_prob = prob_up
            bars_held = 0
            total_invested += investment
            total_fees += entry_fee
            account_value = cash + coin * close
            max_net_value = max(max_net_value, account_value)
            min_net_value = min(min_net_value, account_value)

    trades = pd.DataFrame(rows, columns=empty_trade_columns())
    ending_cash = cash + coin * float(df["close"].iloc[-1])
    trade_count = int(len(trades))
    winning_trades = int((trades["net_profit"] > 0.0).sum()) if trade_count else 0
    losing_trades = int((trades["net_profit"] <= 0.0).sum()) if trade_count else 0
    best = summarize_trade(trades.loc[trades["net_profit"].idxmax()]) if trade_count else {}
    worst = summarize_trade(trades.loc[trades["net_profit"].idxmin()]) if trade_count else {}
    start_price = float(df["close"].iloc[0])
    end_price = float(df["close"].iloc[-1])
    report = {
        "start_utc": str(start),
        "end_utc": str(end),
        "duration": str(end - start),
        "total_days": float((end - start) / pd.Timedelta(days=1)),
        "asset_start_price": start_price,
        "asset_end_price": end_price,
        "asset_return_pct": float((end_price / start_price - 1.0) * 100.0) if start_price > 0.0 else 0.0,
        "assumptions": {
            "logic": "single-position hold simulation from prediction score thresholds",
            "starting_cash": float(args.starting_cash),
            "min_invest": float(args.min_invest),
            "max_invest": str(args.max_invest),
            "threshold": float(args.threshold),
            "exit_threshold": float(args.exit_threshold),
            "confidence_multiplier": float(args.confidence_multiplier),
            "fee_per_side": float(args.fee),
            "max_hold_bars": int(args.max_hold_bars),
            "stop_loss": float(args.stop_loss),
            "take_profit": float(args.take_profit),
        },
        "starting_cash": float(args.starting_cash),
        "ending_cash": float(ending_cash),
        "total_profit": float(ending_cash - args.starting_cash),
        "total_return_pct": float((ending_cash / args.starting_cash - 1.0) * 100.0),
        "max_net_value": float(max_net_value),
        "min_net_value": float(min_net_value),
        "trade_count": trade_count,
        "entry_count": trade_count,
        "exit_count": trade_count,
        "winning_trades": winning_trades,
        "losing_trades": losing_trades,
        "profitable_trade_rate": float(winning_trades / trade_count) if trade_count else 0.0,
        "total_invested": float(total_invested),
        "total_fees": float(total_fees),
        "skipped_signals": int(skipped_signals),
        "best_invest": best,
        "worst_invest": worst,
    }
    return report, trades


def main() -> None:
    args = parse_args()
    predictions = load_predictions(Path(args.predictions))
    if args.start:
        window, start, end = filter_window(predictions, args.start, args.duration or "1D")
    else:
        window, start, end = default_test_window(predictions, args.default_test_fraction)
    report, trades = simulate(window, args, start, end)

    report_path = Path(args.report_out)
    trades_path = Path(args.trades_out)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    trades_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    trades.to_csv(trades_path, index=False)

    print(f"Saved strategy simulation report to {report_path}")
    print(f"Saved strategy simulation trades to {trades_path}")
    print(json.dumps({k: report[k] for k in [
        "start_utc",
        "end_utc",
        "total_days",
        "asset_start_price",
        "asset_end_price",
        "starting_cash",
        "ending_cash",
        "total_profit",
        "total_return_pct",
        "max_net_value",
        "min_net_value",
        "trade_count",
        "profitable_trade_rate",
    ]}, indent=2))


if __name__ == "__main__":
    main()
