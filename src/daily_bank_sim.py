#!/usr/bin/env python3
"""Simulate account-level trades from model predictions over a UTC window."""

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
    parser.add_argument(
        "--position-mode",
        choices=["live", "one_bar"],
        default="live",
        help="live = one position with entry/exit rules; one_bar = independent one-bar trades",
    )
    parser.add_argument("--starting-cash", type=float, default=100.0, help="Starting cash balance")
    parser.add_argument("--min-invest", type=float, default=1.0, help="Minimum investment per trade")
    parser.add_argument(
        "--max-invest",
        default="m",
        help="Maximum investment cap. Supports m, 0.5m, m/2, or a fixed number.",
    )
    parser.add_argument(
        "--confidence-multiplier",
        type=float,
        default=1.0,
        help="Multiplies confidence before sizing; >1 sizes near-threshold buys more aggressively",
    )
    parser.add_argument(
        "--short-confidence-multiplier",
        type=float,
        default=3.0,
        help="Short-only sizing multiplier; >1 increases short notional without changing long sizing",
    )
    parser.add_argument(
        "--trade-mode",
        choices=["long_only", "short_only", "long_short"],
        default="long_only",
        help="Allowed direction. Shorts are unleveraged paper simulation.",
    )
    parser.add_argument("--threshold", type=float, default=0.55, help="Long entry probability threshold")
    parser.add_argument("--exit-threshold", type=float, default=0.48, help="Long exit probability threshold")
    parser.add_argument("--short-entry-threshold", type=float, default=0.55, help="Enter short when prob_down is at or above this threshold")
    parser.add_argument("--short-exit-threshold", type=float, default=0.48, help="Cover short when prob_down falls below this threshold")
    parser.add_argument("--max-short-invest", default="m", help="Maximum short notional: m, 0.5m, m/2, or fixed cash")
    parser.add_argument("--allow-flip-position", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--borrow-fee", type=float, default=0.0, help="Short borrow/funding fee per held bar")
    parser.add_argument("--leverage", type=float, default=1.0, help="Only unleveraged 1x paper shorting is supported")
    parser.add_argument("--liquidation-simulation", choices=["off", "basic"], default="off")
    parser.add_argument("--max-hold-bars", type=int, default=60, help="Max bars to hold in live mode; 0 disables")
    parser.add_argument("--stop-loss", type=float, default=0.0, help="Gross stop loss from entry; 0 disables")
    parser.add_argument("--take-profit", type=float, default=0.0, help="Gross take profit from entry; 0 disables")
    parser.add_argument("--fee", type=float, default=0.0, help="Per-side fee rate")
    parser.add_argument("--slippage", type=float, default=0.0, help="Extra execution slippage per side")
    parser.add_argument(
        "--spread-pct",
        type=float,
        default=0.00015,
        help="Synthetic bid/ask spread for historical simulation. Entry ask=close*(1+spread/2), exit bid=close*(1-spread/2).",
    )
    parser.add_argument("--report-out", default="models/daily_bank_report_5m.json", help="JSON report output path")
    parser.add_argument("--trades-out", default="data/reports/daily_bank_trades_5m.csv", help="Trade CSV output path")
    parser.add_argument(
        "--comparison-trade-mode",
        choices=["long_only", "short_only", "long_short"],
        default=None,
        help="Optional second simulation using the same rows and settings.",
    )
    parser.add_argument("--comparison-report-out", default=None, help="Optional comparison JSON report output")
    parser.add_argument("--comparison-trades-out", default=None, help="Optional comparison trade CSV output")
    return parser.parse_args()


def load_predictions(path: Path) -> pd.DataFrame:
    df = pd.read_parquet(path)
    required = {"open_time", "close", "prob_up", "forward_return"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Predictions file missing required columns: {sorted(missing)}")

    df = df.sort_values("open_time").drop_duplicates("open_time").reset_index(drop=True)
    df["open_time"] = pd.to_datetime(df["open_time"], utc=True)
    missing_prob_down = "prob_down" not in df.columns
    if missing_prob_down:
        df["prob_down"] = 0.0
    df.attrs["prob_down_missing"] = missing_prob_down
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
    out.attrs.update(df.attrs)
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
    if "dataset_split" in df.columns:
        split_values = df["dataset_split"].astype(str).str.lower()
        out = df[split_values == "test"].copy().reset_index(drop=True)
        out.attrs.update(df.attrs)
        if out.empty:
            raise ValueError("Predictions contain dataset_split but no test rows")
        out.attrs["window_selection"] = "explicit_dataset_split_test"
        start = pd.Timestamp(out["open_time"].iloc[0])
        end = pd.Timestamp(out["open_time"].iloc[-1]) + infer_bar_delta(out)
        return out, start, end

    if not (0.0 < test_fraction < 1.0):
        raise ValueError("--default-test-fraction must be between 0 and 1")
    split_idx = int(len(df) * (1.0 - test_fraction))
    split_idx = max(0, min(len(df) - 1, split_idx))
    out = df.iloc[split_idx:].copy().reset_index(drop=True)
    out.attrs.update(df.attrs)
    out.attrs["window_selection"] = "fallback_final_fraction"
    start = pd.Timestamp(out["open_time"].iloc[0])
    end = pd.Timestamp(out["open_time"].iloc[-1]) + infer_bar_delta(out)
    return out, start, end


def infer_bar_delta(df: pd.DataFrame) -> pd.Timedelta:
    diffs = df["open_time"].diff().dropna()
    if diffs.empty:
        return pd.Timedelta(minutes=5)
    return pd.Timedelta(diffs.median())


def parse_max_invest(expr: str, available_cash: float) -> float:
    if available_cash < 0:
        raise ValueError("available_cash cannot be negative")
    raw = str(expr).strip().lower().replace(" ", "")
    if raw == "m":
        return available_cash
    if _NUMERIC_RE.match(raw):
        return float(raw)
    coeff_match = _COEFF_M_RE.match(raw)
    if coeff_match:
        coeff = float(coeff_match.group(1))
        if coeff < 0:
            raise ValueError("--max-invest multiplier cannot be negative")
        return coeff * available_cash
    div_match = _M_DIV_RE.match(raw)
    if div_match:
        divisor = float(div_match.group(1))
        if divisor <= 0:
            raise ValueError("--max-invest divisor must be positive")
        return available_cash / divisor
    raise ValueError(f"Unsupported --max-invest expression: {expr!r}")


def investment_size(
    prob_up: float,
    threshold: float,
    min_invest: float,
    max_invest: float,
    confidence_multiplier: float,
) -> float:
    confidence = (prob_up - threshold) / max(1e-12, 1.0 - threshold)
    confidence *= confidence_multiplier
    confidence = min(max(confidence, 0.0), 1.0)
    return min_invest + (max_invest - min_invest) * math.sqrt(confidence)


def short_investment_size(
    prob_down: float,
    threshold: float,
    min_invest: float,
    max_invest: float,
    confidence_multiplier: float,
) -> float:
    confidence = (prob_down - threshold) / max(1e-12, 1.0 - threshold)
    confidence = min(max(confidence * confidence_multiplier, 0.0), 1.0)
    return min_invest + (max_invest - min_invest) * math.sqrt(confidence)


def normalized_long_confidence(prob_up: float, threshold: float) -> float:
    return min(max((prob_up - threshold) / max(1e-12, 1.0 - threshold), 0.0), 1.0)


def normalized_short_confidence(prob_down: float, threshold: float) -> float:
    return min(max((prob_down - threshold) / max(1e-12, 1.0 - threshold), 0.0), 1.0)


def choose_entry_side(
    *,
    prob_up: float,
    prob_down: float,
    trade_mode: str,
    long_threshold: float,
    short_threshold: float,
) -> str:
    long_signal = trade_mode in {"long_only", "long_short"} and prob_up >= long_threshold
    short_signal = trade_mode in {"short_only", "long_short"} and prob_down >= short_threshold
    if long_signal and short_signal:
        long_confidence = normalized_long_confidence(prob_up, long_threshold)
        short_confidence = normalized_short_confidence(prob_down, short_threshold)
        return "long" if long_confidence >= short_confidence else "short"
    if long_signal:
        return "long"
    if short_signal:
        return "short"
    return ""


def synthetic_bid_ask(close: float, spread_pct: float) -> tuple[float, float]:
    spread_pct = max(0.0, float(spread_pct))
    half_spread = spread_pct / 2.0
    bid = close * (1.0 - half_spread)
    ask = close * (1.0 + half_spread)
    return bid, ask


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
        "side": str(row.get("side", "long")),
        "entry_time": str(row["entry_time"]),
        "exit_time": str(row["exit_time"]),
        "prob_up": float(row["prob_up"]),
        "investment": float(row["investment"]),
        "gross_return": float(row["gross_return"]),
        "net_profit": float(row["net_profit"]),
    }


def validate_inputs(
    starting_cash: float,
    min_invest: float,
    threshold: float,
    fee: float,
    confidence_multiplier: float,
    slippage: float,
    spread_pct: float,
) -> None:
    if starting_cash <= 0.0:
        raise ValueError("--starting-cash must be positive")
    if min_invest <= 0.0:
        raise ValueError("--min-invest must be positive")
    if confidence_multiplier <= 0.0:
        raise ValueError("--confidence-multiplier must be positive")
    if not (0.0 <= fee < 1.0):
        raise ValueError("--fee must be >= 0 and < 1")
    if not (0.0 < threshold < 1.0):
        raise ValueError("--threshold must be between 0 and 1")
    if slippage < 0.0:
        raise ValueError("--slippage cannot be negative")
    if spread_pct < 0.0:
        raise ValueError("--spread-pct cannot be negative")


def planned_investment(
    *,
    cash: float,
    max_invest_expr: str,
    min_invest: float,
    fee: float,
    prob_up: float,
    threshold: float,
    confidence_multiplier: float,
) -> float | None:
    max_by_cash = cash / (1.0 + fee)
    max_cap = max(0.0, min(parse_max_invest(max_invest_expr, cash), max_by_cash))
    if max_cap + 1e-12 < min_invest:
        return None
    investment = investment_size(prob_up, threshold, min_invest, max_cap, confidence_multiplier)
    if investment + 1e-12 < min_invest:
        return None
    return investment


def planned_short_investment(
    *,
    cash: float,
    max_invest_expr: str,
    min_invest: float,
    fee: float,
    prob_up: float,
    threshold: float,
    confidence_multiplier: float,
) -> float | None:
    max_by_cash = cash / (1.0 + fee)
    max_cap = max(0.0, min(parse_max_invest(max_invest_expr, cash), max_by_cash))
    if max_cap + 1e-12 < min_invest:
        return None
    investment = short_investment_size(prob_up, threshold, min_invest, max_cap, confidence_multiplier)
    return investment if investment + 1e-12 >= min_invest else None


def exit_reason(
    *,
    prob_up: float,
    prob_down: float,
    exit_threshold: float,
    gross_return: float,
    bars_held: int,
    max_hold_bars: int,
    stop_loss: float,
    take_profit: float,
    last_row: bool,
) -> str | None:
    # Same order as live_sim/app/trading.py: model exit first, then risk exits.
    if prob_up < exit_threshold:
        return "exit_threshold"
    if stop_loss > 0.0 and gross_return <= -stop_loss:
        return "stop_loss"
    if take_profit > 0.0 and gross_return >= take_profit:
        return "take_profit"
    if max_hold_bars > 0 and bars_held >= max_hold_bars:
        return "max_hold_bars"
    if last_row:
        return "end_of_data"
    return None


def short_exit_reason(
    *,
    prob_up: float,
    prob_down: float,
    exit_threshold: float,
    gross_return: float,
    bars_held: int,
    max_hold_bars: int,
    stop_loss: float,
    take_profit: float,
    last_row: bool,
) -> str | None:
    if prob_down < exit_threshold:
        return "short_exit_threshold"
    if stop_loss > 0.0 and gross_return <= -stop_loss:
        return "stop_loss"
    if take_profit > 0.0 and gross_return >= take_profit:
        return "take_profit"
    if max_hold_bars > 0 and bars_held >= max_hold_bars:
        return "max_hold_bars"
    if last_row:
        return "end_of_data"
    return None


def append_trade(
    rows: list[dict],
    *,
    entry_time: pd.Timestamp,
    exit_time: pd.Timestamp,
    entry_price: float,
    exit_price: float,
    entry_bid: float,
    entry_ask: float,
    exit_bid: float,
    exit_ask: float,
    entry_prob_up: float,
    entry_prob_down: float,
    exit_prob_up: float,
    exit_prob_down: float,
    investment: float,
    quantity: float,
    entry_fee: float,
    exit_fee: float,
    gross_exit_value: float,
    net_profit: float,
    cash_after_trade: float,
    account_value_after_trade: float,
    bars_held: int,
    reason: str,
    side: str = "long",
    borrow_fee: float = 0.0,
) -> None:
    gross_return = (
        (entry_price - exit_price) / entry_price
        if side == "short" and entry_price > 0.0
        else gross_exit_value / investment - 1.0 if investment > 0.0 else 0.0
    )
    rows.append(
        {
            "side": side,
            "entry_time": entry_time.isoformat(),
            "exit_time": exit_time.isoformat(),
            "entry_price": entry_price,
            "exit_price": exit_price,
            "entry_bid": entry_bid,
            "entry_ask": entry_ask,
            "exit_bid": exit_bid,
            "exit_ask": exit_ask,
            "prob_up": entry_prob_up,
            "prob_down": entry_prob_down,
            "entry_prob_up": entry_prob_up,
            "entry_prob_down": entry_prob_down,
            "exit_prob_up": exit_prob_up,
            "exit_prob_down": exit_prob_down,
            "investment": investment,
            "quantity": quantity,
            "gross_exit_value": gross_exit_value,
            "entry_fee": entry_fee,
            "exit_fee": exit_fee,
            "borrow_fee": borrow_fee,
            "gross_return": gross_return,
            "net_profit": net_profit,
            "cash_after_trade": cash_after_trade,
            "account_value_after_trade": account_value_after_trade,
            "bars_held": bars_held,
            "exit_reason": reason,
        }
    )


def simulate_one_bar(
    df: pd.DataFrame,
    *,
    starting_cash: float,
    min_invest: float,
    max_invest: str,
    threshold: float,
    fee: float,
    confidence_multiplier: float,
    short_confidence_multiplier: float,
    slippage: float,
    spread_pct: float,
    trade_mode: str = "long_only",
    short_entry_threshold: float = 0.45,
    max_short_invest: str = "m",
    borrow_fee: float = 0.0,
) -> tuple[float, float, float, float, float, int, list[dict]]:
    cash = float(starting_cash)
    max_net_value = cash
    min_net_value = cash
    total_invested = 0.0
    total_fees = 0.0
    skipped_signals = 0
    rows: list[dict] = []

    for i, row in df.iterrows():
        if i >= len(df) - 1:
            break
        prob_up = float(row["prob_up"])
        prob_down = float(row.get("prob_down", 0.0))
        side = choose_entry_side(
            prob_up=prob_up,
            prob_down=prob_down,
            trade_mode=trade_mode,
            long_threshold=threshold,
            short_threshold=short_entry_threshold,
        )
        if not side:
            continue
        planner = planned_investment if side == "long" else planned_short_investment
        investment = planner(
            cash=cash,
            max_invest_expr=max_invest if side == "long" else max_short_invest,
            min_invest=min_invest,
            fee=fee,
            prob_up=prob_up if side == "long" else prob_down,
            threshold=threshold if side == "long" else short_entry_threshold,
            confidence_multiplier=(
                confidence_multiplier if side == "long" else short_confidence_multiplier
            ),
        )
        if investment is None:
            skipped_signals += 1
            continue

        entry_bid, entry_ask = synthetic_bid_ask(float(row["close"]), spread_pct)
        exit_row = df.iloc[i + 1]
        exit_bid, exit_ask = synthetic_bid_ask(float(exit_row["close"]), spread_pct)
        entry_price = (entry_ask * (1.0 + slippage)) if side == "long" else (entry_bid * (1.0 - slippage))
        exit_price = (exit_bid * (1.0 - slippage)) if side == "long" else (exit_ask * (1.0 + slippage))
        entry_time = pd.Timestamp(row["open_time"])
        exit_time = pd.Timestamp(exit_row["open_time"])
        entry_fee = investment * fee
        quantity = investment / entry_price
        cash -= investment + entry_fee
        gross_exit_value = quantity * exit_price
        exit_fee = gross_exit_value * fee
        short_borrow_fee = investment * borrow_fee if side == "short" else 0.0
        if side == "long":
            net_profit = gross_exit_value - exit_fee - investment - entry_fee
            cash += gross_exit_value - exit_fee
        else:
            gross_pnl = quantity * (entry_price - exit_price)
            net_profit = gross_pnl - entry_fee - exit_fee - short_borrow_fee
            cash += investment + gross_pnl - exit_fee - short_borrow_fee
        total_invested += investment
        total_fees += entry_fee + exit_fee + short_borrow_fee
        max_net_value = max(max_net_value, cash)
        min_net_value = min(min_net_value, cash)
        append_trade(
            rows,
            entry_time=entry_time,
            exit_time=exit_time,
            entry_price=entry_price,
            exit_price=exit_price,
            entry_bid=entry_bid,
            entry_ask=entry_ask,
            exit_bid=exit_bid,
            exit_ask=exit_ask,
            entry_prob_up=prob_up,
            entry_prob_down=prob_down,
            exit_prob_up=float(exit_row["prob_up"]),
            exit_prob_down=float(exit_row.get("prob_down", 0.0)),
            investment=investment,
            quantity=quantity,
            entry_fee=entry_fee,
            exit_fee=exit_fee,
            gross_exit_value=gross_exit_value,
            net_profit=net_profit,
            cash_after_trade=cash,
            account_value_after_trade=cash,
            bars_held=1,
            reason="one_bar",
            side=side,
            borrow_fee=short_borrow_fee,
        )

    return cash, max_net_value, min_net_value, total_invested, total_fees, skipped_signals, rows


def simulate_live_positions(
    df: pd.DataFrame,
    *,
    starting_cash: float,
    min_invest: float,
    max_invest: str,
    threshold: float,
    exit_threshold: float,
    fee: float,
    confidence_multiplier: float,
    short_confidence_multiplier: float,
    max_hold_bars: int,
    stop_loss: float,
    take_profit: float,
    slippage: float,
    spread_pct: float,
    trade_mode: str = "long_only",
    short_entry_threshold: float = 0.55,
    short_exit_threshold: float = 0.48,
    max_short_invest: str = "m",
    allow_flip_position: bool = False,
    borrow_fee: float = 0.0,
) -> tuple[float, float, float, float, float, int, list[dict]]:
    cash = float(starting_cash)
    max_net_value = cash
    min_net_value = cash
    total_invested = 0.0
    total_fees = 0.0
    skipped_signals = 0
    rows: list[dict] = []

    position_side = ""
    entry_time = pd.Timestamp.min.tz_localize("UTC")
    entry_idx = -1
    entry_price = 0.0
    entry_bid = 0.0
    entry_ask = 0.0
    entry_prob_up = 0.0
    entry_prob_down = 0.0
    investment = 0.0
    entry_fee = 0.0
    quantity = 0.0

    for i, row in df.iterrows():
        timestamp = pd.Timestamp(row["open_time"])
        close = float(row["close"])
        prob_up = float(row["prob_up"])
        prob_down = float(row.get("prob_down", 0.0))
        bid, ask = synthetic_bid_ask(close, spread_pct)
        last_row = i == len(df) - 1

        had_position_at_start = bool(position_side)
        closing_side = ""
        if position_side:
            exit_price = bid * (1.0 - slippage) if position_side == "long" else ask * (1.0 + slippage)
            gross_exit_value = quantity * exit_price
            bars_held = i - entry_idx
            accrued_borrow_fee = investment * borrow_fee * bars_held if position_side == "short" else 0.0
            if position_side == "long":
                liquidation_value = gross_exit_value * (1.0 - fee)
                equity = cash + liquidation_value
                gross_return = gross_exit_value / investment - 1.0 if investment > 0.0 else 0.0
            else:
                cover_fee = gross_exit_value * fee
                gross_pnl = quantity * (entry_price - exit_price)
                equity = cash + investment + gross_pnl - cover_fee - accrued_borrow_fee
                gross_return = (entry_price - exit_price) / entry_price if entry_price > 0.0 else 0.0
            max_net_value = max(max_net_value, equity)
            min_net_value = min(min_net_value, equity)
            reason_fn = exit_reason if position_side == "long" else short_exit_reason
            reason = reason_fn(
                prob_up=prob_up,
                prob_down=prob_down,
                exit_threshold=exit_threshold if position_side == "long" else short_exit_threshold,
                gross_return=gross_return,
                bars_held=bars_held,
                max_hold_bars=max_hold_bars,
                stop_loss=stop_loss,
                take_profit=take_profit,
                last_row=last_row,
            )
            if reason is not None:
                exit_fee = gross_exit_value * fee
                closing_side = position_side
                if closing_side == "long":
                    net_profit = gross_exit_value - exit_fee - investment - entry_fee
                    cash += gross_exit_value - exit_fee
                else:
                    gross_pnl = quantity * (entry_price - exit_price)
                    net_profit = gross_pnl - entry_fee - exit_fee - accrued_borrow_fee
                    cash += investment + gross_pnl - exit_fee - accrued_borrow_fee
                total_fees += exit_fee + accrued_borrow_fee
                append_trade(
                    rows,
                    entry_time=entry_time,
                    exit_time=timestamp,
                    entry_price=entry_price,
                    exit_price=exit_price,
                    entry_bid=entry_bid,
                    entry_ask=entry_ask,
                    exit_bid=bid,
                    exit_ask=ask,
                    entry_prob_up=entry_prob_up,
                    entry_prob_down=entry_prob_down,
                    exit_prob_up=prob_up,
                    exit_prob_down=prob_down,
                    investment=investment,
                    quantity=quantity,
                    entry_fee=entry_fee,
                    exit_fee=exit_fee,
                    gross_exit_value=gross_exit_value,
                    net_profit=net_profit,
                    cash_after_trade=cash,
                    account_value_after_trade=cash,
                    bars_held=bars_held,
                    reason=reason,
                    side=closing_side,
                    borrow_fee=accrued_borrow_fee,
                )
                position_side = ""
                entry_idx = -1
                entry_price = 0.0
                investment = 0.0
                entry_fee = 0.0
                quantity = 0.0
                max_net_value = max(max_net_value, cash)
                min_net_value = min(min_net_value, cash)

        can_enter = (not had_position_at_start or allow_flip_position) and not position_side and not last_row
        selected_side = choose_entry_side(
            prob_up=prob_up,
            prob_down=prob_down,
            trade_mode=trade_mode,
            long_threshold=threshold,
            short_threshold=short_entry_threshold,
        )
        long_signal = selected_side == "long"
        short_signal = selected_side == "short"
        if had_position_at_start and allow_flip_position:
            long_signal = long_signal and closing_side == "short"
            short_signal = short_signal and closing_side == "long"
        if can_enter and (long_signal or short_signal):
            next_side = "long" if long_signal else "short"
            planner = planned_investment if next_side == "long" else planned_short_investment
            next_investment = planner(
                cash=cash,
                max_invest_expr=max_invest if next_side == "long" else max_short_invest,
                min_invest=min_invest,
                fee=fee,
                prob_up=prob_up if next_side == "long" else prob_down,
                threshold=threshold if next_side == "long" else short_entry_threshold,
                confidence_multiplier=(
                    confidence_multiplier if next_side == "long" else short_confidence_multiplier
                ),
            )
            if next_investment is None:
                skipped_signals += 1
            else:
                entry_price = ask * (1.0 + slippage) if next_side == "long" else bid * (1.0 - slippage)
                entry_fee = next_investment * fee
                quantity = next_investment / entry_price
                cash -= next_investment + entry_fee
                total_fees += entry_fee
                total_invested += next_investment
                entry_time = timestamp
                entry_idx = i
                entry_bid = bid
                entry_ask = ask
                entry_prob_up = prob_up
                entry_prob_down = prob_down
                investment = next_investment
                position_side = next_side
                if next_side == "long":
                    equity = cash + quantity * bid * (1.0 - fee)
                else:
                    cover_price = ask * (1.0 + slippage)
                    equity = cash + next_investment + quantity * (entry_price - cover_price) - quantity * cover_price * fee
                max_net_value = max(max_net_value, equity)
                min_net_value = min(min_net_value, equity)

    return cash, max_net_value, min_net_value, total_invested, total_fees, skipped_signals, rows


def simulate_day(
    df: pd.DataFrame,
    start: pd.Timestamp,
    end: pd.Timestamp,
    starting_cash: float,
    min_invest: float,
    max_invest: str,
    threshold: float,
    fee: float,
    confidence_multiplier: float,
    position_mode: str,
    exit_threshold: float,
    max_hold_bars: int,
    stop_loss: float,
    take_profit: float,
    slippage: float,
    spread_pct: float,
    short_confidence_multiplier: float = 3.0,
    window_selection: str = "explicit_window",
    trade_mode: str = "long_only",
    short_entry_threshold: float = 0.55,
    short_exit_threshold: float = 0.48,
    max_short_invest: str = "m",
    allow_flip_position: bool = False,
    borrow_fee: float = 0.0,
    leverage: float = 1.0,
    liquidation_simulation: str = "off",
) -> tuple[dict, pd.DataFrame]:
    validate_inputs(starting_cash, min_invest, threshold, fee, confidence_multiplier, slippage, spread_pct)
    if short_confidence_multiplier <= 0.0:
        raise ValueError("--short-confidence-multiplier must be positive")
    if not (0.0 <= exit_threshold <= 1.0):
        raise ValueError("--exit-threshold must be between 0 and 1")
    if max_hold_bars < 0:
        raise ValueError("--max-hold-bars cannot be negative")
    if stop_loss < 0.0 or take_profit < 0.0:
        raise ValueError("--stop-loss and --take-profit cannot be negative")
    if trade_mode not in {"long_only", "short_only", "long_short"}:
        raise ValueError("--trade-mode must be long_only, short_only, or long_short")
    if trade_mode in {"short_only", "long_short"} and bool(df.attrs.get("prob_down_missing", False)):
        raise ValueError(
            "Prediction file does not contain prob_down. Regenerate predictions with dual up/down training "
            "before running short_only or long_short simulation."
        )
    if not (0.0 < short_entry_threshold < 1.0 and 0.0 < short_exit_threshold < 1.0):
        raise ValueError("Short thresholds must be between 0 and 1")
    if borrow_fee < 0.0:
        raise ValueError("--borrow-fee cannot be negative")
    if not math.isclose(leverage, 1.0):
        raise ValueError("Only 1x leverage is supported")

    if position_mode == "one_bar":
        ending_cash, max_net_value, min_net_value, total_invested, total_fees, skipped_signals, rows = simulate_one_bar(
            df,
            starting_cash=starting_cash,
            min_invest=min_invest,
            max_invest=max_invest,
            threshold=threshold,
            fee=fee,
            confidence_multiplier=confidence_multiplier,
            short_confidence_multiplier=short_confidence_multiplier,
            slippage=slippage,
            spread_pct=spread_pct,
            trade_mode=trade_mode,
            short_entry_threshold=short_entry_threshold,
            max_short_invest=max_short_invest,
            borrow_fee=borrow_fee,
        )
        logic = "independent one-bar trades using synthetic bid/ask execution"
        position_overlap = "none; each signal opens and closes separately"
    else:
        ending_cash, max_net_value, min_net_value, total_invested, total_fees, skipped_signals, rows = simulate_live_positions(
            df,
            starting_cash=starting_cash,
            min_invest=min_invest,
            max_invest=max_invest,
            threshold=threshold,
            exit_threshold=exit_threshold,
            fee=fee,
            confidence_multiplier=confidence_multiplier,
            short_confidence_multiplier=short_confidence_multiplier,
            max_hold_bars=max_hold_bars,
            stop_loss=stop_loss,
            take_profit=take_profit,
            slippage=slippage,
            spread_pct=spread_pct,
            trade_mode=trade_mode,
            short_entry_threshold=short_entry_threshold,
            short_exit_threshold=short_exit_threshold,
            max_short_invest=max_short_invest,
            allow_flip_position=allow_flip_position,
            borrow_fee=borrow_fee,
        )
        logic = "live-like single-position long/short/cash simulation using synthetic bid/ask execution"
        position_overlap = "one open long or short position maximum"

    trades_df = pd.DataFrame(
        rows,
        columns=[
            "side",
            "entry_time",
            "exit_time",
            "entry_price",
            "exit_price",
            "entry_bid",
            "entry_ask",
            "exit_bid",
            "exit_ask",
            "prob_up",
            "prob_down",
            "entry_prob_up",
            "entry_prob_down",
            "exit_prob_up",
            "exit_prob_down",
            "investment",
            "quantity",
            "gross_exit_value",
            "entry_fee",
            "exit_fee",
            "borrow_fee",
            "gross_return",
            "net_profit",
            "cash_after_trade",
            "account_value_after_trade",
            "bars_held",
            "exit_reason",
        ],
    )

    trade_count = len(trades_df)
    if trade_count:
        winning_trades = int((trades_df["net_profit"] > 0.0).sum())
        losing_trades = int((trades_df["net_profit"] <= 0.0).sum())
        profitable_trade_rate = float(winning_trades / trade_count)
        best_invest = summarize_trade(trades_df.loc[trades_df["net_profit"].idxmax()])
        worst_invest = summarize_trade(trades_df.loc[trades_df["net_profit"].idxmin()])
        exit_reason_counts = {str(k): int(v) for k, v in trades_df["exit_reason"].value_counts().items()}
    else:
        winning_trades = 0
        losing_trades = 0
        profitable_trade_rate = 0.0
        best_invest = empty_best_worst()
        worst_invest = empty_best_worst()
        exit_reason_counts = {}

    total_profit = ending_cash - starting_cash
    summed_net_profit = float(trades_df["net_profit"].sum()) if trade_count else 0.0
    accounting_error = float(total_profit - summed_net_profit)
    if abs(accounting_error) > max(1e-8, starting_cash * 1e-10):
        raise RuntimeError(
            f"Simulation accounting mismatch: total_profit={total_profit}, "
            f"summed_net_profit={summed_net_profit}"
        )
    asset_start_price = float(df["close"].iloc[0])
    asset_end_price = float(df["close"].iloc[-1])
    asset_return_pct = (asset_end_price / asset_start_price - 1.0) * 100.0
    entry_count = int(trade_count)
    exit_count = int(trade_count)

    report = {
        "start_utc": str(start),
        "end_utc": str(end),
        "duration": str(end - start),
        "total_days": float((end - start) / pd.Timedelta(days=1)),
        "asset_start_price": asset_start_price,
        "asset_end_price": asset_end_price,
        "asset_return_pct": float(asset_return_pct),
        # Backward-compatible field names used by existing UI text.
        "btc_start_price": asset_start_price,
        "btc_end_price": asset_end_price,
        "btc_return_pct": float(asset_return_pct),
        "predictions": {
            "rows_for_window": int(len(df)),
            "first_open_time": str(df["open_time"].iloc[0]),
            "last_open_time": str(df["open_time"].iloc[-1]),
            "window_selection": window_selection,
        },
        "assumptions": {
            "logic": logic,
            "position_mode": position_mode,
            "starting_cash": float(starting_cash),
            "min_invest": float(min_invest),
            "max_invest": str(max_invest),
            "threshold": float(threshold),
            "exit_threshold": float(exit_threshold),
            "trade_mode": trade_mode,
            "long_entry_threshold": float(threshold),
            "long_exit_threshold": float(exit_threshold),
            "short_entry_threshold": float(short_entry_threshold),
            "short_exit_threshold": float(short_exit_threshold),
            "max_short_invest": str(max_short_invest),
            "allow_flip_position": bool(allow_flip_position),
            "borrow_fee_per_bar": float(borrow_fee),
            "leverage": float(leverage),
            "liquidation_simulation": liquidation_simulation,
            "confidence_multiplier": float(confidence_multiplier),
            "short_confidence_multiplier": float(short_confidence_multiplier),
            "fee_per_side": float(fee),
            "spread_pct": float(spread_pct),
            "slippage": float(slippage),
            "max_hold_bars": int(max_hold_bars),
            "stop_loss": float(stop_loss),
            "take_profit": float(take_profit),
            "position_overlap": position_overlap,
            "execution": "entry at synthetic ask, exit/liquidation at synthetic bid",
        },
        "starting_cash": float(starting_cash),
        "ending_cash": float(ending_cash),
        "ending_equity": float(ending_cash),
        "total_profit": float(total_profit),
        "summed_trade_net_profit": summed_net_profit,
        "accounting_error": accounting_error,
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
        "total_borrow_fees": float(trades_df["borrow_fee"].sum()) if trade_count else 0.0,
        "long_trade_count": int((trades_df["side"] == "long").sum()) if trade_count else 0,
        "short_trade_count": int((trades_df["side"] == "short").sum()) if trade_count else 0,
        "skipped_signals": int(skipped_signals),
        "exit_reason_counts": exit_reason_counts,
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
        window_selection = "explicit_start_duration"
    else:
        window_predictions, start, end = default_test_window(predictions, args.default_test_fraction)
        window_selection = str(window_predictions.attrs.get("window_selection", "default_test_window"))
    report, trades_df = simulate_day(
        df=window_predictions,
        start=start,
        end=end,
        starting_cash=args.starting_cash,
        min_invest=args.min_invest,
        max_invest=args.max_invest,
        threshold=args.threshold,
        fee=args.fee,
        confidence_multiplier=args.confidence_multiplier,
        short_confidence_multiplier=args.short_confidence_multiplier,
        position_mode=args.position_mode,
        exit_threshold=args.exit_threshold,
        max_hold_bars=args.max_hold_bars,
        stop_loss=args.stop_loss,
        take_profit=args.take_profit,
        slippage=args.slippage,
        spread_pct=args.spread_pct,
        window_selection=window_selection,
        trade_mode=args.trade_mode,
        short_entry_threshold=args.short_entry_threshold,
        short_exit_threshold=args.short_exit_threshold,
        max_short_invest=args.max_short_invest,
        allow_flip_position=args.allow_flip_position,
        borrow_fee=args.borrow_fee,
        leverage=args.leverage,
        liquidation_simulation=args.liquidation_simulation,
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
        "asset_start_price",
        "asset_end_price",
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

    if args.comparison_trade_mode:
        if not args.comparison_report_out or not args.comparison_trades_out:
            raise ValueError(
                "--comparison-report-out and --comparison-trades-out are required "
                "with --comparison-trade-mode"
            )
        comparison_report, comparison_trades = simulate_day(
            df=window_predictions,
            start=start,
            end=end,
            starting_cash=args.starting_cash,
            min_invest=args.min_invest,
            max_invest=args.max_invest,
            threshold=args.threshold,
            fee=args.fee,
            confidence_multiplier=args.confidence_multiplier,
            short_confidence_multiplier=args.short_confidence_multiplier,
            position_mode=args.position_mode,
            exit_threshold=args.exit_threshold,
            max_hold_bars=args.max_hold_bars,
            stop_loss=args.stop_loss,
            take_profit=args.take_profit,
            slippage=args.slippage,
            spread_pct=args.spread_pct,
            window_selection=window_selection,
            trade_mode=args.comparison_trade_mode,
            short_entry_threshold=args.short_entry_threshold,
            short_exit_threshold=args.short_exit_threshold,
            max_short_invest=args.max_short_invest,
            allow_flip_position=args.allow_flip_position,
            borrow_fee=args.borrow_fee,
            leverage=args.leverage,
            liquidation_simulation=args.liquidation_simulation,
        )
        comparison_report_path = Path(args.comparison_report_out)
        comparison_trades_path = Path(args.comparison_trades_out)
        comparison_report_path.parent.mkdir(parents=True, exist_ok=True)
        comparison_trades_path.parent.mkdir(parents=True, exist_ok=True)
        comparison_report_path.write_text(json.dumps(comparison_report, indent=2))
        comparison_trades.to_csv(comparison_trades_path, index=False)
        print(
            f"Saved {args.comparison_trade_mode} comparison report to "
            f"{comparison_report_path}"
        )
        print(
            f"Saved {args.comparison_trade_mode} comparison trades to "
            f"{comparison_trades_path}"
        )


if __name__ == "__main__":
    main()
