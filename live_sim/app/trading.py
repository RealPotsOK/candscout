"""Trading and accounting rules for live paper simulation."""

from __future__ import annotations

import re
import math
from dataclasses import dataclass
from datetime import datetime, timezone


@dataclass(frozen=True)
class BuyResult:
    cash_after: float
    quantity: float
    entry_price: float
    investment: float
    entry_fee: float


@dataclass(frozen=True)
class SellResult:
    cash_after: float
    exit_price: float
    gross_exit_value: float
    exit_fee: float
    net_profit: float
    gross_return: float


@dataclass(frozen=True)
class ShortOpenResult:
    cash_after: float
    quantity: float
    entry_price: float
    investment: float
    entry_fee: float


@dataclass(frozen=True)
class ShortCoverResult:
    cash_after: float
    exit_price: float
    gross_exit_value: float
    exit_fee: float
    borrow_fee: float
    net_profit: float
    gross_return: float


_NUMERIC_RE = re.compile(r"^(?:\d+(?:\.\d*)?|\.\d+)$")
_COEFF_M_RE = re.compile(r"^((?:\d+(?:\.\d*)?|\.\d+))\s*\*?\s*m$")
_M_DIV_RE = re.compile(r"^m\s*/\s*((?:\d+(?:\.\d*)?|\.\d+))$")


def parse_max_invest(expr: str, available_cash: float) -> float:
    """Parse safe MAX_INVEST expressions.

    Supported forms:
    - m: all available cash
    - 0.5m or 0.5*m: half available cash
    - m/2: half available cash
    - 25: fixed USDT amount
    """
    if available_cash < 0:
        raise ValueError("available_cash cannot be negative")
    raw = expr.strip().lower().replace(" ", "")
    if raw == "m":
        return available_cash
    if _NUMERIC_RE.match(raw):
        return float(raw)
    coeff_match = _COEFF_M_RE.match(raw)
    if coeff_match:
        coeff = float(coeff_match.group(1))
        if coeff < 0:
            raise ValueError("MAX_INVEST multiplier cannot be negative")
        return coeff * available_cash
    div_match = _M_DIV_RE.match(raw)
    if div_match:
        divisor = float(div_match.group(1))
        if divisor <= 0:
            raise ValueError("MAX_INVEST divisor must be positive")
        return available_cash / divisor
    raise ValueError(f"Unsupported MAX_INVEST expression: {expr!r}")


def calculate_buy(
    *,
    cash: float,
    ask: float,
    max_invest_expr: str,
    min_invest: float,
    fee: float,
    slippage: float,
    prob_up: float = 1.0,
    entry_threshold: float = 0.0,
    confidence_multiplier: float = 1.0,
) -> BuyResult | None:
    if cash <= 0 or ask <= 0:
        return None
    capped_by_cash = cash / (1.0 + fee)
    requested_max = max(0.0, min(parse_max_invest(max_invest_expr, cash), capped_by_cash))
    if requested_max + 1e-12 < min_invest:
        return None
    confidence = (prob_up - entry_threshold) / max(1e-12, 1.0 - entry_threshold)
    confidence = min(max(confidence * confidence_multiplier, 0.0), 1.0)
    investment = min_invest + (requested_max - min_invest) * math.sqrt(confidence)
    if investment + 1e-12 < min_invest:
        return None
    entry_price = ask * (1.0 + slippage)
    entry_fee = investment * fee
    quantity = investment / entry_price
    cash_after = cash - investment - entry_fee
    return BuyResult(
        cash_after=max(cash_after, 0.0),
        quantity=quantity,
        entry_price=entry_price,
        investment=investment,
        entry_fee=entry_fee,
    )


def calculate_sell(
    *,
    cash: float,
    bid: float,
    quantity: float,
    investment: float,
    entry_fee: float,
    fee: float,
    slippage: float,
) -> SellResult:
    if bid <= 0 or quantity < 0:
        raise ValueError("bid must be positive and quantity cannot be negative")
    exit_price = bid * (1.0 - slippage)
    gross_exit_value = quantity * exit_price
    exit_fee = gross_exit_value * fee
    net_profit = gross_exit_value - exit_fee - investment - entry_fee
    gross_return = gross_exit_value / investment - 1.0 if investment > 0 else 0.0
    cash_after = cash + gross_exit_value - exit_fee
    return SellResult(
        cash_after=cash_after,
        exit_price=exit_price,
        gross_exit_value=gross_exit_value,
        exit_fee=exit_fee,
        net_profit=net_profit,
        gross_return=gross_return,
    )


def calculate_short_open(
    *,
    cash: float,
    bid: float,
    max_invest_expr: str,
    min_invest: float,
    fee: float,
    slippage: float,
    prob_up: float,
    entry_threshold: float,
    confidence_multiplier: float,
) -> ShortOpenResult | None:
    if cash <= 0 or bid <= 0:
        return None
    capped_by_cash = cash / (1.0 + fee)
    requested_max = max(0.0, min(parse_max_invest(max_invest_expr, cash), capped_by_cash))
    if requested_max + 1e-12 < min_invest:
        return None
    confidence = (entry_threshold - prob_up) / max(1e-12, entry_threshold)
    confidence = min(max(confidence * confidence_multiplier, 0.0), 1.0)
    investment = min_invest + (requested_max - min_invest) * math.sqrt(confidence)
    entry_price = bid * (1.0 - slippage)
    entry_fee = investment * fee
    return ShortOpenResult(
        cash_after=max(0.0, cash - investment - entry_fee),
        quantity=investment / entry_price,
        entry_price=entry_price,
        investment=investment,
        entry_fee=entry_fee,
    )


def calculate_short_cover(
    *,
    cash: float,
    ask: float,
    quantity: float,
    investment: float,
    entry_price: float,
    entry_fee: float,
    fee: float,
    slippage: float,
    borrow_fee_rate: float,
    bars_held: int,
) -> ShortCoverResult:
    exit_price = ask * (1.0 + slippage)
    gross_exit_value = quantity * exit_price
    exit_fee = gross_exit_value * fee
    borrow_fee = investment * borrow_fee_rate * bars_held
    gross_pnl = quantity * (entry_price - exit_price)
    net_profit = gross_pnl - entry_fee - exit_fee - borrow_fee
    cash_after = cash + investment + gross_pnl - exit_fee - borrow_fee
    return ShortCoverResult(
        cash_after=cash_after,
        exit_price=exit_price,
        gross_exit_value=gross_exit_value,
        exit_fee=exit_fee,
        borrow_fee=borrow_fee,
        net_profit=net_profit,
        gross_return=(entry_price - exit_price) / entry_price if entry_price > 0 else 0.0,
    )


def exit_reason(
    *,
    prob_up: float,
    exit_threshold: float,
    bid: float,
    entry_price: float,
    bars_held: int,
    max_hold_bars: int,
    stop_loss: float,
    take_profit: float,
) -> str | None:
    if prob_up < exit_threshold:
        return "exit_threshold"
    if stop_loss > 0 and bid <= entry_price * (1.0 - stop_loss):
        return "stop_loss"
    if take_profit > 0 and bid >= entry_price * (1.0 + take_profit):
        return "take_profit"
    if bars_held >= max_hold_bars:
        return "max_hold_bars"
    return None


def short_exit_reason(
    *,
    prob_up: float,
    exit_threshold: float,
    ask: float,
    entry_price: float,
    bars_held: int,
    max_hold_bars: int,
    stop_loss: float,
    take_profit: float,
) -> str | None:
    if prob_up > exit_threshold:
        return "short_exit_threshold"
    if stop_loss > 0 and ask >= entry_price * (1.0 + stop_loss):
        return "stop_loss"
    if take_profit > 0 and ask <= entry_price * (1.0 - take_profit):
        return "take_profit"
    if bars_held >= max_hold_bars:
        return "max_hold_bars"
    return None


def parse_utc(value: str) -> datetime:
    raw = value.replace("Z", "+00:00")
    dt = datetime.fromisoformat(raw)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def bars_between(start_iso: str, end_iso: str, interval_seconds: int) -> int:
    start = parse_utc(start_iso)
    end = parse_utc(end_iso)
    elapsed = max(0.0, (end - start).total_seconds())
    return int(elapsed // interval_seconds)
