#!/usr/bin/env python3
"""Choose a Yahoo stock symbol with enough historical data."""

from __future__ import annotations

import argparse
import random
import sys
from datetime import timedelta

import pandas as pd

from download import DEFAULT_RANDOM_STOCKS, fetch_yahoo_ohlcv, parse_utc_timestamp

YAHOO_INTRADAY_LIMIT_DAYS = {
    "1m": 7,
    "2m": 60,
    "5m": 60,
    "15m": 60,
    "30m": 60,
    "60m": 730,
    "90m": 60,
    "1h": 730,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Select a stock with sufficient Yahoo history.")
    parser.add_argument("--symbol", default="", help="Specific symbol to validate. If omitted, choose a random eligible stock.")
    parser.add_argument("--stock-list", default=",".join(DEFAULT_RANDOM_STOCKS), help="Comma-separated candidate symbols")
    parser.add_argument("--interval", default="1d", help="Yahoo interval to validate")
    parser.add_argument("--start", required=True, help="UTC start date/time for validation")
    parser.add_argument("--end", required=True, help="UTC end date/time for validation")
    parser.add_argument("--min-rows", type=int, default=700, help="Minimum downloaded rows required")
    parser.add_argument("--min-history-years", type=float, default=3.0, help="Minimum calendar history span required")
    parser.add_argument("--seed", type=int, default=None, help="Optional deterministic random seed")
    return parser.parse_args()


def candidate_symbols(raw_symbols: str, explicit_symbol: str) -> list[str]:
    if explicit_symbol.strip():
        return [explicit_symbol.strip().upper()]
    symbols = [x.strip().upper() for x in raw_symbols.split(",") if x.strip()]
    if not symbols:
        symbols = DEFAULT_RANDOM_STOCKS.copy()
    return list(dict.fromkeys(symbols))


def history_summary(symbol: str, interval: str, start: pd.Timestamp, end: pd.Timestamp) -> tuple[int, pd.Timestamp, pd.Timestamp]:
    df = fetch_yahoo_ohlcv(symbol, interval, start.to_pydatetime(), end.to_pydatetime())
    open_times = pd.to_datetime(df["open_time"], utc=True)
    return len(df), pd.Timestamp(open_times.iloc[0]), pd.Timestamp(open_times.iloc[-1])


def validate_window(interval: str, start: pd.Timestamp, end: pd.Timestamp, min_history_years: float) -> None:
    requested_days = float((end - start) / pd.Timedelta(days=1))
    min_days = max(0.0, min_history_years * 365.0 - 14.0)

    limit_days = YAHOO_INTRADAY_LIMIT_DAYS.get(interval)
    if limit_days is not None and requested_days > limit_days + 1.0:
        raise SystemExit(
            "Yahoo Finance does not provide that much history for this intraday interval.\n"
            f"Interval: {interval}\n"
            f"Requested date window: {requested_days:.1f} days\n"
            f"Approx Yahoo limit for {interval}: {limit_days} days\n"
            "Fix one of these:\n"
            f"  STOCK_LOOKBACK_DAYS={limit_days}\n"
            "  STOCK_INTERVAL=1d\n"
            "If you need multi-year intraday stock data, Yahoo is the wrong data source."
        )

    if min_days > requested_days + 1.0:
        raise SystemExit(
            "Stock selection cannot succeed with the current settings.\n"
            f"Requested date window: {requested_days:.1f} days\n"
            f"Minimum history required: {min_history_years:g} years (~{min_days:.0f} days)\n"
            "Fix one of these:\n"
            f"  STOCK_LOOKBACK_DAYS={max(int(min_history_years * 365), int(requested_days))}\n"
            f"  STOCK_MIN_HISTORY_YEARS={requested_days / 365.0:.2f}\n"
            "For true 3-year stock training, use STOCK_INTERVAL=1d and STOCK_LOOKBACK_DAYS=1095."
        )

    if limit_days is None:
        return

    if min_days > limit_days + 1.0:
        raise SystemExit(
            "Minimum history requirement is impossible for this Yahoo intraday interval.\n"
            f"Interval: {interval}\n"
            f"Minimum history required: {min_history_years:g} years (~{min_days:.0f} days)\n"
            f"Approx Yahoo limit for {interval}: {limit_days} days\n"
            "Fix one of these:\n"
            f"  STOCK_MIN_HISTORY_YEARS={limit_days / 365.0:.2f}\n"
            "  STOCK_INTERVAL=1d\n"
            "For 3-year stock runs, use daily candles."
        )


def main() -> None:
    args = parse_args()
    start = pd.Timestamp(parse_utc_timestamp(args.start))
    end = pd.Timestamp(parse_utc_timestamp(args.end))
    if end <= start:
        raise ValueError("--end must be after --start")
    validate_window(args.interval, start, end, args.min_history_years)

    symbols = candidate_symbols(args.stock_list, args.symbol)
    rng = random.Random(args.seed)
    if not args.symbol.strip():
        rng.shuffle(symbols)

    min_days = max(0.0, args.min_history_years * 365.0 - 14.0)
    checked: list[str] = []
    failures: list[str] = []

    print(
        f"Checking {len(symbols)} stock candidate(s) for >= {args.min_history_years:g} years "
        f"and >= {args.min_rows} rows from {start} to {end}...",
        file=sys.stderr,
    )

    for symbol in symbols:
        checked.append(symbol)
        try:
            rows, first_time, last_time = history_summary(symbol, args.interval, start, end)
            span_days = (last_time - first_time) / pd.Timedelta(days=1)
            if rows >= args.min_rows and span_days >= min_days:
                print(
                    f"Selected {symbol}: rows={rows}, range={first_time} to {last_time}, span_days={span_days:.1f}",
                    file=sys.stderr,
                )
                print(symbol)
                return
            failures.append(f"{symbol}: rows={rows}, span_days={span_days:.1f}")
            print(f"Rejected {symbol}: rows={rows}, span_days={span_days:.1f}", file=sys.stderr)
        except Exception as exc:  # yfinance can fail per-symbol; keep looking.
            failures.append(f"{symbol}: {exc}")
            print(f"Rejected {symbol}: {exc}", file=sys.stderr)

    print("No eligible stock found.", file=sys.stderr)
    print("Criteria:", file=sys.stderr)
    print(f"  min rows: {args.min_rows}", file=sys.stderr)
    print(f"  min history years: {args.min_history_years:g}", file=sys.stderr)
    print("Checked:", file=sys.stderr)
    for failure in failures[:50]:
        print(f"  {failure}", file=sys.stderr)
    if len(failures) > 50:
        print(f"  ... {len(failures) - 50} more", file=sys.stderr)
    raise SystemExit(1)


if __name__ == "__main__":
    main()
