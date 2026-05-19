#!/usr/bin/env python3
"""Download crypto or stock OHLCV data to Parquet."""

from __future__ import annotations

import argparse
import math
import random
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import requests

BINANCE_KLINES_URL = "https://api.binance.com/api/v3/klines"
DEFAULT_RANDOM_STOCKS = [
    "AAPL",
    "MSFT",
    "NVDA",
    "AMZN",
    "GOOGL",
    "META",
    "TSLA",
    "AMD",
    "NFLX",
    "JPM",
    "V",
    "UNH",
    "COST",
    "AVGO",
    "WMT",
]


def parse_utc_timestamp(value: str) -> datetime:
    """Parse a UTC timestamp, accepting strict ISO and non-zero-padded dates."""
    ts = pd.Timestamp(value)
    if ts.tzinfo is None:
        ts = ts.tz_localize(timezone.utc)
    else:
        ts = ts.tz_convert(timezone.utc)
    return ts.to_pydatetime()


def datetime_to_ms(dt: datetime) -> int:
    return int(dt.timestamp() * 1000)


def interval_to_minutes(interval: str) -> int:
    unit = interval[-1]
    value = int(interval[:-1])
    if unit == "m":
        return value
    if unit == "h":
        return value * 60
    if unit == "d":
        return value * 60 * 24
    raise ValueError(f"Unsupported interval: {interval}")


def format_bytes(num_bytes: float) -> str:
    if num_bytes >= 1024**3:
        return f"{num_bytes / 1024**3:.3f} GiB"
    if num_bytes >= 1024**2:
        return f"{num_bytes / 1024**2:.2f} MiB"
    if num_bytes >= 1024:
        return f"{num_bytes / 1024:.1f} KiB"
    return f"{num_bytes:.0f} B"


def print_progress(
    rows_done: int,
    rows_expected: int,
    requests_done: int,
    requests_expected: int,
    bytes_done: int,
    started_at: float,
) -> None:
    elapsed = max(time.time() - started_at, 1e-9)
    row_ratio = min(rows_done / rows_expected, 1.0) if rows_expected else 1.0
    bar_width = 28
    filled = int(bar_width * row_ratio)
    bar = "#" * filled + "." * (bar_width - filled)
    speed = bytes_done / elapsed
    estimated_total = (bytes_done / rows_done) * rows_expected if rows_done else 0.0

    line = (
        f"\r[{bar}] {row_ratio * 100:6.2f}% "
        f"candles {rows_done:,}/{rows_expected:,} "
        f"requests {requests_done:,}/{requests_expected:,} "
        f"downloaded {format_bytes(bytes_done)}"
    )

    if estimated_total:
        line += f" / est total {format_bytes(estimated_total)}"
    else:
        line += " / est total unknown"

    line += f" speed {format_bytes(speed)}/s"
    sys.stderr.write(line)
    sys.stderr.flush()


def fetch_klines(symbol: str, interval: str, start_ms: int, end_ms: int) -> list[list]:
    all_rows: list[list] = []
    cursor = start_ms
    step_ms = interval_to_minutes(interval) * 60_000
    expected_rows = max(0, math.ceil((end_ms - start_ms) / step_ms))
    expected_requests = max(1, math.ceil(expected_rows / 1000))
    requests_done = 0
    bytes_done = 0
    started_at = time.time()

    print(
        f"Expected candles: {expected_rows:,}; requests: {expected_requests:,}; "
        "total download size will be estimated after the first response.",
        file=sys.stderr,
    )

    # Binance returns at most 1000 rows per request.
    while cursor < end_ms:
        params = {
            "symbol": symbol,
            "interval": interval,
            "startTime": cursor,
            "endTime": end_ms - 1,
            "limit": 1000,
        }
        response = requests.get(BINANCE_KLINES_URL, params=params, timeout=30)
        response.raise_for_status()
        bytes_done += len(response.content)
        rows = response.json()
        requests_done += 1

        if not rows:
            break

        all_rows.extend(rows)

        last_open_time = int(rows[-1][0])
        next_cursor = last_open_time + step_ms

        if next_cursor <= cursor:
            # Safety against infinite loops in malformed responses.
            next_cursor = cursor + step_ms

        cursor = next_cursor
        print_progress(
            rows_done=len(all_rows),
            rows_expected=expected_rows,
            requests_done=requests_done,
            requests_expected=expected_requests,
            bytes_done=bytes_done,
            started_at=started_at,
        )

    print(file=sys.stderr)

    return all_rows


def build_dataframe(rows: list[list], start_ms: int, end_ms: int) -> pd.DataFrame:
    if not rows:
        raise ValueError("No klines returned for the requested range.")

    frame = pd.DataFrame(
        rows,
        columns=[
            "open_time_ms",
            "open",
            "high",
            "low",
            "close",
            "volume",
            "close_time_ms",
            "quote_asset_volume",
            "number_of_trades",
            "taker_buy_base_volume",
            "taker_buy_quote_volume",
            "ignore",
        ],
    )

    numeric_cols = [
        "open",
        "high",
        "low",
        "close",
        "volume",
        "quote_asset_volume",
        "taker_buy_base_volume",
        "taker_buy_quote_volume",
    ]
    int_cols = ["open_time_ms", "close_time_ms", "number_of_trades"]

    for col in numeric_cols:
        frame[col] = pd.to_numeric(frame[col], errors="coerce")
    for col in int_cols:
        frame[col] = pd.to_numeric(frame[col], errors="coerce").astype("Int64")

    frame = frame.drop(columns=["ignore"]).dropna()

    frame = frame[(frame["open_time_ms"] >= start_ms) & (frame["open_time_ms"] < end_ms)]
    frame = frame.drop_duplicates(subset=["open_time_ms"]).sort_values("open_time_ms").reset_index(drop=True)

    frame["open_time"] = pd.to_datetime(frame["open_time_ms"], unit="ms", utc=True)
    frame["close_time"] = pd.to_datetime(frame["close_time_ms"], unit="ms", utc=True)

    if frame.empty:
        raise ValueError("Dataframe is empty after boundary filtering.")

    if not frame["open_time_ms"].is_monotonic_increasing:
        raise ValueError("open_time_ms is not sorted ascending.")

    if frame["open_time_ms"].duplicated().any():
        raise ValueError("Duplicate open_time_ms values found after de-duplication.")

    return frame


def fetch_yahoo_ohlcv(symbol: str, interval: str, start_dt: datetime, end_dt: datetime) -> pd.DataFrame:
    try:
        import yfinance as yf
    except ImportError as exc:
        raise RuntimeError(
            "Stock downloads require yfinance. Run `make install` after updating requirements.txt."
        ) from exc

    frame = yf.download(
        symbol,
        start=start_dt,
        end=end_dt,
        interval=interval,
        auto_adjust=False,
        progress=False,
        threads=False,
    )
    if frame.empty:
        raise ValueError(f"No Yahoo Finance rows returned for {symbol} {interval}.")

    if isinstance(frame.columns, pd.MultiIndex):
        frame.columns = frame.columns.get_level_values(0)

    frame = frame.reset_index()
    time_col = "Datetime" if "Datetime" in frame.columns else "Date"
    frame = frame.rename(
        columns={
            time_col: "open_time",
            "Open": "open",
            "High": "high",
            "Low": "low",
            "Close": "close",
            "Volume": "volume",
        }
    )

    required = ["open_time", "open", "high", "low", "close", "volume"]
    missing = set(required) - set(frame.columns)
    if missing:
        raise ValueError(f"Yahoo Finance response missing columns: {sorted(missing)}")

    frame = frame[required].copy()
    frame["open_time"] = pd.to_datetime(frame["open_time"], utc=True)
    for col in ["open", "high", "low", "close", "volume"]:
        frame[col] = pd.to_numeric(frame[col], errors="coerce")

    start_ts = pd.Timestamp(start_dt)
    end_ts = pd.Timestamp(end_dt)
    frame = frame[(frame["open_time"] >= start_ts) & (frame["open_time"] < end_ts)]
    frame = frame.dropna(subset=required).drop_duplicates("open_time").sort_values("open_time").reset_index(drop=True)
    if frame.empty:
        raise ValueError("Yahoo Finance dataframe is empty after boundary filtering.")

    frame["symbol"] = symbol.upper()
    frame["source"] = "yahoo"
    return frame


def choose_random_stock(raw_symbols: str) -> str:
    symbols = [x.strip().upper() for x in raw_symbols.split(",") if x.strip()]
    if not symbols:
        symbols = DEFAULT_RANDOM_STOCKS
    return random.choice(symbols)


def default_output_path(source: str, symbol: str, interval: str, start: datetime, end: datetime) -> Path:
    start_str = start.strftime("%Y%m%dT%H%M%SZ")
    end_str = end.strftime("%Y%m%dT%H%M%SZ")
    prefix = symbol.lower().replace("/", "_").replace(".", "_")
    filename = f"{prefix}_{source}_{interval}_{start_str}_{end_str}.parquet"
    return Path("data") / filename


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download crypto or stock OHLCV candles to Parquet.")
    parser.add_argument(
        "--source",
        choices=["binance", "yahoo"],
        default="binance",
        help="Data source: binance for crypto spot klines, yahoo for stocks/ETFs",
    )
    parser.add_argument("--symbol", default="BTCUSDT", help="Trading symbol (default: BTCUSDT)")
    parser.add_argument(
        "--random-stock",
        action="store_true",
        help="With --source yahoo, choose a random stock symbol from --stock-list",
    )
    parser.add_argument(
        "--stock-list",
        default=",".join(DEFAULT_RANDOM_STOCKS),
        help="Comma-separated symbols used by --random-stock",
    )
    parser.add_argument("--interval", default="5m", help="Kline interval (default: 5m)")
    parser.add_argument("--start", required=True, help="UTC start (ISO8601), inclusive")
    parser.add_argument("--end", required=True, help="UTC end (ISO8601), exclusive")
    parser.add_argument("--out", default=None, help="Output Parquet file path")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    start_dt = parse_utc_timestamp(args.start)
    end_dt = parse_utc_timestamp(args.end)

    if end_dt <= start_dt:
        raise ValueError("--end must be greater than --start")

    symbol = args.symbol.upper()
    if args.random_stock:
        if args.source != "yahoo":
            raise ValueError("--random-stock requires --source yahoo")
        symbol = choose_random_stock(args.stock_list)
        print(f"Selected random stock symbol: {symbol}")

    if args.source == "binance":
        start_ms = datetime_to_ms(start_dt)
        end_ms = datetime_to_ms(end_dt)
        rows = fetch_klines(symbol, args.interval, start_ms, end_ms)
        frame = build_dataframe(rows, start_ms, end_ms)
        frame["symbol"] = symbol
        frame["source"] = "binance"
    else:
        frame = fetch_yahoo_ohlcv(symbol, args.interval, start_dt, end_dt)

    output_path = Path(args.out) if args.out else default_output_path(args.source, symbol, args.interval, start_dt, end_dt)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(output_path, index=False)

    print(f"Saved {len(frame)} candles for {symbol} from {args.source} to {output_path}")
    print(f"First open_time: {frame['open_time'].iloc[0]}")
    print(f"Last open_time:  {frame['open_time'].iloc[-1]}")


if __name__ == "__main__":
    main()
