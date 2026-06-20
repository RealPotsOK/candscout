"""Binance public market data client for live paper trading."""

from __future__ import annotations

import time
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from typing import Any

import requests


@dataclass(frozen=True)
class Candle:
    open_time_ms: int
    close_time_ms: int
    open: float
    high: float
    low: float
    close: float
    volume: float

    @property
    def open_time(self) -> str:
        return iso_from_ms(self.open_time_ms)

    @property
    def close_time(self) -> str:
        return iso_from_ms(self.close_time_ms)

    def to_dict(self) -> dict:
        data = asdict(self)
        data["open_time"] = self.open_time
        data["close_time"] = self.close_time
        return data


@dataclass(frozen=True)
class BookTicker:
    ts_ms: int
    bid: float
    ask: float

    @property
    def ts(self) -> str:
        return iso_from_ms(self.ts_ms)

    @property
    def spread(self) -> float:
        return self.ask - self.bid

    @property
    def spread_pct(self) -> float:
        mid = (self.bid + self.ask) / 2.0
        return self.spread / mid if mid > 0 else 0.0

    def to_dict(self) -> dict:
        return {
            "ts": self.ts,
            "ts_ms": self.ts_ms,
            "bid": self.bid,
            "ask": self.ask,
            "spread": self.spread,
            "spread_pct": self.spread_pct,
        }


class BinanceClient:
    def __init__(self, base_url: str, timeout: float = 15.0) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.session = requests.Session()

    def fetch_klines(self, symbol: str, interval: str, limit: int) -> list[Candle]:
        url = f"{self.base_url}/api/v3/klines"
        response = self.session.get(
            url,
            params={"symbol": symbol.upper(), "interval": interval, "limit": int(limit)},
            timeout=self.timeout,
        )
        response.raise_for_status()
        rows = response.json()
        if not isinstance(rows, list):
            raise ValueError(f"Unexpected Binance klines response: {rows!r}")
        return [parse_kline(row) for row in rows]

    def fetch_completed_klines(self, symbol: str, interval: str, limit: int, *, now_ms: int | None = None) -> list[Candle]:
        now = now_ms if now_ms is not None else int(time.time() * 1000)
        candles = self.fetch_klines(symbol, interval, limit)
        completed = [c for c in candles if c.close_time_ms <= now - 1000]
        return completed

    def fetch_klines_range(
        self,
        symbol: str,
        interval: str,
        start_ms: int,
        end_ms: int,
        interval_ms: int,
    ) -> list[Candle]:
        """Fetch an arbitrary time range using Binance's 1,000-row pages."""
        if start_ms >= end_ms:
            return []
        if interval_ms <= 0:
            raise ValueError("interval_ms must be positive")

        url = f"{self.base_url}/api/v3/klines"
        cursor = int(start_ms)
        candles_by_open: dict[int, Candle] = {}
        while cursor < end_ms:
            response = self.session.get(
                url,
                params={
                    "symbol": symbol.upper(),
                    "interval": interval,
                    "startTime": cursor,
                    "endTime": end_ms - 1,
                    "limit": 1000,
                },
                timeout=self.timeout,
            )
            response.raise_for_status()
            rows = response.json()
            if not isinstance(rows, list):
                raise ValueError(f"Unexpected Binance klines response: {rows!r}")
            if not rows:
                break

            page = [parse_kline(row) for row in rows]
            for candle in page:
                if start_ms <= candle.open_time_ms < end_ms:
                    candles_by_open[candle.open_time_ms] = candle

            next_cursor = page[-1].open_time_ms + interval_ms
            if next_cursor <= cursor:
                raise RuntimeError("Binance kline pagination did not advance")
            cursor = next_cursor
            if len(rows) < 1000:
                break

        return [candles_by_open[key] for key in sorted(candles_by_open)]

    def fetch_book_ticker(self, symbol: str) -> BookTicker:
        url = f"{self.base_url}/api/v3/ticker/bookTicker"
        response = self.session.get(url, params={"symbol": symbol.upper()}, timeout=self.timeout)
        response.raise_for_status()
        data: dict[str, Any] = response.json()
        return BookTicker(
            ts_ms=int(time.time() * 1000),
            bid=float(data["bidPrice"]),
            ask=float(data["askPrice"]),
        )


def parse_kline(row: list[Any]) -> Candle:
    if len(row) < 7:
        raise ValueError(f"Invalid Binance kline row: {row!r}")
    return Candle(
        open_time_ms=int(row[0]),
        open=float(row[1]),
        high=float(row[2]),
        low=float(row[3]),
        close=float(row[4]),
        volume=float(row[5]),
        close_time_ms=int(row[6]),
    )


def iso_from_ms(ms: int) -> str:
    return datetime.fromtimestamp(ms / 1000.0, tz=timezone.utc).isoformat().replace("+00:00", "Z")


def ms_from_iso(value: str) -> int:
    raw = value.strip().replace("Z", "+00:00")
    dt = datetime.fromisoformat(raw)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return int(dt.timestamp() * 1000)


def synthetic_book_ticker(candle: Candle, spread_pct: float) -> BookTicker:
    """Approximate historical execution around the candle close."""
    if not 0.0 <= spread_pct < 2.0:
        raise ValueError("spread_pct must be >= 0 and < 2")
    half_spread = spread_pct / 2.0
    return BookTicker(
        ts_ms=candle.close_time_ms,
        bid=candle.close * (1.0 - half_spread),
        ask=candle.close * (1.0 + half_spread),
    )
