from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch
import os
import unittest

import numpy as np
import pandas as pd

from app.bot import LivePaperBot
from app.coinbase_exec import RealOrderResult, RealTradeService
from app.config import load_config, parse_retrain_frequency
from app.market import Candle, synthetic_book_ticker
from app.model_runner import build_sequence_input
from app.scheduler import floor_to_interval, live_candle_cache_path, seed_live_candle_cache
from app.store import Store
from app.trading import (
    calculate_buy,
    calculate_sell,
    calculate_short_cover,
    calculate_short_open,
    exit_reason,
    parse_max_invest,
)


class MaxInvestParserTest(unittest.TestCase):
    def test_supported_forms(self) -> None:
        self.assertEqual(parse_max_invest("m", 100.0), 100.0)
        self.assertEqual(parse_max_invest("0.5m", 100.0), 50.0)
        self.assertEqual(parse_max_invest("0.5*m", 100.0), 50.0)
        self.assertEqual(parse_max_invest("m/2", 100.0), 50.0)
        self.assertEqual(parse_max_invest("25", 100.0), 25.0)

    def test_rejects_unsafe_forms(self) -> None:
        for expr in ["__import__('os')", "m*0.5", "cash", "m/0", "1+2"]:
            with self.subTest(expr=expr):
                with self.assertRaises(ValueError):
                    parse_max_invest(expr, 100.0)


class AccountingTest(unittest.TestCase):
    def test_buy_and_sell_accounting_with_fee(self) -> None:
        buy = calculate_buy(
            cash=100.0,
            ask=20.0,
            max_invest_expr="0.5m",
            min_invest=1.0,
            fee=0.001,
            slippage=0.0,
        )
        assert buy is not None
        self.assertAlmostEqual(buy.investment, 50.0)
        self.assertAlmostEqual(buy.entry_fee, 0.05)
        self.assertAlmostEqual(buy.quantity, 2.5)
        self.assertAlmostEqual(buy.cash_after, 49.95)

        sell = calculate_sell(
            cash=buy.cash_after,
            bid=21.0,
            quantity=buy.quantity,
            investment=buy.investment,
            entry_fee=buy.entry_fee,
            fee=0.001,
            slippage=0.0,
        )
        self.assertAlmostEqual(sell.gross_exit_value, 52.5)
        self.assertAlmostEqual(sell.exit_fee, 0.0525)
        self.assertAlmostEqual(sell.net_profit, 2.3975)
        self.assertAlmostEqual(sell.cash_after, 102.3975)

    def test_buy_sizing_uses_confidence(self) -> None:
        near_threshold = calculate_buy(
            cash=100.0,
            ask=10.0,
            max_invest_expr="m",
            min_invest=1.0,
            fee=0.0,
            slippage=0.0,
            prob_up=0.52,
            entry_threshold=0.52,
            confidence_multiplier=1.0,
        )
        high_confidence = calculate_buy(
            cash=100.0,
            ask=10.0,
            max_invest_expr="m",
            min_invest=1.0,
            fee=0.0,
            slippage=0.0,
            prob_up=1.0,
            entry_threshold=0.52,
            confidence_multiplier=1.0,
        )
        assert near_threshold is not None
        assert high_confidence is not None
        self.assertAlmostEqual(near_threshold.investment, 1.0)
        self.assertAlmostEqual(high_confidence.investment, 100.0)

    def test_short_open_and_cover_accounting(self) -> None:
        short = calculate_short_open(
            cash=100.0,
            bid=20.0,
            max_invest_expr="0.5m",
            min_invest=1.0,
            fee=0.001,
            slippage=0.0,
            prob_up=0.0,
            entry_threshold=0.45,
            confidence_multiplier=1.0,
        )
        assert short is not None
        self.assertAlmostEqual(short.investment, 50.0)
        self.assertAlmostEqual(short.cash_after, 49.95)
        cover = calculate_short_cover(
            cash=short.cash_after,
            ask=18.0,
            quantity=short.quantity,
            investment=short.investment,
            entry_price=short.entry_price,
            entry_fee=short.entry_fee,
            fee=0.001,
            slippage=0.0,
            borrow_fee_rate=0.0001,
            bars_held=2,
        )
        self.assertGreater(cover.net_profit, 0.0)
        self.assertAlmostEqual(cover.cash_after, 100.0 + cover.net_profit)

    def test_exit_rules(self) -> None:
        self.assertEqual(
            exit_reason(
                prob_up=0.44,
                exit_threshold=0.45,
                bid=100.0,
                entry_price=100.0,
                bars_held=1,
                max_hold_bars=60,
                stop_loss=0.002,
                take_profit=0.004,
            ),
            "exit_threshold",
        )
        self.assertEqual(
            exit_reason(
                prob_up=0.6,
                exit_threshold=0.45,
                bid=99.7,
                entry_price=100.0,
                bars_held=1,
                max_hold_bars=60,
                stop_loss=0.002,
                take_profit=0.004,
            ),
            "stop_loss",
        )
        self.assertEqual(
            exit_reason(
                prob_up=0.6,
                exit_threshold=0.45,
                bid=100.5,
                entry_price=100.0,
                bars_held=1,
                max_hold_bars=60,
                stop_loss=0.002,
                take_profit=0.004,
            ),
            "take_profit",
        )
        self.assertEqual(
            exit_reason(
                prob_up=0.6,
                exit_threshold=0.45,
                bid=100.0,
                entry_price=100.0,
                bars_held=60,
                max_hold_bars=60,
                stop_loss=0.002,
                take_profit=0.004,
            ),
            "max_hold_bars",
        )


class RealTradingSafetyTest(unittest.TestCase):
    def live_env(self, **overrides: str) -> dict[str, str]:
        env = {
            "EXECUTION_MODE": "coinbase_live",
            "REAL_TRADING_ENABLED": "true",
            "REAL_REQUIRE_MANUAL_ARM": "true",
            "REAL_MAX_TOTAL_USD": "20",
            "REAL_MAX_ORDER_USD": "5",
            "REAL_MIN_ORDER_USD": "1",
            "COINBASE_PRODUCT_ID": "SOL-USD",
            "COINBASE_API_KEY": "organizations/test/apiKeys/test",
            "COINBASE_API_SECRET": "test-secret",
            "REAL_ARM_TOKEN": "token",
            "SYMBOL": "SOLUSDT",
            "TRADE_MODE": "long_only",
            "LEVERAGE": "1",
            "BORROW_FEE": "0",
            "LIQUIDATION_SIMULATION": "off",
        }
        env.update(overrides)
        return env

    def test_config_rejects_live_cap_above_20(self) -> None:
        with patch.dict(os.environ, self.live_env(REAL_MAX_TOTAL_USD="21"), clear=True):
            with self.assertRaises(ValueError):
                load_config()

    def test_config_rejects_real_short_mode(self) -> None:
        with patch.dict(os.environ, self.live_env(TRADE_MODE="long_short"), clear=True):
            with self.assertRaises(ValueError):
                load_config()

    def test_config_rejects_missing_coinbase_credentials(self) -> None:
        with patch.dict(os.environ, self.live_env(COINBASE_API_KEY="", COINBASE_API_SECRET=""), clear=True):
            with self.assertRaises(ValueError):
                load_config()

    def test_real_buy_never_exceeds_order_or_total_cap(self) -> None:
        class FakeExecutor:
            def __init__(self) -> None:
                self.buy_quotes: list[float] = []

            def available_balances(self) -> dict[str, float]:
                return {"USD": 100.0, "SOL": 0.0}

            def market_buy_quote(self, quote_usd: float) -> RealOrderResult:
                self.buy_quotes.append(quote_usd)
                return RealOrderResult(
                    status="filled",
                    product_id="SOL-USD",
                    side="BUY",
                    client_order_id="test-buy",
                    coinbase_order_id="order-buy",
                    requested_usd=quote_usd,
                    requested_sol=0.0,
                    filled_usd=quote_usd,
                    filled_sol=quote_usd / 100.0,
                    average_price=100.0,
                    fee_usd=0.0,
                    raw_response={"order": {"status": "FILLED"}},
                )

        with patch.dict(os.environ, self.live_env(), clear=True), TemporaryDirectory() as tmp:
            cfg = load_config()
            store = Store(str(Path(tmp) / "live.db"))
            store.initialize_account(100.0, "2026-01-01T00:00:00Z")
            store.set_real_armed(armed=True, ts="2026-01-01T00:00:00Z")
            fake = FakeExecutor()
            service = RealTradeService(cfg, store, fake)  # type: ignore[arg-type]
            service.execute_buy(
                ts="2026-01-01T00:05:00Z",
                candle_open_time="2026-01-01T00:00:00Z",
                planned_usd=1000.0,
                reason="test",
            )
            self.assertEqual(fake.buy_quotes, [5.0])
            self.assertAlmostEqual(store.real_state()["bot_cost_usd"], 5.0)
            store.close()

    def test_real_sell_uses_only_bot_tracked_sol(self) -> None:
        class FakeExecutor:
            sell_calls = 0

            def market_sell_base(self, base_sol: float) -> RealOrderResult:
                self.sell_calls += 1
                return RealOrderResult(
                    status="filled",
                    product_id="SOL-USD",
                    side="SELL",
                    client_order_id="test-sell",
                    coinbase_order_id="order-sell",
                    requested_usd=0.0,
                    requested_sol=base_sol,
                    filled_usd=10.0,
                    filled_sol=base_sol,
                    average_price=100.0,
                    fee_usd=0.0,
                    raw_response={"order": {"status": "FILLED"}},
                )

        with patch.dict(os.environ, self.live_env(), clear=True), TemporaryDirectory() as tmp:
            cfg = load_config()
            store = Store(str(Path(tmp) / "live.db"))
            store.initialize_account(100.0, "2026-01-01T00:00:00Z")
            store.set_real_armed(armed=True, ts="2026-01-01T00:00:00Z")
            fake = FakeExecutor()
            service = RealTradeService(cfg, store, fake)  # type: ignore[arg-type]
            service.execute_sell_all(
                ts="2026-01-01T00:05:00Z",
                candle_open_time="2026-01-01T00:00:00Z",
                reason="test",
            )
            self.assertEqual(fake.sell_calls, 0)
            self.assertEqual(store.latest_real_order()["status"], "skipped")
            store.close()


class SequenceBuilderTest(unittest.TestCase):
    def test_build_sequence_shape(self) -> None:
        candles = []
        base_ms = 1_700_000_000_000
        for idx in range(6):
            close = 100.0 + idx
            candles.append(
                Candle(
                    open_time_ms=base_ms + idx * 300_000,
                    close_time_ms=base_ms + (idx + 1) * 300_000 - 1,
                    open=close - 0.2,
                    high=close + 0.5,
                    low=close - 0.5,
                    close=close,
                    volume=1000.0 + idx,
                )
            )
        x, last = build_sequence_input(candles, lookback=5)
        self.assertEqual(x.shape, (1, 5, 5))
        self.assertTrue(np.isfinite(x).all())
        self.assertEqual(last.close, 105.0)

    def test_build_technical_sequence_shape(self) -> None:
        candles = []
        base_ms = 1_700_000_000_000
        for idx in range(130):
            close = 100.0 + idx * 0.1
            candles.append(
                Candle(
                    open_time_ms=base_ms + idx * 300_000,
                    close_time_ms=base_ms + (idx + 1) * 300_000 - 1,
                    open=close - 0.05,
                    high=close + 0.2,
                    low=close - 0.2,
                    close=close,
                    volume=1000.0 + idx,
                )
            )
        channels = [
            "open_to_prev_close",
            "high_to_prev_close",
            "low_to_prev_close",
            "close_to_prev_close",
            "log_volume",
            "return_1bar",
            "return_20bar",
            "volatility_20bar",
            "sma_50_ratio",
            "ema_26_ratio",
            "macd_pct",
            "rsi_14",
            "volume_sma_ratio_20",
            "close_position_in_range_20",
        ]
        x, last = build_sequence_input(candles, lookback=70, channel_names=channels)
        self.assertEqual(x.shape, (1, 70, len(channels)))
        self.assertTrue(np.isfinite(x).all())
        self.assertEqual(last.close, candles[-1].close)


class CatchUpTest(unittest.TestCase):
    def test_synthetic_ticker_uses_close_and_configured_spread(self) -> None:
        candle = Candle(
            open_time_ms=1_700_000_000_000,
            close_time_ms=1_700_000_299_999,
            open=99.0,
            high=101.0,
            low=98.0,
            close=100.0,
            volume=1000.0,
        )
        ticker = synthetic_book_ticker(candle, 0.002)
        self.assertAlmostEqual(ticker.bid, 99.9)
        self.assertAlmostEqual(ticker.ask, 100.1)
        self.assertAlmostEqual(ticker.spread_pct, 0.002)
        self.assertEqual(ticker.ts_ms, candle.close_time_ms)

    def test_replays_every_completed_candle_after_latest_decision(self) -> None:
        interval_ms = 300_000
        base_ms = 1_700_000_000_000
        candles = [
            Candle(
                open_time_ms=base_ms + idx * interval_ms,
                close_time_ms=base_ms + (idx + 1) * interval_ms - 1,
                open=100.0 + idx,
                high=100.5 + idx,
                low=99.5 + idx,
                close=100.0 + idx,
                volume=1000.0 + idx,
            )
            for idx in range(6)
        ]

        class FakeMarket:
            def __init__(self, rows: list[Candle]) -> None:
                self.rows = rows
                self.range_calls: list[tuple[int, int, int]] = []

            def fetch_klines_range(
                self,
                symbol: str,
                interval: str,
                start_ms: int,
                end_ms: int,
                requested_interval_ms: int,
            ) -> list[Candle]:
                self.range_calls.append((start_ms, end_ms, requested_interval_ms))
                return [row for row in self.rows if start_ms <= row.open_time_ms < end_ms]

        class FakeModel:
            lookback = 2

            def maybe_reload(self) -> bool:
                return False

            def required_candles(self, buffer: int = 8) -> int:
                return 3

            def predict(self, rows: list[Candle]) -> tuple[float, Candle]:
                self.last_history = rows
                return 0.1, rows[-1]

        with TemporaryDirectory() as tmp:
            cfg = load_config()
            object.__setattr__(cfg, "interval", "5m")
            object.__setattr__(cfg, "entry_threshold", 0.05)
            object.__setattr__(cfg, "catchup_enabled", True)
            object.__setattr__(cfg, "catchup_spread_pct", 0.001)
            object.__setattr__(cfg, "catchup_max_bars", 0)
            store = Store(str(Path(tmp) / "live.db"))
            store.initialize_account(100.0, candles[0].open_time)
            seed_ticker = synthetic_book_ticker(candles[2], cfg.catchup_spread_pct)
            store.insert_decision(
                {
                    "ts": candles[2].close_time,
                    "candle_open_time": candles[2].open_time,
                    "prob_up": 0.1,
                    "action": "cash",
                    "reason": "below_entry_threshold",
                    "entry_threshold": cfg.entry_threshold,
                    "exit_threshold": cfg.exit_threshold,
                    "cash": 100.0,
                    "sol_qty": 0.0,
                    "equity": 100.0,
                    "bid": seed_ticker.bid,
                    "ask": seed_ticker.ask,
                    "spread_pct": seed_ticker.spread_pct,
                }
            )

            market = FakeMarket(candles)
            model = FakeModel()

            class FakeRealTrader:
                def execute_buy(self, **_kwargs: object) -> None:
                    raise AssertionError("catch-up must not place real buys")

                def execute_sell_all(self, **_kwargs: object) -> None:
                    raise AssertionError("catch-up must not place real sells")

            bot = LivePaperBot(cfg, store, market, model, FakeRealTrader())  # type: ignore[arg-type]
            result = bot.catch_up(now_ms=candles[-1].close_time_ms + 2_000, progress_every=0)

            self.assertEqual(result.status, "complete")
            self.assertEqual(result.processed_bars, 3)
            self.assertEqual(result.start_candle_open_time, candles[3].open_time)
            self.assertEqual(result.end_candle_open_time, candles[5].open_time)
            decisions = store.recent_rows("model_decisions", 10)
            self.assertEqual(len(decisions), 4)
            self.assertEqual(decisions[-1]["candle_open_time"], candles[5].open_time)
            self.assertEqual(len(store.recent_rows("account_snapshots", 10)), 3)
            self.assertEqual(market.range_calls[0][2], interval_ms)
            store.close()


class RetrainFrequencyTest(unittest.TestCase):
    def test_supported_frequencies(self) -> None:
        self.assertEqual(parse_retrain_frequency("10h"), (10, "h"))
        self.assertEqual(parse_retrain_frequency("3d"), (3, "d"))
        self.assertEqual(parse_retrain_frequency("1w"), (1, "w"))
        self.assertEqual(parse_retrain_frequency("1m"), (1, "m"))
        self.assertEqual(parse_retrain_frequency("2months"), (2, "m"))

    def test_rejects_invalid_frequency(self) -> None:
        for value in ["", "0d", "15min", "abc"]:
            with self.subTest(value=value):
                with self.assertRaises(ValueError):
                    parse_retrain_frequency(value)


class RetrainCacheTest(unittest.TestCase):
    def test_live_cache_path_uses_symbol_and_interval(self) -> None:
        cfg = load_config()
        object.__setattr__(cfg, "symbol", "SOLUSDT")
        object.__setattr__(cfg, "interval", "3m")
        object.__setattr__(cfg, "retrain_cache_dir", "/app/state/downloads")
        self.assertEqual(
            live_candle_cache_path(cfg),
            Path("/app/state/downloads/binance/SOLUSDT/3m/cache.parquet"),
        )

    def test_floor_to_interval(self) -> None:
        ts = pd.Timestamp("2026-06-02T22:05:00Z").to_pydatetime()
        self.assertEqual(floor_to_interval(ts, "3m"), pd.Timestamp("2026-06-02T22:03:00Z").to_pydatetime())
        self.assertEqual(floor_to_interval(ts, "15m"), pd.Timestamp("2026-06-02T22:00:00Z").to_pydatetime())

    def test_seed_cache_rejects_wrong_interval(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            run_dir = root / "training_runs" / "20260602T000000Z"
            run_dir.mkdir(parents=True)
            cache_path = root / "downloads" / "binance" / "SOLUSDT" / "3m" / "cache.parquet"
            frame = pd.DataFrame(
                {
                    "open_time": pd.date_range("2026-01-01", periods=4, freq="15min", tz="UTC"),
                    "symbol": ["SOLUSDT"] * 4,
                    "open": [1.0, 1.1, 1.2, 1.3],
                    "high": [1.0, 1.1, 1.2, 1.3],
                    "low": [1.0, 1.1, 1.2, 1.3],
                    "close": [1.0, 1.1, 1.2, 1.3],
                    "volume": [10.0, 11.0, 12.0, 13.0],
                }
            )
            frame.to_parquet(run_dir / "candles.parquet", index=False)
            seed_live_candle_cache(cache_path, root / "training_runs", "SOLUSDT", "3m")
            self.assertFalse(cache_path.exists())

    def test_seed_cache_copies_matching_interval(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            run_dir = root / "training_runs" / "20260602T000000Z"
            run_dir.mkdir(parents=True)
            cache_path = root / "downloads" / "binance" / "SOLUSDT" / "3m" / "cache.parquet"
            frame = pd.DataFrame(
                {
                    "open_time": pd.date_range("2026-01-01", periods=4, freq="3min", tz="UTC"),
                    "symbol": ["SOLUSDT"] * 4,
                    "open": [1.0, 1.1, 1.2, 1.3],
                    "high": [1.0, 1.1, 1.2, 1.3],
                    "low": [1.0, 1.1, 1.2, 1.3],
                    "close": [1.0, 1.1, 1.2, 1.3],
                    "volume": [10.0, 11.0, 12.0, 13.0],
                }
            )
            frame.to_parquet(run_dir / "candles.parquet", index=False)
            seed_live_candle_cache(cache_path, root / "training_runs", "SOLUSDT", "3m")
            self.assertTrue(cache_path.exists())
            self.assertEqual(len(pd.read_parquet(cache_path)), 4)


if __name__ == "__main__":
    unittest.main()
