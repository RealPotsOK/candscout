"""Live paper-trading bot orchestration."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

from .coinbase_exec import RealTradeService
from .config import Config
from .market import BinanceClient, BookTicker, Candle, ms_from_iso, synthetic_book_ticker
from .model_runner import LiveModel
from .store import Store
from .trading import (
    bars_between,
    calculate_buy,
    calculate_sell,
    calculate_short_cover,
    calculate_short_open,
    exit_reason,
    short_exit_reason,
)


@dataclass(frozen=True)
class StepResult:
    ts: str
    action: str
    reason: str
    prob_up: float | None
    candle_open_time: str | None
    equity: float | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "ts": self.ts,
            "action": self.action,
            "reason": self.reason,
            "prob_up": self.prob_up,
            "candle_open_time": self.candle_open_time,
            "equity": self.equity,
        }


@dataclass(frozen=True)
class CatchUpResult:
    status: str
    reason: str
    start_candle_open_time: str | None
    end_candle_open_time: str | None
    processed_bars: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "reason": self.reason,
            "start_candle_open_time": self.start_candle_open_time,
            "end_candle_open_time": self.end_candle_open_time,
            "processed_bars": self.processed_bars,
        }


class LivePaperBot:
    def __init__(
        self,
        cfg: Config,
        store: Store,
        market: BinanceClient,
        model: LiveModel,
        real_trader: RealTradeService | None = None,
    ) -> None:
        self.cfg = cfg
        self.store = store
        self.market = market
        self.model = model
        self.real_trader = real_trader

    def step(self) -> StepResult:
        self.model.maybe_reload()
        candles = self.market.fetch_completed_klines(
            self.cfg.symbol,
            self.cfg.interval,
            self.model.required_candles(self.cfg.kline_limit_buffer),
        )
        if len(candles) < self.model.lookback + 1:
            raise RuntimeError(
                f"Need {self.model.lookback + 1} completed candles, Binance returned {len(candles)}"
            )
        ticker = self.market.fetch_book_ticker(self.cfg.symbol)
        return self.process_market_data(candles, ticker, decision_ts=now_iso())

    def process_market_data(
        self,
        candles: list[Candle],
        ticker: BookTicker,
        *,
        decision_ts: str,
        allow_real_execution: bool = True,
    ) -> StepResult:
        """Run one completed candle through the normal live trading path."""
        if not candles:
            raise ValueError("At least one completed candle is required")
        self.store.insert_candles(candles[-(self.model.lookback + 2) :])
        self.store.insert_ticker(ticker)

        prob_up, last_candle = self.model.predict(candles)
        candle_open_time = last_candle.open_time
        if self.store.decision_exists(candle_open_time):
            snapshot = self._snapshot(decision_ts, ticker)
            self.store.insert_account_snapshot(snapshot)
            return StepResult(
                ts=decision_ts,
                action="skip",
                reason="duplicate_candle",
                prob_up=prob_up,
                candle_open_time=candle_open_time,
                equity=snapshot["equity"],
            )

        decision = self._trade_decision(
            decision_ts,
            prob_up,
            last_candle,
            ticker,
            allow_real_execution=allow_real_execution,
        )
        self.store.insert_decision(decision)
        self.store.insert_account_snapshot(self._snapshot(decision_ts, ticker))
        return StepResult(
            ts=decision_ts,
            action=decision["action"],
            reason=decision["reason"],
            prob_up=prob_up,
            candle_open_time=candle_open_time,
            equity=decision["equity"],
        )

    def catch_up(
        self,
        *,
        now_ms: int | None = None,
        progress_every: int = 100,
    ) -> CatchUpResult:
        """Replay every completed candle after the last persisted decision."""
        latest = self.store.latest_decision()
        if latest is None:
            return CatchUpResult("skipped", "no_prior_decision", None, None, 0)

        self.model.maybe_reload()
        interval_ms = self.cfg.interval_seconds * 1000
        last_open_ms = ms_from_iso(str(latest["candle_open_time"]))
        first_missing_ms = last_open_ms + interval_ms
        current_ms = int(datetime.now(timezone.utc).timestamp() * 1000) if now_ms is None else int(now_ms)
        completed_cutoff_ms = current_ms - 1000
        if first_missing_ms + interval_ms - 1 > completed_cutoff_ms:
            return CatchUpResult("complete", "already_current", None, None, 0)

        required_candles = self.model.required_candles(self.cfg.kline_limit_buffer)
        fetch_start_ms = max(0, first_missing_ms - (required_candles - 1) * interval_ms)
        candles = self.market.fetch_klines_range(
            self.cfg.symbol,
            self.cfg.interval,
            fetch_start_ms,
            current_ms,
            interval_ms,
        )
        completed = [candle for candle in candles if candle.close_time_ms <= completed_cutoff_ms]
        target_indexes = [
            index for index, candle in enumerate(completed) if candle.open_time_ms >= first_missing_ms
        ]
        if not target_indexes:
            return CatchUpResult("complete", "no_completed_gap", None, None, 0)
        if self.cfg.catchup_max_bars and len(target_indexes) > self.cfg.catchup_max_bars:
            raise RuntimeError(
                f"Catch-up requires {len(target_indexes)} bars, exceeding "
                f"CATCHUP_MAX_BARS={self.cfg.catchup_max_bars}"
            )

        processed = 0
        first_processed: str | None = None
        last_processed: str | None = None
        for target_index in target_indexes:
            history_start = max(0, target_index - required_candles + 1)
            history = completed[history_start : target_index + 1]
            target = completed[target_index]
            ticker = synthetic_book_ticker(target, self.cfg.catchup_spread_pct)
            result = self.process_market_data(
                history,
                ticker,
                decision_ts=target.close_time,
                allow_real_execution=False,
            )
            if result.reason == "duplicate_candle":
                continue
            processed += 1
            first_processed = first_processed or target.open_time
            last_processed = target.open_time
            if progress_every > 0 and processed % progress_every == 0:
                print(
                    f"catchup_progress processed={processed}/{len(target_indexes)} "
                    f"candle={target.open_time}",
                    flush=True,
                )

        return CatchUpResult(
            "complete",
            "replayed_missing_candles",
            first_processed,
            last_processed,
            processed,
        )

    def _trade_decision(
        self,
        ts: str,
        prob_up: float,
        candle: Candle,
        ticker: BookTicker,
        *,
        allow_real_execution: bool,
    ) -> dict[str, Any]:
        account = self.store.account_state()
        position = self.store.open_position()
        cash = float(account["cash"])
        realized_pnl = float(account["realized_pnl"])
        total_fees = float(account["total_fees"])
        action = "cash"
        reason = "neutral_probability"
        closed_side = ""

        if position is not None:
            side = str(position.get("side", "long"))
            bars_held = bars_between(
                str(position["entry_candle_open_time"]),
                candle.open_time,
                self.cfg.interval_seconds,
            )
            if side == "short":
                reason_to_exit = short_exit_reason(
                    prob_up=prob_up,
                    exit_threshold=self.cfg.short_exit_threshold,
                    ask=ticker.ask,
                    entry_price=float(position["entry_price"]),
                    bars_held=bars_held,
                    max_hold_bars=self.cfg.max_hold_bars,
                    stop_loss=self.cfg.stop_loss,
                    take_profit=self.cfg.take_profit,
                )
            else:
                reason_to_exit = exit_reason(
                    prob_up=prob_up,
                    exit_threshold=self.cfg.exit_threshold,
                    bid=ticker.bid,
                    entry_price=float(position["entry_price"]),
                    bars_held=bars_held,
                    max_hold_bars=self.cfg.max_hold_bars,
                    stop_loss=self.cfg.stop_loss,
                    take_profit=self.cfg.take_profit,
                )
            if reason_to_exit is None:
                action = "hold"
                reason = f"hold_{side}"
            else:
                if side == "short":
                    close_result = calculate_short_cover(
                        cash=cash,
                        ask=ticker.ask,
                        quantity=float(position["quantity"]),
                        investment=float(position["investment"]),
                        entry_price=float(position["entry_price"]),
                        entry_fee=float(position["entry_fee"]),
                        fee=self.cfg.fee,
                        slippage=self.cfg.slippage,
                        borrow_fee_rate=self.cfg.borrow_fee,
                        bars_held=bars_held,
                    )
                    borrow_fee = close_result.borrow_fee
                    action = "short_cover"
                else:
                    close_result = calculate_sell(
                        cash=cash,
                        bid=ticker.bid,
                        quantity=float(position["quantity"]),
                        investment=float(position["investment"]),
                        entry_fee=float(position["entry_fee"]),
                        fee=self.cfg.fee,
                        slippage=self.cfg.slippage,
                    )
                    borrow_fee = 0.0
                    action = "long_sell"
                cash = close_result.cash_after
                realized_pnl += close_result.net_profit
                total_fees += close_result.exit_fee + borrow_fee
                self.store.insert_trade(
                    {
                        "entry_time": position["entry_time"],
                        "exit_time": ts,
                        "entry_price": position["entry_price"],
                        "exit_price": close_result.exit_price,
                        "quantity": position["quantity"],
                        "investment": position["investment"],
                        "gross_exit_value": close_result.gross_exit_value,
                        "entry_fee": position["entry_fee"],
                        "exit_fee": close_result.exit_fee,
                        "borrow_fee": borrow_fee,
                        "net_profit": close_result.net_profit,
                        "gross_return": close_result.gross_return,
                        "bars_held": bars_held,
                        "exit_reason": reason_to_exit,
                        "side": side,
                    }
                )
                self.store.clear_open_position()
                self.store.update_account_state(
                    cash=cash,
                    realized_pnl=realized_pnl,
                    total_fees=total_fees,
                    updated_at=ts,
                )
                if allow_real_execution and side == "long":
                    self._execute_real_sell(ts=ts, candle_open_time=candle.open_time, reason=reason_to_exit)
                reason = reason_to_exit
                closed_side = side
                position = None

        can_enter = position is None and (not closed_side or self.cfg.allow_flip_position)
        long_signal = self.cfg.trade_mode in {"long_only", "long_short"} and prob_up >= self.cfg.entry_threshold
        short_signal = self.cfg.trade_mode in {"short_only", "long_short"} and prob_up <= self.cfg.short_entry_threshold
        if closed_side and self.cfg.allow_flip_position:
            long_signal = long_signal and closed_side == "short"
            short_signal = short_signal and closed_side == "long"
        if can_enter and (long_signal or short_signal):
            next_side = "long" if long_signal else "short"
            if next_side == "long":
                buy = calculate_buy(
                    cash=cash,
                    ask=ticker.ask,
                    max_invest_expr=self.cfg.max_invest,
                    min_invest=self.cfg.min_invest,
                    fee=self.cfg.fee,
                    slippage=self.cfg.slippage,
                    prob_up=prob_up,
                    entry_threshold=self.cfg.entry_threshold,
                    confidence_multiplier=self.cfg.confidence_multiplier,
                )
                if buy is None:
                    action = "skip"
                    reason = "insufficient_cash_for_min_invest"
                else:
                    cash = buy.cash_after
                    total_fees += buy.entry_fee
                    self.store.update_account_state(
                        cash=cash,
                        realized_pnl=realized_pnl,
                        total_fees=total_fees,
                        updated_at=ts,
                    )
                    self.store.set_open_position(
                        {
                            "entry_time": ts,
                            "entry_candle_open_time": candle.open_time,
                            "entry_price": buy.entry_price,
                            "quantity": buy.quantity,
                            "investment": buy.investment,
                            "entry_fee": buy.entry_fee,
                            "entry_prob_up": prob_up,
                            "entry_bid": ticker.bid,
                            "entry_ask": ticker.ask,
                            "side": "long",
                        }
                    )
                    action = "long_buy" if not closed_side else "flip_long"
                    reason = "long_entry_threshold"
                    if allow_real_execution:
                        self._execute_real_buy(
                            ts=ts,
                            candle_open_time=candle.open_time,
                            planned_usd=buy.investment,
                            reason=reason,
                        )
            else:
                short = calculate_short_open(
                    cash=cash,
                    bid=ticker.bid,
                    max_invest_expr=self.cfg.max_short_invest,
                    min_invest=self.cfg.min_invest,
                    fee=self.cfg.fee,
                    slippage=self.cfg.slippage,
                    prob_up=prob_up,
                    entry_threshold=self.cfg.short_entry_threshold,
                    confidence_multiplier=self.cfg.confidence_multiplier,
                )
                if short is None:
                    action = "skip"
                    reason = "insufficient_cash_for_min_short"
                else:
                    cash = short.cash_after
                    total_fees += short.entry_fee
                    self.store.update_account_state(
                        cash=cash,
                        realized_pnl=realized_pnl,
                        total_fees=total_fees,
                        updated_at=ts,
                    )
                    self.store.set_open_position(
                        {
                            "entry_time": ts,
                            "entry_candle_open_time": candle.open_time,
                            "entry_price": short.entry_price,
                            "quantity": short.quantity,
                            "investment": short.investment,
                            "entry_fee": short.entry_fee,
                            "entry_prob_up": prob_up,
                            "entry_bid": ticker.bid,
                            "entry_ask": ticker.ask,
                            "side": "short",
                        }
                    )
                    action = "short_open" if not closed_side else "flip_short"
                    reason = "short_entry_threshold"

        snapshot = self._snapshot(ts, ticker)
        return {
            "ts": ts,
            "candle_open_time": candle.open_time,
            "prob_up": prob_up,
            "action": action,
            "reason": reason,
            "entry_threshold": self.cfg.entry_threshold,
            "exit_threshold": self.cfg.exit_threshold,
            "cash": snapshot["cash"],
            "sol_qty": snapshot["sol_qty"],
            "equity": snapshot["equity"],
            "bid": ticker.bid,
            "ask": ticker.ask,
            "spread_pct": ticker.spread_pct,
        }

    def _execute_real_buy(self, *, ts: str, candle_open_time: str, planned_usd: float, reason: str) -> None:
        if self.real_trader is None:
            return
        try:
            self.real_trader.execute_buy(
                ts=ts,
                candle_open_time=candle_open_time,
                planned_usd=planned_usd,
                reason=reason,
            )
        except Exception as exc:  # noqa: BLE001 - real execution must fail closed, not stop paper loop.
            self._record_real_execution_error(ts, str(exc))

    def _execute_real_sell(self, *, ts: str, candle_open_time: str, reason: str) -> None:
        if self.real_trader is None:
            return
        try:
            self.real_trader.execute_sell_all(
                ts=ts,
                candle_open_time=candle_open_time,
                reason=reason,
            )
        except Exception as exc:  # noqa: BLE001 - real execution must fail closed, not stop paper loop.
            self._record_real_execution_error(ts, str(exc))

    def _record_real_execution_error(self, ts: str, message: str) -> None:
        self.store.set_real_armed(armed=False, ts=ts, error=message)
        self.store.insert_event(ts, "error", f"real execution failed and was disarmed: {message}")

    def _snapshot(self, ts: str, ticker: BookTicker) -> dict[str, Any]:
        account = self.store.account_state()
        position = self.store.open_position()
        cash = float(account["cash"])
        realized_pnl = float(account["realized_pnl"])
        total_fees = float(account["total_fees"])
        if position is None:
            return {
                "ts": ts,
                "cash": cash,
                "sol_qty": 0.0,
                "equity": cash,
                "position_status": "cash",
                "entry_price": None,
                "unrealized_pnl": 0.0,
                "realized_pnl": realized_pnl,
                "total_fees": total_fees,
                "last_price": ticker.bid,
            }

        quantity = float(position["quantity"])
        side = str(position.get("side", "long"))
        if side == "short":
            bars_held = bars_between(
                str(position["entry_candle_open_time"]),
                ts,
                self.cfg.interval_seconds,
            )
            cover_value = quantity * ticker.ask * (1.0 + self.cfg.slippage)
            cover_fee = cover_value * self.cfg.fee
            borrow_fee = float(position["investment"]) * self.cfg.borrow_fee * bars_held
            gross_pnl = quantity * (float(position["entry_price"]) - ticker.ask * (1.0 + self.cfg.slippage))
            equity = cash + float(position["investment"]) + gross_pnl - cover_fee - borrow_fee
            unrealized_pnl = gross_pnl - float(position["entry_fee"]) - cover_fee - borrow_fee
            display_quantity = -quantity
            last_price = ticker.ask
        else:
            liquidation_value = quantity * ticker.bid * (1.0 - self.cfg.fee)
            equity = cash + liquidation_value
            unrealized_pnl = liquidation_value - float(position["investment"]) - float(position["entry_fee"])
            display_quantity = quantity
            last_price = ticker.bid
        return {
            "ts": ts,
            "cash": cash,
            "sol_qty": display_quantity,
            "equity": equity,
            "position_status": side,
            "entry_price": float(position["entry_price"]),
            "unrealized_pnl": unrealized_pnl,
            "realized_pnl": realized_pnl,
            "total_fees": total_fees,
            "last_price": last_price,
        }


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
