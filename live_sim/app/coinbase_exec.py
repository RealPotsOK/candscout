"""Coinbase Advanced Trade spot execution for tightly capped live experiments."""

from __future__ import annotations

import json
import time
import uuid
from dataclasses import asdict, dataclass
from typing import Any

from .config import Config
from .store import Store


class RealTradingError(RuntimeError):
    """Raised when real-money execution cannot safely continue."""


@dataclass(frozen=True)
class RealOrderResult:
    status: str
    product_id: str
    side: str
    client_order_id: str
    coinbase_order_id: str | None
    requested_usd: float
    requested_sol: float
    filled_usd: float
    filled_sol: float
    average_price: float | None
    fee_usd: float
    raw_response: dict[str, Any]
    error: str | None = None


def _response_to_dict(value: Any) -> dict[str, Any]:
    if value is None:
        return {}
    if isinstance(value, dict):
        return value
    if hasattr(value, "to_dict"):
        out = value.to_dict()
        return out if isinstance(out, dict) else {"value": out}
    if hasattr(value, "__dict__"):
        return dict(value.__dict__)
    return {"value": str(value)}


def _float_or_zero(value: Any) -> float:
    if value is None or value == "":
        return 0.0
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _extract_nested(data: dict[str, Any], *paths: tuple[str, ...]) -> Any:
    for path in paths:
        cur: Any = data
        for key in path:
            if not isinstance(cur, dict) or key not in cur:
                cur = None
                break
            cur = cur[key]
        if cur not in (None, ""):
            return cur
    return None


def _format_usd(value: float) -> str:
    return f"{value:.2f}"


def _format_sol(value: float) -> str:
    return f"{value:.8f}".rstrip("0").rstrip(".")


def _client_order_id(prefix: str, product_id: str) -> str:
    product = product_id.lower().replace("-", "")
    return f"cryptopred-{prefix}-{product}-{uuid.uuid4().hex[:18]}"


class CoinbaseSpotExecutor:
    def __init__(self, cfg: Config) -> None:
        self.cfg = cfg
        try:
            from coinbase.rest import RESTClient  # type: ignore
        except ImportError as exc:  # pragma: no cover - covered by integration env.
            raise RealTradingError(
                "coinbase-advanced-py is not installed. Run live_sim Docker build or install the dependency."
            ) from exc
        self.client = RESTClient(
            api_key=cfg.coinbase_api_key,
            api_secret=cfg.coinbase_api_secret,
            timeout=cfg.coinbase_timeout,
        )

    def health_check(self) -> dict[str, Any]:
        product = _response_to_dict(self.client.get_product(self.cfg.coinbase_product_id))
        balances = self.available_balances()
        return {
            "ok": True,
            "product_id": self.cfg.coinbase_product_id,
            "product": {
                "product_id": product.get("product_id") or product.get("product", {}).get("product_id"),
                "price": product.get("price") or product.get("product", {}).get("price"),
                "status": product.get("status") or product.get("product", {}).get("status"),
            },
            "balances": {
                "USD": balances.get("USD", 0.0),
                "SOL": balances.get("SOL", 0.0),
            },
        }

    def available_balances(self) -> dict[str, float]:
        data = _response_to_dict(self.client.get_accounts())
        accounts = data.get("accounts", [])
        if not isinstance(accounts, list) and hasattr(accounts, "__iter__"):
            accounts = list(accounts)
        out: dict[str, float] = {}
        for raw in accounts if isinstance(accounts, list) else []:
            account = _response_to_dict(raw)
            currency = str(account.get("currency") or account.get("asset") or "").upper()
            if not currency:
                continue
            balance = account.get("available_balance") or account.get("balance") or {}
            if not isinstance(balance, dict):
                balance = _response_to_dict(balance)
            out[currency] = out.get(currency, 0.0) + _float_or_zero(balance.get("value"))
        return out

    def market_buy_quote(self, quote_usd: float) -> RealOrderResult:
        client_order_id = _client_order_id("buy", self.cfg.coinbase_product_id)
        response = self.client.market_order_buy(
            client_order_id=client_order_id,
            product_id=self.cfg.coinbase_product_id,
            quote_size=_format_usd(quote_usd),
        )
        create_data = _response_to_dict(response)
        return self._confirm_order(
            create_data=create_data,
            side="BUY",
            client_order_id=client_order_id,
            requested_usd=quote_usd,
            requested_sol=0.0,
        )

    def market_sell_base(self, base_sol: float) -> RealOrderResult:
        client_order_id = _client_order_id("sell", self.cfg.coinbase_product_id)
        response = self.client.market_order_sell(
            client_order_id=client_order_id,
            product_id=self.cfg.coinbase_product_id,
            base_size=_format_sol(base_sol),
        )
        create_data = _response_to_dict(response)
        return self._confirm_order(
            create_data=create_data,
            side="SELL",
            client_order_id=client_order_id,
            requested_usd=0.0,
            requested_sol=base_sol,
        )

    def _confirm_order(
        self,
        *,
        create_data: dict[str, Any],
        side: str,
        client_order_id: str,
        requested_usd: float,
        requested_sol: float,
    ) -> RealOrderResult:
        if create_data.get("success") is False:
            error = create_data.get("error_response") or create_data
            return RealOrderResult(
                status="rejected",
                product_id=self.cfg.coinbase_product_id,
                side=side,
                client_order_id=client_order_id,
                coinbase_order_id=None,
                requested_usd=requested_usd,
                requested_sol=requested_sol,
                filled_usd=0.0,
                filled_sol=0.0,
                average_price=None,
                fee_usd=0.0,
                raw_response=create_data,
                error=json.dumps(error, default=str),
            )

        order_id = _extract_nested(
            create_data,
            ("success_response", "order_id"),
            ("order_id",),
            ("order", "order_id"),
        )
        order_data = create_data
        if order_id:
            for _ in range(max(0, self.cfg.real_order_status_polls)):
                time.sleep(self.cfg.real_order_status_delay_seconds)
                order_data = self._get_order(str(order_id))
                parsed = self._parse_order(
                    data=order_data,
                    side=side,
                    client_order_id=client_order_id,
                    requested_usd=requested_usd,
                    requested_sol=requested_sol,
                    fallback_order_id=str(order_id),
                )
                if parsed.status in {"filled", "partial", "rejected", "cancelled"}:
                    return parsed

        parsed = self._parse_order(
            data=order_data,
            side=side,
            client_order_id=client_order_id,
            requested_usd=requested_usd,
            requested_sol=requested_sol,
            fallback_order_id=str(order_id) if order_id else None,
        )
        if parsed.filled_sol > 0.0 or parsed.filled_usd > 0.0 or parsed.status in {"rejected", "cancelled"}:
            return parsed
        return RealOrderResult(
            status="unconfirmed",
            product_id=self.cfg.coinbase_product_id,
            side=side,
            client_order_id=client_order_id,
            coinbase_order_id=str(order_id) if order_id else None,
            requested_usd=requested_usd,
            requested_sol=requested_sol,
            filled_usd=0.0,
            filled_sol=0.0,
            average_price=None,
            fee_usd=0.0,
            raw_response=order_data,
            error="Order submitted but fill could not be confirmed; real trading was disarmed.",
        )

    def _get_order(self, order_id: str) -> dict[str, Any]:
        try:
            return _response_to_dict(self.client.get_order(order_id=order_id))
        except TypeError:
            return _response_to_dict(self.client.get_order(order_id))

    def _parse_order(
        self,
        *,
        data: dict[str, Any],
        side: str,
        client_order_id: str,
        requested_usd: float,
        requested_sol: float,
        fallback_order_id: str | None,
    ) -> RealOrderResult:
        order = data.get("order") if isinstance(data.get("order"), dict) else data
        status_raw = str(order.get("status") or "").upper()
        completion_pct = _float_or_zero(order.get("completion_percentage"))
        filled_sol = _float_or_zero(order.get("filled_size"))
        filled_usd = _float_or_zero(order.get("filled_value") or order.get("total_value_after_fees"))
        fee_usd = _float_or_zero(order.get("total_fees") or order.get("fee"))
        average_price = _float_or_zero(order.get("average_filled_price")) or None
        order_id = str(order.get("order_id") or fallback_order_id or "")
        if status_raw in {"FILLED", "DONE", "SETTLED"} or completion_pct >= 99.999:
            status = "filled"
        elif filled_sol > 0.0 or filled_usd > 0.0:
            status = "partial"
        elif status_raw in {"CANCELLED", "CANCELED", "EXPIRED"}:
            status = "cancelled"
        elif status_raw in {"REJECTED", "FAILED"} or order.get("reject_reason"):
            status = "rejected"
        elif status_raw:
            status = status_raw.lower()
        else:
            status = "submitted"
        error = None
        if status in {"rejected", "cancelled"}:
            error = str(order.get("reject_message") or order.get("cancel_message") or order.get("reject_reason") or status)
        return RealOrderResult(
            status=status,
            product_id=self.cfg.coinbase_product_id,
            side=side,
            client_order_id=str(order.get("client_order_id") or client_order_id),
            coinbase_order_id=order_id or None,
            requested_usd=requested_usd,
            requested_sol=requested_sol,
            filled_usd=filled_usd,
            filled_sol=filled_sol,
            average_price=average_price,
            fee_usd=fee_usd,
            raw_response=data,
            error=error,
        )


class RealTradeService:
    def __init__(self, cfg: Config, store: Store, executor: CoinbaseSpotExecutor | None = None) -> None:
        self.cfg = cfg
        self.store = store
        self.executor = executor

    @property
    def configured(self) -> bool:
        return self.cfg.execution_mode == "coinbase_live"

    @property
    def enabled(self) -> bool:
        return self.configured and self.cfg.real_trading_enabled

    def public_status(self, *, refresh: bool = False) -> dict[str, Any]:
        state = self.store.real_state()
        payload: dict[str, Any] = {
            "configured": self.configured,
            "enabled": self.enabled,
            "armed": bool(state.get("armed")),
            "product_id": self.cfg.coinbase_product_id,
            "max_total_usd": self.cfg.real_max_total_usd,
            "max_order_usd": self.cfg.real_max_order_usd,
            "min_order_usd": self.cfg.real_min_order_usd,
            "bot_sol_qty": state.get("bot_sol_qty", 0.0),
            "bot_cost_usd": state.get("bot_cost_usd", 0.0),
            "realized_pnl_usd": state.get("realized_pnl_usd", 0.0),
            "total_fees_usd": state.get("total_fees_usd", 0.0),
            "last_error": state.get("last_error"),
            "latest_order": self.store.latest_real_order(),
            "credentials_present": bool(self.cfg.coinbase_api_key and self.cfg.coinbase_api_secret),
        }
        if refresh and self.executor is not None:
            try:
                payload["exchange"] = self.executor.health_check()
            except Exception as exc:  # noqa: BLE001 - API status should report failures.
                payload["exchange"] = {"ok": False, "error": str(exc)}
        return payload

    def preflight(self) -> dict[str, Any]:
        self._require_executor()
        assert self.executor is not None
        return self.executor.health_check()

    def arm(self, *, token: str, confirmation: str, ts: str) -> dict[str, Any]:
        self._require_executor()
        if token != self.cfg.real_arm_token:
            raise RealTradingError("Invalid REAL_ARM_TOKEN")
        expected = self.arm_confirmation_text()
        if confirmation.strip() != expected:
            raise RealTradingError(f"Confirmation text must exactly equal: {expected}")
        self.preflight()
        self.store.set_real_armed(armed=True, ts=ts, error=None)
        self.store.insert_event(ts, "warning", "REAL TRADING ARMED")
        return self.public_status(refresh=True)

    def disarm(self, *, ts: str, reason: str = "manual_disarm") -> dict[str, Any]:
        self.store.set_real_armed(armed=False, ts=ts, error=None)
        self.store.insert_event(ts, "info", f"real trading disarmed: {reason}")
        return self.public_status(refresh=False)

    def execute_buy(
        self,
        *,
        ts: str,
        candle_open_time: str,
        planned_usd: float,
        reason: str,
    ) -> None:
        if not self._can_execute(ts=ts, action="buy", candle_open_time=candle_open_time):
            return
        state = self.store.real_state()
        remaining_cap = self.cfg.real_max_total_usd - float(state["bot_cost_usd"])
        requested_usd = min(float(planned_usd), self.cfg.real_max_order_usd, remaining_cap)
        if requested_usd < self.cfg.real_min_order_usd:
            self._record_skip(ts, candle_open_time, "buy", "BUY", requested_usd, 0.0, "below_real_min_or_cap")
            return
        assert self.executor is not None
        try:
            balances = self.executor.available_balances()
            requested_usd = min(requested_usd, balances.get("USD", 0.0))
            if requested_usd < self.cfg.real_min_order_usd:
                self._record_skip(ts, candle_open_time, "buy", "BUY", requested_usd, 0.0, "insufficient_usd_balance")
                return
            result = self.executor.market_buy_quote(requested_usd)
            self._record_result(ts, candle_open_time, "buy", result, reason)
            if result.status in {"filled", "partial"} and result.filled_sol > 0.0:
                self._apply_buy_fill(ts, result, requested_usd)
            elif result.status in {"unconfirmed", "rejected", "cancelled"}:
                self._disarm_on_uncertain_order(ts, result.error or f"{result.status} buy")
        except Exception as exc:  # noqa: BLE001 - persist and disarm on live errors.
            self._record_error(ts, candle_open_time, "buy", "BUY", requested_usd, 0.0, str(exc))
            self._disarm_on_uncertain_order(ts, str(exc))

    def execute_sell_all(
        self,
        *,
        ts: str,
        candle_open_time: str,
        reason: str,
        require_armed: bool = True,
    ) -> None:
        if not self._can_execute(
            ts=ts,
            action="sell",
            candle_open_time=candle_open_time,
            require_armed=require_armed,
        ):
            return
        state = self.store.real_state()
        bot_sol_qty = float(state["bot_sol_qty"])
        if bot_sol_qty <= 0.0:
            self._record_skip(ts, candle_open_time, "sell", "SELL", 0.0, 0.0, "no_bot_tracked_sol")
            return
        assert self.executor is not None
        try:
            result = self.executor.market_sell_base(bot_sol_qty)
            self._record_result(ts, candle_open_time, "sell", result, reason)
            if result.status in {"filled", "partial"} and result.filled_sol > 0.0:
                self._apply_sell_fill(ts, result)
            elif result.status in {"unconfirmed", "rejected", "cancelled"}:
                self._disarm_on_uncertain_order(ts, result.error or f"{result.status} sell")
        except Exception as exc:  # noqa: BLE001 - persist and disarm on live errors.
            self._record_error(ts, candle_open_time, "sell", "SELL", 0.0, bot_sol_qty, str(exc))
            self._disarm_on_uncertain_order(ts, str(exc))

    def flatten(self, *, token: str, confirmation: str, ts: str) -> dict[str, Any]:
        self._require_executor()
        if token != self.cfg.real_arm_token:
            raise RealTradingError("Invalid REAL_ARM_TOKEN")
        expected = self.flatten_confirmation_text()
        if confirmation.strip() != expected:
            raise RealTradingError(f"Confirmation text must exactly equal: {expected}")
        self.execute_sell_all(ts=ts, candle_open_time="", reason="manual_flatten", require_armed=False)
        self.disarm(ts=ts, reason="manual_flatten")
        return self.public_status(refresh=True)

    def arm_confirmation_text(self) -> str:
        cap = f"{self.cfg.real_max_total_usd:g}"
        return f"ARM REAL TRADING {self.cfg.coinbase_product_id} MAX {cap}"

    def flatten_confirmation_text(self) -> str:
        return f"FLATTEN REAL {self.cfg.coinbase_product_id}"

    def _require_executor(self) -> None:
        if not self.configured:
            raise RealTradingError("Set EXECUTION_MODE=coinbase_live first")
        if not self.enabled:
            raise RealTradingError("Set REAL_TRADING_ENABLED=true first")
        if self.executor is None:
            self.executor = CoinbaseSpotExecutor(self.cfg)

    def _can_execute(
        self,
        *,
        ts: str,
        action: str,
        candle_open_time: str,
        require_armed: bool = True,
    ) -> bool:
        if not self.enabled:
            return False
        state = self.store.real_state()
        if require_armed and not bool(state.get("armed")):
            self._record_skip(ts, candle_open_time, action, action.upper(), 0.0, 0.0, "not_armed")
            return False
        self._require_executor()
        return True

    def _apply_buy_fill(self, ts: str, result: RealOrderResult, requested_usd: float) -> None:
        state = self.store.real_state()
        filled_cost = result.filled_usd + result.fee_usd if result.filled_usd > 0 else requested_usd
        self.store.update_real_position(
            bot_sol_qty=float(state["bot_sol_qty"]) + result.filled_sol,
            bot_cost_usd=float(state["bot_cost_usd"]) + min(filled_cost, requested_usd),
            realized_pnl_usd=float(state["realized_pnl_usd"]),
            total_fees_usd=float(state["total_fees_usd"]) + result.fee_usd,
            ts=ts,
            error=None,
        )

    def _apply_sell_fill(self, ts: str, result: RealOrderResult) -> None:
        state = self.store.real_state()
        old_qty = float(state["bot_sol_qty"])
        old_cost = float(state["bot_cost_usd"])
        fill_qty = min(result.filled_sol, old_qty)
        if old_qty <= 0.0 or fill_qty <= 0.0:
            return
        sold_fraction = min(1.0, fill_qty / old_qty)
        cost_reduction = old_cost * sold_fraction
        new_qty = max(0.0, old_qty - fill_qty)
        new_cost = 0.0 if new_qty < 1e-10 else max(0.0, old_cost - cost_reduction)
        realized = result.filled_usd - result.fee_usd - cost_reduction
        self.store.update_real_position(
            bot_sol_qty=new_qty,
            bot_cost_usd=new_cost,
            realized_pnl_usd=float(state["realized_pnl_usd"]) + realized,
            total_fees_usd=float(state["total_fees_usd"]) + result.fee_usd,
            ts=ts,
            error=None,
        )

    def _record_result(self, ts: str, candle_open_time: str, action: str, result: RealOrderResult, reason: str) -> None:
        payload = asdict(result)
        self.store.insert_real_order(
            {
                **payload,
                "ts": ts,
                "candle_open_time": candle_open_time,
                "action": action,
                "reason": reason,
                "raw_response": json.dumps(result.raw_response, default=str),
            }
        )

    def _record_skip(
        self,
        ts: str,
        candle_open_time: str,
        action: str,
        side: str,
        requested_usd: float,
        requested_sol: float,
        reason: str,
    ) -> None:
        self.store.insert_real_order(
            {
                "ts": ts,
                "candle_open_time": candle_open_time,
                "action": action,
                "status": "skipped",
                "product_id": self.cfg.coinbase_product_id,
                "side": side,
                "requested_usd": requested_usd,
                "requested_sol": requested_sol,
                "reason": reason,
            }
        )

    def _record_error(
        self,
        ts: str,
        candle_open_time: str,
        action: str,
        side: str,
        requested_usd: float,
        requested_sol: float,
        error: str,
    ) -> None:
        self.store.insert_real_order(
            {
                "ts": ts,
                "candle_open_time": candle_open_time,
                "action": action,
                "status": "failed",
                "product_id": self.cfg.coinbase_product_id,
                "side": side,
                "requested_usd": requested_usd,
                "requested_sol": requested_sol,
                "error": error,
            }
        )

    def _disarm_on_uncertain_order(self, ts: str, error: str) -> None:
        self.store.set_real_armed(armed=False, ts=ts, error=error)
        self.store.insert_event(ts, "error", f"real trading disarmed: {error}")
