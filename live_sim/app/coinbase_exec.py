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
    execution_source: str = "coinbase"
    transaction_signature: str | None = None
    input_mint: str | None = None
    output_mint: str | None = None
    input_amount_raw: float = 0.0
    expected_output_amount_raw: float = 0.0
    confirmed_output_amount_raw: float = 0.0
    network_fee_lamports: float = 0.0
    priority_fee_lamports: float = 0.0
    slippage_bps: float = 0.0


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
    return f"candscout-{prefix}-{product}-{uuid.uuid4().hex[:18]}"


def _product_base_quote(product_id: str) -> tuple[str, str]:
    parts = product_id.upper().split("-")
    if len(parts) >= 2:
        return parts[0], parts[1]
    return "SOL", "USD"


def _cfg_base_quote(cfg: Config) -> tuple[str, str]:
    return cfg.real_base_asset.upper(), cfg.real_cash_asset.upper()


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
        product = self.product_snapshot()
        balances = self.available_balances()
        base_currency, quote_currency = _product_base_quote(self.cfg.coinbase_product_id)
        return {
            "ok": True,
            "product_id": self.cfg.coinbase_product_id,
            "base_currency": base_currency,
            "quote_currency": quote_currency,
            "product": product,
            "balances": {
                quote_currency: balances.get(quote_currency, 0.0),
                base_currency: balances.get(base_currency, 0.0),
            },
        }

    def product_snapshot(self) -> dict[str, Any]:
        data = _response_to_dict(self.client.get_product(self.cfg.coinbase_product_id))
        product = data.get("product") if isinstance(data.get("product"), dict) else data
        return {
            "product_id": product.get("product_id") or self.cfg.coinbase_product_id,
            "price": _float_or_zero(product.get("price")),
            "status": product.get("status"),
            "base_currency_id": product.get("base_currency_id") or product.get("base_currency"),
            "quote_currency_id": product.get("quote_currency_id") or product.get("quote_currency"),
            "raw": product,
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

    def account_details(self) -> list[dict[str, Any]]:
        data = _response_to_dict(self.client.get_accounts())
        accounts = data.get("accounts", [])
        if not isinstance(accounts, list) and hasattr(accounts, "__iter__"):
            accounts = list(accounts)
        details: list[dict[str, Any]] = []
        for raw in accounts if isinstance(accounts, list) else []:
            account = _response_to_dict(raw)
            currency = str(account.get("currency") or account.get("asset") or "").upper()
            available = account.get("available_balance") or account.get("balance") or {}
            hold = account.get("hold") or account.get("hold_balance") or {}
            if not isinstance(available, dict):
                available = _response_to_dict(available)
            if not isinstance(hold, dict):
                hold = _response_to_dict(hold)
            if currency:
                details.append(
                    {
                        "currency": currency,
                        "available": _float_or_zero(available.get("value")),
                        "hold": _float_or_zero(hold.get("value")),
                        "uuid": account.get("uuid") or account.get("account_id"),
                    }
                )
        return details

    def list_orders_read_only(self, limit: int = 100, *, bot_only: bool = True) -> dict[str, Any]:
        safe_limit = max(1, min(int(limit), 250))
        if not hasattr(self.client, "list_orders"):
            return {
                "ok": False,
                "error": "Installed Coinbase SDK does not expose list_orders(). Bot SQLite orders are still available.",
                "orders": [],
            }
        attempts = [
            {"product_id": self.cfg.coinbase_product_id, "limit": safe_limit},
            {"product_ids": [self.cfg.coinbase_product_id], "limit": safe_limit},
            {"limit": safe_limit},
        ]
        last_error: Exception | None = None
        for kwargs in attempts:
            try:
                response = self.client.list_orders(**kwargs)
                data = _response_to_dict(response)
                return {"ok": True, "orders": self._normalize_orders(data, bot_only=bot_only)[:safe_limit]}
            except TypeError as exc:
                last_error = exc
                continue
            except Exception as exc:  # noqa: BLE001 - expose read-only API errors safely.
                return {"ok": False, "error": str(exc), "orders": []}
        return {"ok": False, "error": str(last_error or "list_orders call failed"), "orders": []}

    def _normalize_orders(self, data: dict[str, Any], *, bot_only: bool) -> list[dict[str, Any]]:
        raw_orders = data.get("orders") or data.get("results") or []
        if not isinstance(raw_orders, list) and hasattr(raw_orders, "__iter__"):
            raw_orders = list(raw_orders)
        orders: list[dict[str, Any]] = []
        for raw in raw_orders if isinstance(raw_orders, list) else []:
            order = _response_to_dict(raw)
            product_id = str(order.get("product_id") or "")
            if product_id and product_id != self.cfg.coinbase_product_id:
                continue
            client_order_id = str(order.get("client_order_id") or "")
            if bot_only and not (client_order_id.startswith("candscout-") or client_order_id.startswith("cryptopred-")):
                continue
            orders.append(
                {
                    "created_time": order.get("created_time") or order.get("created_at") or order.get("time"),
                    "product_id": product_id or self.cfg.coinbase_product_id,
                    "side": order.get("side"),
                    "status": order.get("status"),
                    "client_order_id": client_order_id,
                    "order_id": order.get("order_id"),
                    "filled_size": _float_or_zero(order.get("filled_size")),
                    "filled_value": _float_or_zero(order.get("filled_value") or order.get("total_value_after_fees")),
                    "average_price": _float_or_zero(order.get("average_filled_price")),
                    "total_fees": _float_or_zero(order.get("total_fees") or order.get("fee")),
                }
            )
        return orders

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
        return self.cfg.execution_mode in {"coinbase_live", "solana_jupiter_live"}

    @property
    def product_id(self) -> str:
        return self.cfg.jupiter_product_id if self.cfg.execution_mode == "solana_jupiter_live" else self.cfg.coinbase_product_id

    @property
    def execution_source(self) -> str:
        return "jupiter_solana" if self.cfg.execution_mode == "solana_jupiter_live" else "coinbase"

    def _new_executor(self):
        if self.cfg.execution_mode == "solana_jupiter_live":
            from .jupiter_exec import JupiterSolanaExecutor

            return JupiterSolanaExecutor(self.cfg)
        return CoinbaseSpotExecutor(self.cfg)

    @property
    def enabled(self) -> bool:
        return self.configured and self.cfg.real_trading_enabled

    def public_status(self, *, refresh: bool = False) -> dict[str, Any]:
        state = self.store.real_state()
        base_currency, quote_currency = _cfg_base_quote(self.cfg)
        payload: dict[str, Any] = {
            "configured": self.configured,
            "enabled": self.enabled,
            "armed": bool(state.get("armed")),
            "execution_mode": self.cfg.execution_mode,
            "execution_source": self.execution_source,
            "product_id": self.product_id,
            "base_currency": base_currency,
            "quote_currency": quote_currency,
            "portfolio_mode": self.cfg.real_portfolio_mode,
            "quick_arm_enabled": self.cfg.real_quick_arm_enabled,
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
            "wallet_present": bool(self.cfg.solana_keypair_path),
            "arm_confirmation_text": self.arm_confirmation_text(),
            "flatten_confirmation_text": self.flatten_confirmation_text(),
        }
        if refresh and self.configured:
            try:
                self._ensure_executor_read_only()
                assert self.executor is not None
                payload["exchange"] = self.executor.health_check()
            except Exception as exc:  # noqa: BLE001 - API status should report failures.
                payload["exchange"] = {"ok": False, "error": str(exc)}
        return payload

    def read_only_snapshot(self, *, ts: str, refresh_exchange: bool = True) -> dict[str, Any]:
        state = self.store.real_state()
        base_currency, quote_currency = _cfg_base_quote(self.cfg)
        payload: dict[str, Any] = {
            "ok": True,
            "ts": ts,
            "configured": self.configured,
            "enabled": self.enabled,
            "execution_mode": self.cfg.execution_mode,
            "execution_source": self.execution_source,
            "product_id": self.product_id,
            "base_currency": base_currency,
            "quote_currency": quote_currency,
            "state": state,
            "latest_snapshot": self.store.latest_real_account_snapshot(source=self.execution_source),
        }
        if not refresh_exchange:
            return payload
        if not self.configured:
            payload["ok"] = False
            payload["error"] = "EXECUTION_MODE is not coinbase_live or solana_jupiter_live"
            return payload
        try:
            self._ensure_executor_read_only()
            assert self.executor is not None
            balances = self.executor.available_balances()
            accounts = self.executor.account_details()
            quote_balance = balances.get(quote_currency, 0.0)
            base_balance = balances.get(base_currency, 0.0)
            payload.update(
                {
                    "balances": {quote_currency: quote_balance, base_currency: base_balance},
                    "accounts": accounts,
                }
            )
            product = self.executor.product_snapshot()
            price = float(product.get("price") or 0.0)
            bot_sol_qty = base_balance if self.cfg.real_portfolio_mode == "account_balances" else float(state.get("bot_sol_qty", 0.0))
            bot_cost_usd = quote_balance if self.cfg.real_portfolio_mode == "account_balances" else float(state.get("bot_cost_usd", 0.0))
            bot_market_value = bot_sol_qty * price
            snapshot = {
                "ts": ts,
                "product_id": self.product_id,
                "price": price,
                "usd_balance": quote_balance,
                "sol_balance": base_balance,
                "bot_sol_qty": bot_sol_qty,
                "bot_cost_usd": bot_cost_usd,
                "bot_market_value_usd": bot_market_value,
                "bot_unrealized_pnl_usd": bot_market_value - bot_cost_usd,
                "bot_realized_pnl_usd": float(state.get("realized_pnl_usd", 0.0)),
                "bot_total_fees_usd": float(state.get("total_fees_usd", 0.0)),
                "estimated_account_value_usd": quote_balance + base_balance * price,
                "source": self.execution_source,
            }
            self.store.insert_real_account_snapshot(snapshot)
            payload.update(
                {
                    "product": product,
                    "base_currency": base_currency,
                    "quote_currency": quote_currency,
                    "balances": {quote_currency: quote_balance, base_currency: base_balance},
                    "accounts": accounts,
                    "snapshot": snapshot,
                    "latest_snapshot": snapshot,
                }
            )
        except Exception as exc:  # noqa: BLE001 - dashboard should show safe read-only failure.
            payload["ok"] = False
            payload["error"] = str(exc)
        return payload

    def exchange_orders_read_only(self, *, limit: int = 100, bot_only: bool = True) -> dict[str, Any]:
        if not self.configured:
            return {"ok": False, "error": "EXECUTION_MODE is not coinbase_live or solana_jupiter_live", "orders": []}
        try:
            self._ensure_executor_read_only()
            assert self.executor is not None
            return self.executor.list_orders_read_only(limit=limit, bot_only=bot_only)
        except Exception as exc:  # noqa: BLE001 - expose read-only API errors safely.
            return {"ok": False, "error": str(exc), "orders": []}

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

    def quick_arm(self, *, ts: str) -> dict[str, Any]:
        if not self.cfg.real_quick_arm_enabled:
            raise RealTradingError("Set REAL_QUICK_ARM_ENABLED=true to arm from a one-click dashboard button")
        self._require_executor()
        self.preflight()
        self.store.set_real_armed(armed=True, ts=ts, error=None)
        self.store.insert_event(ts, "warning", "REAL TRADING ARMED BY QUICK BUTTON")
        return self.public_status(refresh=True)

    def toggle_arm(self, *, ts: str) -> dict[str, Any]:
        state = self.store.real_state()
        if bool(state.get("armed")):
            return self.disarm(ts=ts, reason="quick_toggle")
        return self.quick_arm(ts=ts)

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
        if self.cfg.real_portfolio_mode == "account_balances":
            self._execute_account_balance_buy(ts=ts, candle_open_time=candle_open_time, reason=reason)
            return
        state = self.store.real_state()
        remaining_cap = self.cfg.real_max_total_usd - float(state["bot_cost_usd"])
        planned_quote = max(float(planned_usd), self.cfg.real_min_order_usd)
        requested_usd = min(planned_quote, self.cfg.real_max_order_usd, remaining_cap)
        if requested_usd <= 0.0:
            self._record_skip(ts, candle_open_time, "buy", "BUY", requested_usd, 0.0, "real_cap_exhausted")
            return
        assert self.executor is not None
        try:
            try:
                self.executor.product_snapshot()
            except Exception as exc:  # noqa: BLE001 - unsupported products must not place live orders.
                self._record_error(ts, candle_open_time, "buy", "BUY", requested_usd, 0.0, str(exc))
                self._disarm_on_uncertain_order(ts, str(exc))
                return
            balances = self.executor.available_balances()
            _base_currency, quote_currency = _cfg_base_quote(self.cfg)
            available_quote = balances.get(quote_currency, 0.0)
            if available_quote <= 0.0:
                self._record_skip(
                    ts,
                    candle_open_time,
                    "buy",
                    "BUY",
                    requested_usd,
                    0.0,
                    f"insufficient_{quote_currency.lower()}_balance_available_0",
                )
                return
            requested_usd = min(requested_usd, available_quote)
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
        if self.cfg.real_portfolio_mode == "account_balances":
            self._execute_account_balance_sell_all(ts=ts, candle_open_time=candle_open_time, reason=reason)
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

    def _execute_account_balance_buy(self, *, ts: str, candle_open_time: str, reason: str) -> None:
        assert self.executor is not None
        try:
            self.executor.product_snapshot()
            _base_currency, quote_currency = _cfg_base_quote(self.cfg)
            balances = self.executor.available_balances()
            available_quote = balances.get(quote_currency, 0.0)
            if available_quote <= 0.0:
                self._record_skip(ts, candle_open_time, "buy_all_usd", "BUY", 0.0, 0.0, f"no_{quote_currency.lower()}_cash_available")
                return
            requested_quote = min(available_quote, self.cfg.real_max_order_usd)
            if requested_quote < self.cfg.real_min_order_usd:
                self._record_skip(
                    ts,
                    candle_open_time,
                    "buy_capped_usd",
                    "BUY",
                    requested_quote,
                    0.0,
                    f"below_min_order_{quote_currency.lower()}_available_{available_quote:.2f}",
                )
                return
            result = self.executor.market_buy_quote(requested_quote)
            self._record_result(ts, candle_open_time, "buy_capped_usd", result, reason)
            if result.status in {"unconfirmed", "rejected", "cancelled"}:
                self._disarm_on_uncertain_order(ts, result.error or f"{result.status} buy")
        except Exception as exc:  # noqa: BLE001 - persist and disarm on live errors.
            self._record_error(ts, candle_open_time, "buy_all_usd", "BUY", 0.0, 0.0, str(exc))
            self._disarm_on_uncertain_order(ts, str(exc))

    def _execute_account_balance_sell_all(self, *, ts: str, candle_open_time: str, reason: str) -> None:
        assert self.executor is not None
        try:
            product = self.executor.product_snapshot()
            price = float(product.get("price") or 0.0)
            base_currency, _quote_currency = _cfg_base_quote(self.cfg)
            balances = self.executor.available_balances()
            available_base = balances.get(base_currency, 0.0)
            if available_base <= 0.0:
                self._record_skip(ts, candle_open_time, "sell_all_sol", "SELL", 0.0, 0.0, f"no_{base_currency.lower()}_position_available")
                return
            if price <= 0.0:
                raise RealTradingError("Product price is unavailable; refusing capped account-balance sell")
            requested_base = min(available_base, self.cfg.real_max_order_usd / price)
            if requested_base * price < self.cfg.real_min_order_usd:
                self._record_skip(
                    ts,
                    candle_open_time,
                    "sell_capped_sol",
                    "SELL",
                    0.0,
                    requested_base,
                    f"below_min_order_{base_currency.lower()}_available_{available_base:.8f}",
                )
                return
            result = self.executor.market_sell_base(requested_base)
            self._record_result(ts, candle_open_time, "sell_capped_sol", result, reason)
            if result.status in {"unconfirmed", "rejected", "cancelled"}:
                self._disarm_on_uncertain_order(ts, result.error or f"{result.status} sell")
        except Exception as exc:  # noqa: BLE001 - persist and disarm on live errors.
            self._record_error(ts, candle_open_time, "sell_all_sol", "SELL", 0.0, 0.0, str(exc))
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
        if self.cfg.real_portfolio_mode == "account_balances":
            return f"ARM REAL TRADING {self.product_id} ACCOUNT BALANCES"
        cap = f"{self.cfg.real_max_total_usd:g}"
        return f"ARM REAL TRADING {self.product_id} MAX {cap}"

    def flatten_confirmation_text(self) -> str:
        return f"FLATTEN REAL {self.product_id}"

    def _require_executor(self) -> None:
        if not self.configured:
            raise RealTradingError("Set EXECUTION_MODE=coinbase_live or solana_jupiter_live first")
        if not self.enabled:
            raise RealTradingError("Set REAL_TRADING_ENABLED=true first")
        if self.executor is None:
            self.executor = self._new_executor()

    def _ensure_executor_read_only(self) -> None:
        if not self.configured:
            raise RealTradingError("Set EXECUTION_MODE=coinbase_live or solana_jupiter_live first")
        if self.cfg.execution_mode == "coinbase_live" and (not self.cfg.coinbase_api_key or not self.cfg.coinbase_api_secret):
            raise RealTradingError("Coinbase API credentials are missing")
        if self.cfg.execution_mode == "solana_jupiter_live" and (not self.cfg.solana_rpc_url or not self.cfg.solana_keypair_path):
            raise RealTradingError("Solana RPC URL or keypair path is missing")
        if self.executor is None:
            self.executor = self._new_executor()

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
                "product_id": self.product_id,
                "execution_source": self.execution_source,
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
                "product_id": self.product_id,
                "execution_source": self.execution_source,
                "side": side,
                "requested_usd": requested_usd,
                "requested_sol": requested_sol,
                "error": error,
            }
        )

    def _disarm_on_uncertain_order(self, ts: str, error: str) -> None:
        self.store.set_real_armed(armed=False, ts=ts, error=error)
        self.store.insert_event(ts, "error", f"real trading disarmed: {error}")
