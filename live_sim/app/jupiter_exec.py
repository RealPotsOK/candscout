"""Solana Jupiter live spot execution for a dedicated burner wallet."""

from __future__ import annotations

import base64
import json
import time
import uuid
from pathlib import Path
from typing import Any

import requests

from .coinbase_exec import RealOrderResult, RealTradingError
from .config import Config

SOL_MINT = "So11111111111111111111111111111111111111112"
USDC_MINT = "EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v"
LAMPORTS_PER_SOL = 1_000_000_000
USDC_DECIMALS = 1_000_000


def _float_or_zero(value: Any) -> float:
    if value is None or value == "":
        return 0.0
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _client_order_id(prefix: str) -> str:
    return f"candscout-jup-{prefix}-{uuid.uuid4().hex[:18]}"


class JupiterSolanaExecutor:
    """Small Jupiter Swap API + Solana RPC adapter.

    The adapter intentionally supports only SOL/USDC spot swaps for a burner wallet.
    """

    def __init__(self, cfg: Config) -> None:
        self.cfg = cfg
        self.jupiter_base_url = cfg.jupiter_base_url.rstrip("/")
        self.session = requests.Session()
        self.keypair = self._load_keypair(Path(cfg.solana_keypair_path))
        self.public_key = str(self.keypair.pubkey())

    def _load_keypair(self, path: Path):
        if not path.exists():
            raise RealTradingError(f"Solana keypair file not found: {path}")
        try:
            from solders.keypair import Keypair  # type: ignore
        except ImportError as exc:  # pragma: no cover - dependency checked in Docker/build.
            raise RealTradingError("solders is not installed. Rebuild live_sim after updating requirements.") from exc
        raw = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(raw, list):
            secret = bytes(int(x) for x in raw)
        elif isinstance(raw, dict):
            value = raw.get("secret_key") or raw.get("private_key") or raw.get("keypair")
            if isinstance(value, str):
                secret = base64.b64decode(value)
            elif isinstance(value, list):
                secret = bytes(int(x) for x in value)
            else:
                raise RealTradingError("Solana keypair JSON must contain a list or secret_key/private_key list/base64")
        else:
            raise RealTradingError("Solana keypair JSON must be a 64-byte array or object")
        try:
            return Keypair.from_bytes(secret)
        except ValueError as exc:
            raise RealTradingError("Solana keypair must be a valid 64-byte keypair") from exc

    def _rpc(self, method: str, params: list[Any]) -> dict[str, Any]:
        payload = {"jsonrpc": "2.0", "id": 1, "method": method, "params": params}
        response = self.session.post(self.cfg.solana_rpc_url, json=payload, timeout=self.cfg.solana_rpc_timeout)
        response.raise_for_status()
        data = response.json()
        if data.get("error"):
            raise RealTradingError(f"Solana RPC {method} failed: {data['error']}")
        return data.get("result", {})

    def _jupiter_get(self, path: str, params: dict[str, Any]) -> dict[str, Any]:
        response = self.session.get(f"{self.jupiter_base_url}/{path.lstrip('/')}", params=params, timeout=self.cfg.jupiter_timeout)
        response.raise_for_status()
        data = response.json()
        if not isinstance(data, dict):
            raise RealTradingError("Jupiter returned a non-object response")
        if data.get("error"):
            raise RealTradingError(f"Jupiter {path} failed: {data['error']}")
        return data

    def _jupiter_post(self, path: str, payload: dict[str, Any]) -> dict[str, Any]:
        response = self.session.post(f"{self.jupiter_base_url}/{path.lstrip('/')}", json=payload, timeout=self.cfg.jupiter_timeout)
        response.raise_for_status()
        data = response.json()
        if not isinstance(data, dict):
            raise RealTradingError("Jupiter returned a non-object response")
        if data.get("error"):
            raise RealTradingError(f"Jupiter {path} failed: {data['error']}")
        return data

    def health_check(self) -> dict[str, Any]:
        balances = self.available_balances()
        product = self.product_snapshot()
        return {
            "ok": True,
            "execution_source": "jupiter_solana",
            "wallet": self.public_key,
            "product_id": self.cfg.jupiter_product_id,
            "product": product,
            "balances": balances,
            "sol_reserved_for_gas": self.cfg.sol_reserved_for_gas,
        }

    def product_snapshot(self) -> dict[str, Any]:
        quote = self.quote(input_mint=SOL_MINT, output_mint=USDC_MINT, amount=LAMPORTS_PER_SOL)
        out_amount = _float_or_zero(quote.get("outAmount")) / USDC_DECIMALS
        return {
            "product_id": self.cfg.jupiter_product_id,
            "price": out_amount,
            "status": "online",
            "base_currency_id": "SOL",
            "quote_currency_id": "USDC",
            "raw": quote,
        }

    def quote(self, *, input_mint: str, output_mint: str, amount: int) -> dict[str, Any]:
        if amount <= 0:
            raise RealTradingError("Jupiter quote amount must be positive")
        return self._jupiter_get(
            "quote",
            {
                "inputMint": input_mint,
                "outputMint": output_mint,
                "amount": str(amount),
                "slippageBps": str(self.cfg.jupiter_slippage_bps),
            },
        )

    def available_balances(self) -> dict[str, float]:
        sol_lamports = int(self._rpc("getBalance", [self.public_key]).get("value", 0))
        sol = max(0.0, sol_lamports / LAMPORTS_PER_SOL - self.cfg.sol_reserved_for_gas)
        usdc = self._token_balance(USDC_MINT, USDC_DECIMALS)
        return {"SOL": sol, "USDC": usdc}

    def account_details(self) -> list[dict[str, Any]]:
        balances = self.available_balances()
        return [
            {"currency": "SOL", "available": balances.get("SOL", 0.0), "hold": 0.0, "uuid": self.public_key},
            {"currency": "USDC", "available": balances.get("USDC", 0.0), "hold": 0.0, "uuid": self.public_key},
        ]

    def _token_balance(self, mint: str, divisor: int) -> float:
        result = self._rpc(
            "getTokenAccountsByOwner",
            [
                self.public_key,
                {"mint": mint},
                {"encoding": "jsonParsed"},
            ],
        )
        total_raw = 0
        for item in result.get("value", []) if isinstance(result.get("value"), list) else []:
            try:
                token_amount = item["account"]["data"]["parsed"]["info"]["tokenAmount"]
                total_raw += int(token_amount.get("amount", "0"))
            except (KeyError, TypeError, ValueError):
                continue
        return total_raw / divisor

    def list_orders_read_only(self, limit: int = 100, *, bot_only: bool = True) -> dict[str, Any]:
        return {"ok": True, "orders": [], "message": "Jupiter order history is stored locally from bot swaps."}

    def market_buy_quote(self, quote_usd: float) -> RealOrderResult:
        amount = int(max(0.0, quote_usd) * USDC_DECIMALS)
        return self._swap(
            action="buy",
            side="BUY",
            input_mint=USDC_MINT,
            output_mint=SOL_MINT,
            amount=amount,
            requested_usd=quote_usd,
            requested_sol=0.0,
        )

    def market_sell_base(self, base_sol: float) -> RealOrderResult:
        amount = int(max(0.0, base_sol) * LAMPORTS_PER_SOL)
        return self._swap(
            action="sell",
            side="SELL",
            input_mint=SOL_MINT,
            output_mint=USDC_MINT,
            amount=amount,
            requested_usd=0.0,
            requested_sol=base_sol,
        )

    def _swap(
        self,
        *,
        action: str,
        side: str,
        input_mint: str,
        output_mint: str,
        amount: int,
        requested_usd: float,
        requested_sol: float,
    ) -> RealOrderResult:
        client_order_id = _client_order_id(action)
        quote = self.quote(input_mint=input_mint, output_mint=output_mint, amount=amount)
        swap = self._build_swap(quote)
        raw_transaction = str(swap.get("swapTransaction") or "")
        if not raw_transaction:
            return self._failed_result(
                side=side,
                client_order_id=client_order_id,
                requested_usd=requested_usd,
                requested_sol=requested_sol,
                error="Jupiter swap response did not include swapTransaction",
                raw_response={"quote": quote, "swap": swap},
                input_mint=input_mint,
                output_mint=output_mint,
                input_amount_raw=amount,
            )
        try:
            signed_transaction = self._sign_transaction(raw_transaction)
            signature = self._send_transaction(signed_transaction)
            confirmation = self._confirm_signature(signature)
        except Exception as exc:  # noqa: BLE001 - convert to persisted failed real order.
            return self._failed_result(
                side=side,
                client_order_id=client_order_id,
                requested_usd=requested_usd,
                requested_sol=requested_sol,
                error=str(exc),
                raw_response={"quote": quote, "swap": swap},
                input_mint=input_mint,
                output_mint=output_mint,
                input_amount_raw=amount,
            )

        out_raw = int(str(quote.get("outAmount", "0")) or "0")
        if side == "BUY":
            filled_usd = requested_usd
            filled_sol = out_raw / LAMPORTS_PER_SOL
        else:
            filled_sol = requested_sol
            filled_usd = out_raw / USDC_DECIMALS
        average_price = (filled_usd / filled_sol) if filled_sol > 0 else None
        fee_lamports = int(confirmation.get("fee_lamports", 0) or 0)
        raw_response = {"quote": quote, "swap": swap, "confirmation": confirmation}
        return RealOrderResult(
            status="filled" if confirmation.get("confirmed") else "submitted",
            product_id=self.cfg.jupiter_product_id,
            side=side,
            client_order_id=client_order_id,
            coinbase_order_id=str(signature),
            requested_usd=requested_usd,
            requested_sol=requested_sol,
            filled_usd=filled_usd,
            filled_sol=filled_sol,
            average_price=average_price,
            fee_usd=fee_lamports / LAMPORTS_PER_SOL * (average_price or 0.0),
            raw_response=raw_response,
            execution_source="jupiter_solana",
            transaction_signature=str(signature),
            input_mint=input_mint,
            output_mint=output_mint,
            input_amount_raw=float(amount),
            expected_output_amount_raw=float(out_raw),
            confirmed_output_amount_raw=float(out_raw),
            network_fee_lamports=float(fee_lamports),
            priority_fee_lamports=_float_or_zero(swap.get("prioritizationFeeLamports")),
            slippage_bps=float(self.cfg.jupiter_slippage_bps),
        )

    def _build_swap(self, quote: dict[str, Any]) -> dict[str, Any]:
        priority_fee: Any
        if self.cfg.jupiter_priority_fee_lamports.lower() == "auto":
            priority_fee = "auto"
        else:
            priority_fee = int(float(self.cfg.jupiter_priority_fee_lamports))
        return self._jupiter_post(
            "swap",
            {
                "quoteResponse": quote,
                "userPublicKey": self.public_key,
                "wrapAndUnwrapSol": True,
                "dynamicComputeUnitLimit": True,
                "prioritizationFeeLamports": priority_fee,
            },
        )

    def _sign_transaction(self, swap_transaction_b64: str) -> str:
        from solders.message import to_bytes_versioned  # type: ignore
        from solders.transaction import VersionedTransaction  # type: ignore

        raw = VersionedTransaction.from_bytes(base64.b64decode(swap_transaction_b64))
        signature = self.keypair.sign_message(to_bytes_versioned(raw.message))
        signed = VersionedTransaction.populate(raw.message, [signature])
        return base64.b64encode(bytes(signed)).decode("ascii")

    def _send_transaction(self, signed_transaction_b64: str) -> str:
        result = self._rpc(
            "sendTransaction",
            [
                signed_transaction_b64,
                {
                    "encoding": "base64",
                    "skipPreflight": False,
                    "maxRetries": 3,
                    "preflightCommitment": "confirmed",
                },
            ],
        )
        if not isinstance(result, str) or not result:
            raise RealTradingError(f"Unexpected sendTransaction response: {result!r}")
        return result

    def _confirm_signature(self, signature: str) -> dict[str, Any]:
        last_status: dict[str, Any] | None = None
        for _ in range(max(1, self.cfg.solana_confirm_polls)):
            status_result = self._rpc("getSignatureStatuses", [[signature], {"searchTransactionHistory": True}])
            values = status_result.get("value") if isinstance(status_result, dict) else None
            status = values[0] if isinstance(values, list) and values else None
            if isinstance(status, dict):
                last_status = status
                if status.get("err"):
                    raise RealTradingError(f"Solana transaction failed: {status.get('err')}")
                if status.get("confirmationStatus") in {"confirmed", "finalized"}:
                    fee_lamports = self._transaction_fee_lamports(signature)
                    return {"confirmed": True, "signature": signature, "status": status, "fee_lamports": fee_lamports}
            time.sleep(self.cfg.solana_confirm_delay_seconds)
        return {"confirmed": False, "signature": signature, "status": last_status, "fee_lamports": self._transaction_fee_lamports(signature)}

    def _transaction_fee_lamports(self, signature: str) -> int:
        try:
            result = self._rpc(
                "getTransaction",
                [signature, {"encoding": "json", "maxSupportedTransactionVersion": 0}],
            )
            meta = result.get("meta") if isinstance(result, dict) else None
            return int(meta.get("fee", 0)) if isinstance(meta, dict) else 0
        except Exception:
            return 0

    def _failed_result(
        self,
        *,
        side: str,
        client_order_id: str,
        requested_usd: float,
        requested_sol: float,
        error: str,
        raw_response: dict[str, Any],
        input_mint: str,
        output_mint: str,
        input_amount_raw: int,
    ) -> RealOrderResult:
        return RealOrderResult(
            status="failed",
            product_id=self.cfg.jupiter_product_id,
            side=side,
            client_order_id=client_order_id,
            coinbase_order_id=None,
            requested_usd=requested_usd,
            requested_sol=requested_sol,
            filled_usd=0.0,
            filled_sol=0.0,
            average_price=None,
            fee_usd=0.0,
            raw_response=raw_response,
            error=error,
            execution_source="jupiter_solana",
            input_mint=input_mint,
            output_mint=output_mint,
            input_amount_raw=float(input_amount_raw),
            slippage_bps=float(self.cfg.jupiter_slippage_bps),
        )
