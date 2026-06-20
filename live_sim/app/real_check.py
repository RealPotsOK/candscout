"""Validate Coinbase real-trading configuration without placing orders."""

from __future__ import annotations

import json
import sys

from .coinbase_exec import CoinbaseSpotExecutor
from .config import load_config


def main() -> int:
    try:
        cfg = load_config()
        if cfg.execution_mode != "coinbase_live" or not cfg.real_trading_enabled:
            raise RuntimeError(
                "Real trading is disabled. Set EXECUTION_MODE=coinbase_live and REAL_TRADING_ENABLED=true "
                "after adding Coinbase credentials."
            )
        executor = CoinbaseSpotExecutor(cfg)
        status = executor.health_check()
        print(
            json.dumps(
                {
                    "ok": True,
                    "message": "Coinbase real-trading preflight passed. No orders were placed.",
                    "safety": {
                        "product_id": cfg.coinbase_product_id,
                        "trade_mode": cfg.trade_mode,
                        "max_total_usd": cfg.real_max_total_usd,
                        "max_order_usd": cfg.real_max_order_usd,
                        "manual_arm_required": cfg.real_require_manual_arm,
                    },
                    "coinbase": status,
                },
                indent=2,
            )
        )
        return 0
    except Exception as exc:  # noqa: BLE001 - CLI should return clear failure.
        print(
            json.dumps(
                {
                    "ok": False,
                    "error": str(exc),
                    "message": "No orders were placed.",
                },
                indent=2,
            ),
            file=sys.stderr,
        )
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
