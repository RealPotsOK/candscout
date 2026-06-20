"""Entry point for the Docker live paper-trading server."""

from __future__ import annotations

import os

from .bot import LivePaperBot, now_iso
from .coinbase_exec import CoinbaseSpotExecutor, RealTradeService
from .config import load_config
from .market import BinanceClient
from .model_runner import LiveModel
from .scheduler import RetrainScheduler
from .server import AppContext, serve
from .store import Store


def prepare_runtime_env() -> None:
    state_dir = "/app/state"
    cache_dir = f"{state_dir}/.cache"
    for path in [
        cache_dir,
        f"{cache_dir}/torch",
        f"{cache_dir}/torchinductor",
        f"{cache_dir}/triton",
    ]:
        os.makedirs(path, exist_ok=True)
    os.environ.setdefault("HOME", state_dir)
    os.environ.setdefault("USER", "cryptopred")
    os.environ.setdefault("LOGNAME", "cryptopred")
    os.environ.setdefault("XDG_CACHE_HOME", cache_dir)
    os.environ.setdefault("TORCH_HOME", f"{cache_dir}/torch")
    os.environ.setdefault("TORCHINDUCTOR_CACHE_DIR", f"{cache_dir}/torchinductor")
    os.environ.setdefault("TRITON_CACHE_DIR", f"{cache_dir}/triton")
    os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")


def main() -> None:
    prepare_runtime_env()
    cfg = load_config()
    store = Store(cfg.db_path)
    now = now_iso()
    if cfg.reset_on_start:
        store.reset_all(cfg.starting_cash, now)
        store.insert_event(now, "info", "RESET_ON_START=true, runtime state reset")
    else:
        store.initialize_account(cfg.starting_cash, now)

    model = LiveModel(cfg.model_path)
    market = BinanceClient(cfg.binance_base_url)
    real_executor = CoinbaseSpotExecutor(cfg) if cfg.execution_mode == "coinbase_live" else None
    real_trader = RealTradeService(cfg, store, real_executor)
    bot = LivePaperBot(cfg, store, market, model, real_trader)
    retrain_scheduler = RetrainScheduler(cfg, store)
    store.insert_event(now, "info", f"server start symbol={cfg.symbol} interval={cfg.interval}")
    ctx = AppContext(cfg, store, bot, model, retrain_scheduler, real_trader)
    serve(ctx)


if __name__ == "__main__":
    main()
