"""Configuration loading for the live paper-trading simulator."""

from __future__ import annotations

import os
from dataclasses import dataclass, asdict


@dataclass(frozen=True)
class Config:
    symbol: str
    interval: str
    binance_base_url: str
    starting_cash: float
    fee: float
    slippage: float
    entry_threshold: float
    exit_threshold: float
    trade_mode: str
    short_entry_threshold: float
    short_exit_threshold: float
    stop_loss: float
    take_profit: float
    max_hold_bars: int
    max_invest: str
    max_short_invest: str
    allow_flip_position: bool
    borrow_fee: float
    leverage: float
    liquidation_simulation: str
    min_invest: float
    confidence_multiplier: float
    model_path: str
    db_path: str
    reset_on_start: bool
    allow_reset_api: bool
    poll_on_start: bool
    poll_delay_seconds: float
    kline_limit_buffer: int
    catchup_enabled: bool
    catchup_spread_pct: float
    catchup_max_bars: int
    catchup_retry_seconds: float
    retrain_enabled: bool
    retrain_time_utc: str
    retrain_frequency: str
    retrain_lookback_days: int
    retrain_train_start: str
    retrain_train_end: str
    retrain_on_start: bool
    retrain_keep_runs: int
    retrain_cache_dir: str
    training_runs_dir: str
    train_model_type: str
    train_backend: str
    train_device: str
    train_lookback: int
    train_sequence_feature_set: str
    train_edge: float
    train_split: float
    train_cnn_filters: str
    train_cnn_kernel_sizes: str
    train_lstm_hidden_size: int
    train_lstm_layers: int
    train_lstm_dropout: float
    train_gru_hidden_size: int
    train_gru_layers: int
    train_gru_dropout: float
    train_transformer_d_model: int
    train_transformer_heads: int
    train_transformer_layers: int
    train_transformer_ff_dim: int
    train_transformer_dropout: float
    train_hidden_layers: str
    train_lr: float
    train_epochs: int
    train_batch_size: int
    train_l2: float
    train_decision_threshold: float
    train_threshold_grid: str
    train_optimize_metric: str
    train_class_weight_mode: str
    train_seed: int
    train_use_full_window: bool
    execution_mode: str
    real_trading_enabled: bool
    real_require_manual_arm: bool
    real_max_total_usd: float
    real_max_order_usd: float
    real_min_order_usd: float
    coinbase_product_id: str
    coinbase_api_key: str
    coinbase_api_secret: str
    coinbase_timeout: float
    real_arm_token: str
    real_order_status_polls: int
    real_order_status_delay_seconds: float
    host: str
    port: int

    @property
    def interval_seconds(self) -> int:
        return parse_interval_seconds(self.interval)

    def public_dict(self) -> dict:
        data = asdict(self)
        data["interval_seconds"] = self.interval_seconds
        for key in ["coinbase_api_key", "coinbase_api_secret", "real_arm_token"]:
            data[key] = "present" if data.get(key) else ""
        return data


def env_str(name: str, default: str) -> str:
    return os.environ.get(name, default).strip()


def env_float(name: str, default: float) -> float:
    raw = os.environ.get(name)
    if raw is None or raw.strip() == "":
        return default
    try:
        return float(raw)
    except ValueError as exc:
        raise ValueError(f"{name} must be a number, got {raw!r}") from exc


def env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None or raw.strip() == "":
        return default
    try:
        return int(raw)
    except ValueError as exc:
        raise ValueError(f"{name} must be an integer, got {raw!r}") from exc


def env_bool(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None or raw.strip() == "":
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def parse_interval_seconds(interval: str) -> int:
    raw = interval.strip().lower()
    if len(raw) < 2:
        raise ValueError(f"Invalid interval: {interval!r}")
    unit = raw[-1]
    try:
        value = int(raw[:-1])
    except ValueError as exc:
        raise ValueError(f"Invalid interval: {interval!r}") from exc
    if value <= 0:
        raise ValueError("Interval value must be positive")
    multipliers = {"s": 1, "m": 60, "h": 3600, "d": 86400}
    if unit not in multipliers:
        raise ValueError(f"Unsupported interval unit in {interval!r}; use s/m/h/d")
    return value * multipliers[unit]


def load_config() -> Config:
    cfg = Config(
        symbol=env_str("SYMBOL", "SOLUSDT").upper(),
        interval=env_str("INTERVAL", "5m"),
        binance_base_url=env_str("BINANCE_BASE_URL", "https://api.binance.com").rstrip("/"),
        starting_cash=env_float("STARTING_CASH", 100.0),
        fee=env_float("FEE", 0.0001),
        slippage=env_float("SLIPPAGE", 0.0),
        entry_threshold=env_float("ENTRY_THRESHOLD", 0.55),
        exit_threshold=env_float("EXIT_THRESHOLD", 0.48),
        trade_mode=env_str("TRADE_MODE", "long_only"),
        short_entry_threshold=env_float("SHORT_ENTRY_THRESHOLD", 0.55),
        short_exit_threshold=env_float("SHORT_EXIT_THRESHOLD", 0.48),
        stop_loss=env_float("STOP_LOSS", 0.002),
        take_profit=env_float("TAKE_PROFIT", 0.004),
        max_hold_bars=env_int("MAX_HOLD_BARS", 60),
        max_invest=env_str("MAX_INVEST", "m"),
        max_short_invest=env_str("MAX_SHORT_INVEST", env_str("MAX_INVEST", "m")),
        allow_flip_position=env_bool("ALLOW_FLIP_POSITION", False),
        borrow_fee=env_float("BORROW_FEE", 0.0),
        leverage=env_float("LEVERAGE", 1.0),
        liquidation_simulation=env_str("LIQUIDATION_SIMULATION", "off"),
        min_invest=env_float("MIN_INVEST", 1.0),
        confidence_multiplier=env_float("CONFIDENCE_MULTIPLIER", 1.0),
        model_path=env_str("MODEL_PATH", "/app/state/model.npz"),
        db_path=env_str("DB_PATH", "/app/state/live_sim.db"),
        reset_on_start=env_bool("RESET_ON_START", False),
        allow_reset_api=env_bool("ALLOW_RESET_API", False),
        poll_on_start=env_bool("POLL_ON_START", True),
        poll_delay_seconds=env_float("POLL_DELAY_SECONDS", 8.0),
        kline_limit_buffer=env_int("KLINE_LIMIT_BUFFER", 8),
        catchup_enabled=env_bool("CATCHUP_ENABLED", True),
        catchup_spread_pct=env_float("CATCHUP_SPREAD_PCT", 0.00015),
        catchup_max_bars=env_int("CATCHUP_MAX_BARS", 0),
        catchup_retry_seconds=env_float("CATCHUP_RETRY_SECONDS", 60.0),
        retrain_enabled=env_bool("RETRAIN_ENABLED", True),
        retrain_time_utc=env_str("RETRAIN_TIME_UTC", "04:00"),
        retrain_frequency=env_str("RETRAIN_FREQUENCY", "1d"),
        retrain_lookback_days=env_int("RETRAIN_LOOKBACK_DAYS", 365),
        retrain_train_start=env_str("RETRAIN_TRAIN_START", ""),
        retrain_train_end=env_str("RETRAIN_TRAIN_END", ""),
        retrain_on_start=env_bool("RETRAIN_ON_START", False),
        retrain_keep_runs=env_int("RETRAIN_KEEP_RUNS", 10),
        retrain_cache_dir=env_str("RETRAIN_CACHE_DIR", "/app/state/downloads"),
        training_runs_dir=env_str("TRAINING_RUNS_DIR", "/app/state/training_runs"),
        train_model_type=env_str("TRAIN_MODEL_TYPE", "cnn"),
        train_backend=env_str("TRAIN_BACKEND", "torch"),
        train_device=env_str("TRAIN_DEVICE", "cuda"),
        train_lookback=env_int("TRAIN_LOOKBACK", 30),
        train_sequence_feature_set=env_str("TRAIN_SEQUENCE_FEATURE_SET", "basic"),
        train_edge=env_float("TRAIN_EDGE", 0.0003),
        train_split=env_float("TRAIN_SPLIT", 0.95),
        train_cnn_filters=env_str("TRAIN_CNN_FILTERS", "16,32"),
        train_cnn_kernel_sizes=env_str("TRAIN_CNN_KERNEL_SIZES", "5,3"),
        train_lstm_hidden_size=env_int("TRAIN_LSTM_HIDDEN_SIZE", 64),
        train_lstm_layers=env_int("TRAIN_LSTM_LAYERS", 1),
        train_lstm_dropout=env_float("TRAIN_LSTM_DROPOUT", 0.0),
        train_gru_hidden_size=env_int("TRAIN_GRU_HIDDEN_SIZE", 64),
        train_gru_layers=env_int("TRAIN_GRU_LAYERS", 1),
        train_gru_dropout=env_float("TRAIN_GRU_DROPOUT", 0.0),
        train_transformer_d_model=env_int("TRAIN_TRANSFORMER_D_MODEL", 64),
        train_transformer_heads=env_int("TRAIN_TRANSFORMER_HEADS", 4),
        train_transformer_layers=env_int("TRAIN_TRANSFORMER_LAYERS", 2),
        train_transformer_ff_dim=env_int("TRAIN_TRANSFORMER_FF_DIM", 128),
        train_transformer_dropout=env_float("TRAIN_TRANSFORMER_DROPOUT", 0.1),
        train_hidden_layers=env_str("TRAIN_HIDDEN_LAYERS", "32,16"),
        train_lr=env_float("TRAIN_LR", 0.001),
        train_epochs=env_int("TRAIN_EPOCHS", 140),
        train_batch_size=env_int("TRAIN_BATCH_SIZE", 2048),
        train_l2=env_float("TRAIN_L2", 0.0001),
        train_decision_threshold=env_float("TRAIN_DECISION_THRESHOLD", env_float("ENTRY_THRESHOLD", 0.55)),
        train_threshold_grid=env_str("TRAIN_THRESHOLD_GRID", "0.50,0.55,0.60,0.65,0.70,0.75,0.80,0.85,0.90,0.95,0.99"),
        train_optimize_metric=env_str("TRAIN_OPTIMIZE_METRIC", "f1_y1"),
        train_class_weight_mode=env_str("TRAIN_CLASS_WEIGHT_MODE", "balanced"),
        train_seed=env_int("TRAIN_SEED", 18),
        train_use_full_window=env_bool("TRAIN_USE_FULL_WINDOW", True),
        execution_mode=env_str("EXECUTION_MODE", "paper").lower(),
        real_trading_enabled=env_bool("REAL_TRADING_ENABLED", False),
        real_require_manual_arm=env_bool("REAL_REQUIRE_MANUAL_ARM", True),
        real_max_total_usd=env_float("REAL_MAX_TOTAL_USD", 20.0),
        real_max_order_usd=env_float("REAL_MAX_ORDER_USD", 5.0),
        real_min_order_usd=env_float("REAL_MIN_ORDER_USD", 1.0),
        coinbase_product_id=env_str("COINBASE_PRODUCT_ID", "SOL-USD").upper(),
        coinbase_api_key=env_str("COINBASE_API_KEY", ""),
        coinbase_api_secret=env_str("COINBASE_API_SECRET", ""),
        coinbase_timeout=env_float("COINBASE_TIMEOUT", 10.0),
        real_arm_token=env_str("REAL_ARM_TOKEN", ""),
        real_order_status_polls=env_int("REAL_ORDER_STATUS_POLLS", 5),
        real_order_status_delay_seconds=env_float("REAL_ORDER_STATUS_DELAY_SECONDS", 0.75),
        host=env_str("HOST", "0.0.0.0"),
        port=env_int("PORT", 8080),
    )
    validate_config(cfg)
    return cfg


def validate_config(cfg: Config) -> None:
    if cfg.starting_cash <= 0:
        raise ValueError("STARTING_CASH must be positive")
    if cfg.fee < 0:
        raise ValueError("FEE cannot be negative")
    if cfg.slippage < 0:
        raise ValueError("SLIPPAGE cannot be negative")
    if not 0.0 <= cfg.entry_threshold <= 1.0:
        raise ValueError("ENTRY_THRESHOLD must be between 0 and 1")
    if not 0.0 <= cfg.exit_threshold <= 1.0:
        raise ValueError("EXIT_THRESHOLD must be between 0 and 1")
    if cfg.trade_mode not in {"long_only", "short_only", "long_short"}:
        raise ValueError("TRADE_MODE must be long_only, short_only, or long_short")
    if not 0.0 <= cfg.short_entry_threshold <= 1.0:
        raise ValueError("SHORT_ENTRY_THRESHOLD must be between 0 and 1")
    if not 0.0 <= cfg.short_exit_threshold <= 1.0:
        raise ValueError("SHORT_EXIT_THRESHOLD must be between 0 and 1")
    if cfg.borrow_fee < 0:
        raise ValueError("BORROW_FEE cannot be negative")
    if cfg.leverage != 1.0:
        raise ValueError("Only LEVERAGE=1 is supported")
    if cfg.liquidation_simulation not in {"off", "basic"}:
        raise ValueError("LIQUIDATION_SIMULATION must be off or basic")
    if cfg.stop_loss < 0 or cfg.take_profit < 0:
        raise ValueError("STOP_LOSS and TAKE_PROFIT cannot be negative")
    if cfg.max_hold_bars <= 0:
        raise ValueError("MAX_HOLD_BARS must be positive")
    if cfg.min_invest < 0:
        raise ValueError("MIN_INVEST cannot be negative")
    if cfg.confidence_multiplier <= 0:
        raise ValueError("CONFIDENCE_MULTIPLIER must be positive")
    if cfg.kline_limit_buffer < 1:
        raise ValueError("KLINE_LIMIT_BUFFER must be positive")
    if not 0.0 <= cfg.catchup_spread_pct < 2.0:
        raise ValueError("CATCHUP_SPREAD_PCT must be >= 0 and < 2")
    if cfg.catchup_max_bars < 0:
        raise ValueError("CATCHUP_MAX_BARS must be zero (unlimited) or positive")
    if cfg.catchup_retry_seconds <= 0:
        raise ValueError("CATCHUP_RETRY_SECONDS must be positive")
    if cfg.retrain_lookback_days <= 0:
        raise ValueError("RETRAIN_LOOKBACK_DAYS must be positive")
    if cfg.retrain_keep_runs < 1:
        raise ValueError("RETRAIN_KEEP_RUNS must be at least 1")
    parse_retrain_frequency(cfg.retrain_frequency)
    if bool(cfg.retrain_train_start) != bool(cfg.retrain_train_end):
        raise ValueError("Set both RETRAIN_TRAIN_START and RETRAIN_TRAIN_END, or leave both blank")
    if cfg.train_model_type not in {"cnn", "mlp", "gru", "lstm", "transformer"}:
        raise ValueError("TRAIN_MODEL_TYPE must be cnn, mlp, gru, lstm, or transformer")
    if cfg.train_backend not in {"numpy", "torch"}:
        raise ValueError("TRAIN_BACKEND must be numpy or torch")
    if cfg.train_model_type in {"gru", "lstm", "transformer"} and cfg.train_backend != "torch":
        raise ValueError(f"TRAIN_MODEL_TYPE={cfg.train_model_type} requires TRAIN_BACKEND=torch")
    if cfg.train_lookback < 2:
        raise ValueError("TRAIN_LOOKBACK must be at least 2")
    if cfg.train_sequence_feature_set not in {"basic", "technical"}:
        raise ValueError("TRAIN_SEQUENCE_FEATURE_SET must be basic or technical")
    if not 0.0 < cfg.train_split < 1.0:
        raise ValueError("TRAIN_SPLIT must be between 0 and 1")
    if cfg.train_lstm_hidden_size <= 0:
        raise ValueError("TRAIN_LSTM_HIDDEN_SIZE must be positive")
    if cfg.train_lstm_layers <= 0:
        raise ValueError("TRAIN_LSTM_LAYERS must be positive")
    if not 0.0 <= cfg.train_lstm_dropout < 1.0:
        raise ValueError("TRAIN_LSTM_DROPOUT must be >= 0 and < 1")
    if cfg.train_gru_hidden_size <= 0:
        raise ValueError("TRAIN_GRU_HIDDEN_SIZE must be positive")
    if cfg.train_gru_layers <= 0:
        raise ValueError("TRAIN_GRU_LAYERS must be positive")
    if not 0.0 <= cfg.train_gru_dropout < 1.0:
        raise ValueError("TRAIN_GRU_DROPOUT must be >= 0 and < 1")
    if cfg.train_transformer_d_model <= 0:
        raise ValueError("TRAIN_TRANSFORMER_D_MODEL must be positive")
    if cfg.train_transformer_heads <= 0:
        raise ValueError("TRAIN_TRANSFORMER_HEADS must be positive")
    if cfg.train_transformer_d_model % cfg.train_transformer_heads != 0:
        raise ValueError("TRAIN_TRANSFORMER_D_MODEL must be divisible by TRAIN_TRANSFORMER_HEADS")
    if cfg.train_transformer_layers <= 0:
        raise ValueError("TRAIN_TRANSFORMER_LAYERS must be positive")
    if cfg.train_transformer_ff_dim <= 0:
        raise ValueError("TRAIN_TRANSFORMER_FF_DIM must be positive")
    if not 0.0 <= cfg.train_transformer_dropout < 1.0:
        raise ValueError("TRAIN_TRANSFORMER_DROPOUT must be >= 0 and < 1")
    if cfg.train_lr <= 0:
        raise ValueError("TRAIN_LR must be positive")
    if cfg.train_epochs <= 0:
        raise ValueError("TRAIN_EPOCHS must be positive")
    if cfg.train_batch_size <= 0:
        raise ValueError("TRAIN_BATCH_SIZE must be positive")
    if cfg.train_l2 < 0:
        raise ValueError("TRAIN_L2 cannot be negative")
    if cfg.train_class_weight_mode not in {"none", "balanced", "manual"}:
        raise ValueError("TRAIN_CLASS_WEIGHT_MODE must be none, balanced, or manual")
    if cfg.execution_mode not in {"paper", "coinbase_live"}:
        raise ValueError("EXECUTION_MODE must be paper or coinbase_live")
    if cfg.real_max_total_usd <= 0:
        raise ValueError("REAL_MAX_TOTAL_USD must be positive")
    if cfg.real_max_total_usd > 20:
        raise ValueError("REAL_MAX_TOTAL_USD must be <= 20 for v1 real trading")
    if cfg.real_max_order_usd <= 0:
        raise ValueError("REAL_MAX_ORDER_USD must be positive")
    if cfg.real_max_order_usd > cfg.real_max_total_usd:
        raise ValueError("REAL_MAX_ORDER_USD cannot exceed REAL_MAX_TOTAL_USD")
    if cfg.real_min_order_usd <= 0:
        raise ValueError("REAL_MIN_ORDER_USD must be positive")
    if cfg.real_min_order_usd > cfg.real_max_order_usd:
        raise ValueError("REAL_MIN_ORDER_USD cannot exceed REAL_MAX_ORDER_USD")
    if cfg.coinbase_timeout <= 0:
        raise ValueError("COINBASE_TIMEOUT must be positive")
    if cfg.real_order_status_polls < 0:
        raise ValueError("REAL_ORDER_STATUS_POLLS cannot be negative")
    if cfg.real_order_status_delay_seconds < 0:
        raise ValueError("REAL_ORDER_STATUS_DELAY_SECONDS cannot be negative")
    if cfg.execution_mode == "coinbase_live" or cfg.real_trading_enabled:
        if cfg.execution_mode != "coinbase_live":
            raise ValueError("REAL_TRADING_ENABLED=true requires EXECUTION_MODE=coinbase_live")
        if cfg.symbol != "SOLUSDT":
            raise ValueError("Coinbase real trading v1 only supports SYMBOL=SOLUSDT")
        if cfg.coinbase_product_id != "SOL-USD":
            raise ValueError("Coinbase real trading v1 only supports COINBASE_PRODUCT_ID=SOL-USD")
        if cfg.trade_mode != "long_only":
            raise ValueError("Real trading v1 requires TRADE_MODE=long_only")
        if cfg.borrow_fee != 0.0:
            raise ValueError("Real trading v1 requires BORROW_FEE=0")
        if cfg.leverage != 1.0:
            raise ValueError("Real trading v1 requires LEVERAGE=1")
        if cfg.liquidation_simulation != "off":
            raise ValueError("Real trading v1 requires LIQUIDATION_SIMULATION=off")
        if not cfg.real_require_manual_arm:
            raise ValueError("Real trading v1 requires REAL_REQUIRE_MANUAL_ARM=true")
        if not cfg.coinbase_api_key or not cfg.coinbase_api_secret:
            raise ValueError("Coinbase real trading requires COINBASE_API_KEY and COINBASE_API_SECRET")
        if cfg.real_trading_enabled and not cfg.real_arm_token:
            raise ValueError("REAL_TRADING_ENABLED=true requires REAL_ARM_TOKEN")
    if cfg.train_optimize_metric not in {"f1_y1", "recall_y1", "precision_y1", "accuracy"}:
        raise ValueError("TRAIN_OPTIMIZE_METRIC is invalid")
    parse_hhmm(cfg.retrain_time_utc)
    parse_interval_seconds(cfg.interval)


def parse_hhmm(raw: str) -> tuple[int, int]:
    parts = raw.strip().split(":")
    if len(parts) != 2:
        raise ValueError(f"Expected HH:MM UTC time, got {raw!r}")
    try:
        hour = int(parts[0])
        minute = int(parts[1])
    except ValueError as exc:
        raise ValueError(f"Expected HH:MM UTC time, got {raw!r}") from exc
    if not 0 <= hour <= 23 or not 0 <= minute <= 59:
        raise ValueError(f"Invalid HH:MM UTC time: {raw!r}")
    return hour, minute


def parse_retrain_frequency(raw: str) -> tuple[int, str]:
    value = raw.strip().lower()
    if not value:
        raise ValueError("RETRAIN_FREQUENCY cannot be blank")
    aliases = {
        "month": "m",
        "months": "m",
        "mo": "m",
        "mon": "m",
        "week": "w",
        "weeks": "w",
        "day": "d",
        "days": "d",
        "hour": "h",
        "hours": "h",
    }
    unit = ""
    number = ""
    for suffix in sorted([*aliases.keys(), "m", "w", "d", "h"], key=len, reverse=True):
        if value.endswith(suffix):
            unit = aliases.get(suffix, suffix)
            number = value[: -len(suffix)]
            break
    if not unit:
        raise ValueError("RETRAIN_FREQUENCY must use h/d/w/m, e.g. 10h, 3d, 1w, 1m")
    try:
        amount = int(number)
    except ValueError as exc:
        raise ValueError(f"Invalid RETRAIN_FREQUENCY value: {raw!r}") from exc
    if amount <= 0:
        raise ValueError("RETRAIN_FREQUENCY amount must be positive")
    return amount, unit
